"""GB direct-to-WDM LOOKUP scorer (the fused ``gb_lookup_get_ll`` kernel).

Reference-free GB likelihood: each row's WDM coefficients are read straight from
the n_ref lookup table, so there is no heterodyne reference, anchor offset,
stash or refresh -- a drop-in for the chunked delegate's ``get_ll_wdm`` (and the
band engine's ``gb_wdm_comp``). Fills and every other method delegate to the
chunked computation the scorer is built around.

Per row (one CUDA block): the GB response at ``n_nodes`` control points, each
channel's envelope demodulated by the COMMON reference phase splined in Re / Im
(linear in the channels, so the low-f X+Y+Z null survives) plus that phase;
then per active pixel ONE table read per layer at the common carrier
``(f_ref, fdot_ref)`` and the amplitude-slope (K1) term. Python prototype and
design notes: LAT ``scripts/gb/gb_lookup_scorer.py``.

The table must be built at the run's layer duration (the GB recipe:
``scripts/gb/_gb_testbox.py::GB_TABLE_RECIPE``, a 128-layer build record and a
narrow fdot axis -- the shared 32-layer EMRI table carries a ~2e-5 norm bias).

The same fill also gives the reference-free Fisher / Gram information matrix
(:meth:`GBLookupComputations.information_matrix`, the sig-het
``SIGHET_INFOMAT_ENGINE=lookup`` route and LAT's ``GB_CHOL_CACHE`` factors).
"""

from __future__ import annotations

import os

import numpy as np

from lisatools.response.tdionfly import GBTDIonTheFly

from .parallelbase import GBGPUParallelModule


def _host(a):
    """``a`` on the host: cupy arrays through ``.get()`` (``np.asarray`` refuses them)."""
    return a.get() if hasattr(a, "get") else a


class GBLookupComputations(GBGPUParallelModule):
    """Lookup scorer around a chunked ``GBWDMComputations`` (grid, orbits, TDI, t_ref).

    Args:
        chunked_comp: the chunked delegate (``gbgpu.gbcomps.GBWDMComputations``); its
            backend, WDM grid, orbits, TDI configuration and ``t_ref`` are used.
        table: a ``lisatools.domains.WDMLookupTable`` (or a path to one) built at the
            grid's layer duration.
        n_nodes: response control points per row over the active span; <= 0 = AUTO, a
            constant node spacing of ``GB_LOOKUP_NODE_SPACING_DAYS`` (default 2.8 d, i.e.
            64 nodes at 6 months; never fewer than 64). A fixed count loses accuracy
            with the span: at 540 d, 64 nodes left 1e-5 relative h_h error vs chunked,
            192 nodes 3e-7 (the 6-month floor).
        num_m_layers: layers each side of the carrier layer (2 -> 5 layers).
        k1: include the amplitude-slope term.
        fill: ``fill_global_wdm`` writes the lookup template (``fill_lookup``) instead of
            delegating to the chunked fill.
        k_coarse: read the table only at the control points and interpolate each layer's
            (smooth) table value in time at the pixels (Mike, 10-04: "lookup computed over
            large steps in pixels ... spline the rotated representation"); a layer whose
            table support starts or ends inside the span keeps per-pixel reads.
    """

    def __init__(self, chunked_comp, table, *, n_nodes=-1, num_m_layers=2, k1=True,
                 k_coarse=True, fill=False):
        flavor = chunked_comp.backend.name.split("_", 1)[1]
        GBGPUParallelModule.__init__(self, force_backend=flavor)
        from lisatools.domains import WDMLookupTable
        from lisatools.wdm_lookup_eval import WDMLookupEvaluator

        self.chunked = chunked_comp
        wdm = chunked_comp.wdm_settings
        self.wdm = wdm
        if isinstance(table, str):
            table = WDMLookupTable.from_file(table, force_backend=flavor)
        if abs(float(table.layer_dt) - float(wdm.layer_dt)) > 1e-6 * float(wdm.layer_dt):
            raise ValueError(f"lookup table layer_dt {table.layer_dt} != grid layer_dt "
                             f"{wdm.layer_dt}")
        self.ev = WDMLookupEvaluator(table, interp="spline", force_backend=flavor)
        self.cpp = self.backend.GBComputationGroupWrap()
        if int(n_nodes) <= 0:
            spacing = float(os.environ.get("GB_LOOKUP_NODE_SPACING_DAYS", "2.8")) * 86400.0
            span = (int(wdm.Nt_active) + 1) * float(wdm.layer_dt)
            n_nodes = max(64, int(np.ceil(span / spacing)) + 1)
        self.n_nodes = int(n_nodes)
        self.num_m_layers = int(num_m_layers)
        self.k1 = bool(k1)
        self.k_coarse = bool(k_coarse)
        self.fill = bool(fill)
        self.d_d = float(getattr(chunked_comp, "d_d", 0.0))

        dt = float(wdm.data_dt)
        t0 = float(wdm.t0)
        Tobs = float(wdm.Tobs)
        Nobs = int(wdm.Nf) * int(wdm.Nt)
        self.layer_dt = float(wdm.layer_dt)
        self.layer_df = float(wdm.layer_df)
        self.Tobs = Tobs            # (s) scales the information-matrix phase-coefficient steps
        self.t0 = t0
        self.ind_min_t = int(wdm.ind_min_t)
        self.Nt_active = int(wdm.Nt_active)
        self.ind_min_f = int(wdm.ind_min_f)
        self.ind_max_f = int(wdm.ind_max_f)
        self.Nf_active = self.ind_max_f - self.ind_min_f + 1
        # control points span the analysed pixels plus one layer each side
        t_first = t0 + self.ind_min_t * self.layer_dt
        t_last = t0 + (self.ind_min_t + self.Nt_active - 1) * self.layer_dt
        self.t_node0 = t_first - self.layer_dt
        self.dt_node = (t_last + self.layer_dt - self.t_node0) / (self.n_nodes - 1)

        t_tdi = np.linspace(t0, t0 + (Nobs - 1) * dt, 16384)
        gb_gen = GBTDIonTheFly(t_tdi, Tobs, float(chunked_comp.t_ref), 1.0 / dt, 1,
                               tdi_config=chunked_comp.tdi_config, orbits=chunked_comp.orbits,
                               tdi_chan="XYZ", force_backend=flavor)
        self.tdi_wrap = gb_gen.wave_gen
        self._keep_alive = dict(gb_gen=gb_gen)

        ev = self.ev
        xp = self.xp
        self._tab = (xp.ascontiguousarray(xp.asarray(ev.tab_cos, dtype=float).reshape(-1)),
                     xp.ascontiguousarray(xp.asarray(ev.tab_sin, dtype=float).reshape(-1)),
                     int(ev.nfdot), int(ev.nf), float(ev.fdot_min), float(ev.dfdot),
                     float(ev.f_min), float(ev.df), float(ev.f_min), float(ev.f_max),
                     int((ev.m_ref + ev.n_ref) % 2), float(ev.fdot_min), float(ev.fdot_max))
        self.d_h_out = self.h_h_out = self.d_h_im_out = None

    # ---- the band-engine contract --------------------------------------------------
    def setup_in_model(self, *args, **kwargs):
        """No reference to build: the lookup is exact at every point."""
        return False

    def clear_in_model(self):
        return None

    def fill_global_wdm(self, *args, **kwargs):
        """Chunked fill (the default), or the lookup fill when built with ``fill=True``."""
        if self.fill:
            return self.fill_lookup(*args, **kwargs)
        return self.chunked.fill_global_wdm(*args, **kwargs)

    def fill_lookup(self, params, templates, convert_to_ra_dec=None, data_index=None,
                    factors=None, band_slab_Nf=None, slab_min_f=None, **kwargs):
        """Add ``factors[row] * h_row`` (the lookup template) into ``templates`` -- the flat
        buffer of slabs ``(n_slots, 3, W, Nt_active)`` (``band_slab_Nf`` + ``slab_min_f``)
        or the full active band. The chunked ``fill_global_wdm`` signature; its
        ``grid_dim`` / ``m_band_half_width`` do not apply (the band is ``num_m_layers``)."""
        if convert_to_ra_dec:
            raise NotImplementedError("the lookup fill takes ICRS params")
        xp = self.xp
        x = xp.ascontiguousarray(xp.atleast_2d(xp.asarray(params, dtype=float)))
        num_bin, nparams = x.shape
        W = self.Nf_active if (band_slab_Nf is None or slab_min_f is None) else int(band_slab_Nf)
        buf = templates
        n_slots = int(buf.size // (3 * W * self.Nt_active))
        di = xp.zeros(num_bin, dtype=xp.int32) if data_index is None else \
            xp.ascontiguousarray(xp.asarray(data_index, dtype=xp.int32))
        fac = xp.ones(num_bin) if factors is None else \
            xp.ascontiguousarray(xp.asarray(factors, dtype=float))
        self.cpp.gb_lookup_fill(
            self.tdi_wrap, buf, fac, x.reshape(-1), di,
            (xp.zeros(0, dtype=xp.int32) if (band_slab_Nf is None or slab_min_f is None)
             else xp.ascontiguousarray(xp.asarray(slab_min_f, dtype=xp.int32))),
            W, n_slots, num_bin, nparams, 3,
            self.n_nodes, self.t_node0, self.dt_node,
            self.t0, self.layer_dt, self.layer_df,
            self.ind_min_t, self.Nt_active, self.ind_min_f, self.ind_max_f,
            self.num_m_layers, int(self.k1), int(self.k_coarse), *self._tab)

    def make_carrier_reference(self, params, w_lo, W, c0_dense=None, c1_dense=None,
                               with_c1=True):
        """The sig-het v5 CARRIER reference straight from the table.

        The carrier-only reference is a unit envelope on the source's common phase --
        exactly the chirping tone the table stores -- so c0 (its complex WDM transform)
        is one table read per (pixel, layer) and the packet first moment c1 is the same
        B-spline's f-derivative (the scorer's K1 read): no FD build, FFT or polyphase.

        ``params`` (n, 9) ICRS; ``w_lo`` (n,) active-local window origins; ``W`` the
        window width. Fills / returns ``c0_dense`` and ``c1_dense`` (n, W, Nt_active)
        complex -- c1 already carries the ``_c1_scale`` normalization of the FD path.
        Only the ``num_m_layers`` band around the carrier is written; the rest is zero.
        """
        xp = self.xp
        x = xp.ascontiguousarray(xp.atleast_2d(xp.asarray(params, dtype=float)))
        num_bin, nparams = x.shape
        W = int(W)
        shape = (num_bin, W, self.Nt_active)
        if c0_dense is None:
            c0_dense = xp.zeros(shape, dtype=xp.complex128)
        if with_c1 and c1_dense is None:
            c1_dense = xp.zeros(shape, dtype=xp.complex128)
        if num_bin == 0:
            return c0_dense, c1_dense
        self.cpp.gb_lookup_carrier_ref(
            self.tdi_wrap, c0_dense,
            c1_dense if with_c1 else xp.zeros(0, dtype=xp.complex128),
            x.reshape(-1), xp.ascontiguousarray(xp.asarray(w_lo, dtype=xp.int32)), W,
            num_bin, nparams, 3,
            self.n_nodes, self.t_node0, self.dt_node,
            self.t0, self.layer_dt, self.layer_df,
            self.ind_min_t, self.Nt_active, self.ind_min_f, self.ind_max_f,
            self.num_m_layers, int(self.k_coarse), *self._tab)
        return c0_dense, (c1_dense if with_c1 else None)

    # ---- information matrix ----------------------------------------------------------
    #: Phase (rad) every default non-amplitude step of :meth:`information_matrix` moves the
    #: template by: the angles step by it, the phase coefficients f0 / fdot / fddot by
    #: ``INFOMAT_PHASE_STEP / Tobs**k``. See :meth:`info_matrix_param_eps`.
    INFOMAT_PHASE_STEP = 1e-4
    #: Default amplitude step (strain). The template is linear in the amplitude, so the
    #: central difference is exact at any step (the chunked delegate's table value).
    INFOMAT_AMP_STEP = 1e-25

    def info_matrix_param_eps(self):
        """Default central-difference steps of :meth:`information_matrix` (host, ``(9,)``).

        Order ``(amp, f0, fdot, fddot, phi0, inc, psi, lam, beta)``; with
        ``s = INFOMAT_PHASE_STEP`` (1e-4) and ``T = Tobs``::

            (1e-25, s / T, s / T**2, s / T**3, s, s, s, s, s)

        Every non-amplitude step moves the template by O(s) over the observation (f0:
        phase up to ``2 pi s``; fdot: up to ``pi s``; inc / psi / phi0 through the
        polarization factors; lam / beta through the Doppler phase, up to ``2 pi f0
        AU/c`` ~ 80 s at 25 mHz). Why the phase coefficients scale with ``Tobs``: a FIXED
        step moves the phase by ``2 pi df T`` / ``pi dfdot T^2`` / ``pi dfddot T^3 / 3``,
        so the chunked delegate's table (``2e-14`` Hz, ``1e-21`` Hz/s, ``1e-28`` Hz/s^2,
        ``1e-6`` rad angles) is a ~1e-9..2e-6 rad phase step at 30 d - 6 months, where
        the lookup's interpolation noise dominates the template difference.

        Measured against the exact-template Gram (central differences of the chunked
        fill, converged to 8e-5 between steps 1e-3 and 1e-4;
        ``tests/test_lookup_information_matrix.py`` fixture, SNR 300, worst element
        ``|dG_ab| / sqrt(G_aa G_bb)``): ``s`` = 1e-3 / 1e-4 -> 8e-5 / 7e-5 at 30 d and
        1.2e-4 at 6 months; ``s`` = 1e-5 -> 1e-3; ``s`` = 1e-2 -> 8e-3 (truncation in
        lam / beta). The chunked table gives 5 at 30 d (fdot-fdot 6x) and 0.27 at 6
        months; its fddot step is off by ~6e2 at 30 d; with only f0 / fdot rescaled
        (the first version of this matrix) the 1e-6 rad angle steps left 7.5e-3 at
        30 d and 2.9e-2 at 6 months.
        """
        s, T = float(self.INFOMAT_PHASE_STEP), float(self.Tobs)
        return np.array([self.INFOMAT_AMP_STEP, s / T, s / T**2, s / T**3, s, s, s, s, s])

    def _info_matrix_steps(self, nparams, param_eps):
        """Host float64 ``(nparams,)`` step table: ``param_eps`` (numpy or a device
        array, converted through ``.get()``) or :meth:`info_matrix_param_eps`."""
        if param_eps is None:
            eps = self.info_matrix_param_eps()
            if eps.size != nparams:
                raise ValueError(f"information_matrix: default steps are for the "
                                 f"{eps.size} GB parameters, params has {nparams} columns; "
                                 "pass param_eps.")
            return eps
        eps = np.array(_host(param_eps), dtype=float).ravel()
        if eps.size != nparams:
            raise ValueError(f"param_eps length {eps.size} != nparams {nparams}")
        return eps

    def information_matrix(self, params, wdm_holder, inds=None, param_eps=None,
                           noise_index=None, max_bytes=256 * 2**20, **kwargs):
        """Fisher (Gram) information matrix ``G_ab = <dh/dtheta_a | dh/dtheta_b>`` from
        lookup templates, per source.

        The inner product is :meth:`get_ll_wdm`'s (plain pixel sums against the XYZ
        inverse covariance, no extra normalization):
        ``<a|b> = sum_{c,d,m,n} a_c[m,n] invC_cd[m,n] b_d[m,n]``, so e.g.
        ``G_AA * A**2 == h_h``. This is the chunked delegate's definition
        (``WDMComputationsBase.information_matrix``) and, at the expansion point of a
        residual that holds exactly the source, the curvature the ``SIGHET_INFOMAT``
        route measures as second differences of ln L. The matrix is in the RAW
        physical parameters and unregularized (it may be near-singular, e.g. the
        amp / inc / psi / phi0 block of a short observation); LAT's
        ``_compute_proposal_cholesky`` maps it to the sampling basis and floors it.

        How: ``dh_a`` by central differences of the lookup template
        (:meth:`fill_lookup`), two fills per parameter per source, each written into a
        narrow per-source slab of ``W = 2 num_m_layers + 4`` layers starting
        ``num_m_layers + 1`` below the carrier layer of ``f0`` (the fill's
        ``num_m_layers`` band each side of the instantaneous carrier, with a layer of
        drift margin below and two above; on the GB table grid the Doppler drift is
        <= 1e-4 f0, ~0.02 layer at 25 mHz), then ONE contraction per batch against each
        source's invC rows. No heterodyne reference, in-model block or buffer slot is
        involved, so any set of sources can be done in one batch (the ``GB_CHOL_CACHE``
        refresh builds every alive source at once). Cost: ``2 len(inds)`` lookup fills
        per source, against 4 chunked swap launches per parameter PAIR on the chunked
        route.

        Steps: :meth:`info_matrix_param_eps` (Tobs-scaled phase coefficients, 1e-4 rad
        angles; why and the measured accuracy there) unless ``param_eps`` is given.
        Accuracy is the lookup template's: ~1e-4 of ``sqrt(G_aa G_bb)`` against the
        exact-template Gram at 30 d - 6 months.

        Device: the fills and the contraction run on this object's backend (numpy on
        CPU, cupy on GPU) and the result stays there. ``params``, ``inds``,
        ``param_eps`` and ``noise_index`` may be host or device arrays; the steps, slab
        origins and invC rows are host bookkeeping (device arrays are read through
        ``.get()``; ``np.asarray`` refuses a cupy array).

        Args:
            params: ``(n, 9)`` (or ``(9,)``) physical ICRS parameters ``(amp, f0 [Hz],
                fdot [Hz/s], fddot [Hz/s^2], phi0, inc, psi, lam, beta)``, angles in rad,
                ``lam`` / ``beta`` = ICRS right ascension / declination.
            wdm_holder: the full-active-band holder (an ``AnalysisContainerArray``, or a
                single ``AnalysisContainer``, wrapped as the chunked comp does) whose
                ``linear_psd_arr[0]`` is ``(rows, 3, 3, Nf_active, Nt_active)`` XYZ invC.
                A shared-psd mirror (``psd_row_index``) maps ``noise_index`` to its
                rows; per-slot narrow invC slabs are refused.
            inds: parameter columns to differentiate, in output order (default all).
            param_eps: ``(nparams,)`` steps (host or device); default above.
            noise_index: ``(n,)`` invC row (walker) per source; default row 0.
            max_bytes: working-set budget per batch of sources (stepped fills,
                derivatives and the gathered invC rows).
            **kwargs: the chunked delegate's swap-kernel knobs (``grid_dim``,
                ``m_band_half_width``, ...), accepted for signature parity and ignored;
                ``convert_to_ra_dec=True`` is refused (ICRS only).

        Returns:
            ``(n, len(inds), len(inds))`` float64, symmetric, on the backend's device;
            entry ``(a, b)`` in ``1 / (unit_a * unit_b)``.
        """
        if kwargs.get("convert_to_ra_dec"):
            raise NotImplementedError("the lookup information matrix takes ICRS params")
        xp = self.xp
        p = xp.ascontiguousarray(xp.atleast_2d(xp.asarray(params, dtype=float)))
        n, nparams = int(p.shape[0]), int(p.shape[1])
        inds = (list(range(nparams)) if inds is None
                else [int(i) for i in np.asarray(_host(inds)).ravel()])
        nd = len(inds)
        eps = self._info_matrix_steps(nparams, param_eps)
        if not np.all(np.isfinite(eps[inds]) & (eps[inds] != 0.0)):
            raise ValueError(f"information_matrix: non-finite or zero step in "
                             f"{eps[inds]} (inds {inds})")

        # invC rows: one per source, from the full active band of each walker's plane
        holder = self._as_wdm_holder(wdm_holder)
        rows_map = getattr(holder, "psd_row_index", None)
        if rows_map is None and getattr(holder, "band_slab_Nf", None) is not None:
            raise NotImplementedError(
                "information_matrix needs the full-active-band invC (the parent ACA, or a "
                "shared-psd mirror holder); per-slot narrow invC slabs are not supported.")
        Nt = self.Nt_active
        iC_flat = xp.asarray(holder.linear_psd_arr[0])
        plane = 9 * self.Nf_active * Nt
        if int(iC_flat.size) % plane:
            raise ValueError(f"linear_psd_arr[0] has {int(iC_flat.size)} entries, not a "
                             f"multiple of the (3, 3, {self.Nf_active}, {Nt}) invC plane")
        iC_all = iC_flat.reshape(-1, 3, 3, self.Nf_active, Nt)
        ni = (np.zeros(n, dtype=np.int64) if noise_index is None
              else np.asarray(_host(noise_index), dtype=np.int64).ravel())
        if ni.shape[0] != n:
            raise ValueError(f"noise_index has {ni.shape[0]} rows, params {n}")
        if rows_map is not None:
            ni = np.asarray(_host(rows_map), dtype=np.int64).ravel()[ni]
        if n and (ni.min() < 0 or ni.max() >= int(iC_all.shape[0])):
            raise IndexError(f"invC rows {ni.min()}..{ni.max()} outside the holder's "
                             f"{int(iC_all.shape[0])} rows")

        # per-source slab [lo, lo + W) around the carrier layer of f0
        L = self.num_m_layers
        W = min(2 * L + 4, self.Nf_active)
        f0 = np.asarray(_host(p[:, 1]))
        lo = np.clip(np.floor(f0 / self.layer_df).astype(np.int64) - L - 1,
                     self.ind_min_f, self.ind_max_f - W + 1).astype(np.int32)

        out = xp.zeros((n, nd, nd))
        # bytes per source: 2 nd stepped fills + nd derivatives (3 channels each) + the
        # gathered 3 x 3 invC rows, all (W, Nt) float64
        per_src = (3 * nd * 3 + 9) * W * Nt * 8
        chunk = max(1, int(max_bytes // per_src))
        lay = xp.arange(W)
        two_eps = xp.asarray(2.0 * eps[inds])[:, None, None, None, None]
        for s0 in range(0, n, chunk):
            k = min(chunk, n - s0)
            # 2 nd k template rows ordered [parameter a][sign +, -][source]
            P = xp.tile(p[s0:s0 + k], (2 * nd, 1)).reshape(nd, 2, k, nparams)
            for a, i in enumerate(inds):
                P[a, 0, :, i] += eps[i]
                P[a, 1, :, i] -= eps[i]
            nr = 2 * nd * k
            buf = xp.zeros(nr * 3 * W * Nt)
            self.fill_lookup(P.reshape(nr, nparams), buf,
                             data_index=xp.arange(nr, dtype=xp.int32), factors=xp.ones(nr),
                             band_slab_Nf=W, slab_min_f=np.tile(lo[s0:s0 + k], 2 * nd))
            F = buf.reshape(nd, 2, k, 3, W, Nt)
            dh = (F[:, 0] - F[:, 1]) / two_eps                      # (nd, k, 3, W, Nt)
            del buf, F
            # each source's invC rows at its slab layers: (k, W, 3, 3, Nt)
            ll = xp.asarray(lo[s0:s0 + k] - self.ind_min_f)
            iC = iC_all[xp.asarray(ni[s0:s0 + k])[:, None], :, :,
                        ll[:, None] + lay[None, :], :]
            G = xp.einsum("akcwt,kwcdt,bkdwt->kab", dh, iC, dh)
            out[s0:s0 + k] = 0.5 * (G + G.transpose(0, 2, 1))   # exact symmetry
        return out

    def __getattr__(self, name):
        # delegate the rest of the computation-object surface to the chunked comp;
        # never for dunders / before __init__ set ``chunked`` (deepcopy / pickle)
        if name.startswith("__") or "chunked" not in self.__dict__:
            raise AttributeError(name)
        return getattr(self.__dict__["chunked"], name)

    def get_ll_wdm(self, params, wdm_holder, data_index=None, noise_index=None,
                   convert_to_ra_dec=False, **kwargs):
        """``ll = -d_d / 2 + d_h - h_h / 2`` per row (kernel convention, plain sums);
        stores ``d_h_out`` / ``h_h_out`` / ``d_h_im_out``. ``wdm_holder`` exposes
        ``linear_data_arr[0]`` / ``linear_psd_arr[0]`` (XYZ 3x3 invC), with optional
        ``band_slab_Nf`` + ``slab_min_f`` (narrow per-slot slabs) and
        ``psd_row_index`` (shared-psd mirror)."""
        if convert_to_ra_dec:
            raise NotImplementedError("the lookup scorer takes ICRS params")
        xp = self.xp
        x = xp.ascontiguousarray(xp.atleast_2d(xp.asarray(params, dtype=float)))
        num_bin, nparams = x.shape
        nch = 3
        W = getattr(wdm_holder, "band_slab_Nf", None)
        slab_lo = getattr(wdm_holder, "slab_min_f", None)
        if W is None or slab_lo is None:
            W, slab_lo = self.Nf_active, None
        W = int(W)
        data = xp.asarray(wdm_holder.linear_data_arr[0])
        invC = xp.asarray(wdm_holder.linear_psd_arr[0])
        n_slots_d = int(data.size // (nch * W * self.Nt_active))
        rows = getattr(wdm_holder, "psd_row_index", None)
        if rows is not None:
            W_c = self.Nf_active
            invC_row = xp.ascontiguousarray(xp.asarray(rows, dtype=xp.int32))
        else:
            W_c = W
            invC_row = xp.zeros(0, dtype=xp.int32)
        n_slots_c = int(invC.size // (nch * nch * W_c * self.Nt_active))
        di = xp.zeros(num_bin, dtype=xp.int32) if data_index is None else \
            xp.ascontiguousarray(xp.asarray(data_index, dtype=xp.int32))
        ni = di if noise_index is None else \
            xp.ascontiguousarray(xp.asarray(noise_index, dtype=xp.int32))
        d_h = xp.zeros(num_bin)
        h_h = xp.zeros(num_bin)
        d_h_im = xp.zeros(num_bin)
        self.cpp.gb_lookup_get_ll(
            self.tdi_wrap, d_h, h_h, d_h_im, x.reshape(-1), di, ni,
            xp.ascontiguousarray(data.reshape(-1)), xp.ascontiguousarray(invC.reshape(-1)),
            (xp.zeros(0, dtype=xp.int32) if slab_lo is None
             else xp.ascontiguousarray(xp.asarray(slab_lo, dtype=xp.int32))),
            invC_row, W, W_c, n_slots_d, n_slots_c, num_bin, nparams, nch,
            self.n_nodes, self.t_node0, self.dt_node,
            self.t0, self.layer_dt, self.layer_df,
            self.ind_min_t, self.Nt_active, self.ind_min_f, self.ind_max_f,
            self.num_m_layers, int(self.k1), int(self.k_coarse), *self._tab)
        self.d_h_out, self.h_h_out, self.d_h_im_out = d_h, h_h, d_h_im
        self.last_d_h, self.last_h_h, self.last_d_h_im = d_h, h_h, d_h_im
        return -0.5 * self.d_d + d_h - 0.5 * h_h


def _gbwdm_base():
    from .gbcomps import GBWDMComputations

    return GBWDMComputations


class GBLookupWDMComputations(_gbwdm_base()):
    """A chunked ``GBWDMComputations`` whose ``get_ll_wdm`` runs the fused lookup scorer.

    The integration form for the global fit: it IS the chunked computation for every
    other purpose (fills, swaps, F-stat, attributes, sig-het's ``for_band_engine``
    delegate), so nothing that type-dispatches on or writes attributes to the GB comp
    changes; only the per-row likelihood switches to the lookup.

    Args: those of ``GBWDMComputations`` plus ``lookup_table`` (a ``WDMLookupTable`` or
    a path, built at the grid's layer duration), ``lookup_n_nodes`` /
    ``lookup_num_m_layers`` / ``lookup_k_coarse`` (see :class:`GBLookupComputations`) and
    ``lookup_fill`` (default off: fills stay chunked-het).
    """

    def __init__(self, *args, lookup_table, lookup_n_nodes=-1, lookup_num_m_layers=2,
                 lookup_k_coarse=True, lookup_fill=False, **kwargs):
        super().__init__(*args, **kwargs)
        self._lookup = GBLookupComputations(self, lookup_table, n_nodes=lookup_n_nodes,
                                            num_m_layers=lookup_num_m_layers,
                                            k_coarse=lookup_k_coarse)
        self.lookup_fill = bool(lookup_fill)

    def fill_global_wdm(self, params, templates, *args, **kwargs):
        """The chunked fill, or (``lookup_fill``) the lookup template -- then every GB
        template the fit writes is the same model its likelihood scores."""
        if not self.lookup_fill:
            return super().fill_global_wdm(params, templates, *args, **kwargs)
        if kwargs.get("convert_to_ra_dec") is None:
            kwargs["convert_to_ra_dec"] = bool(getattr(self, "convert_to_ra_dec", False))
        if kwargs.get("convert_to_ra_dec"):
            from lisatools.response.directresponse import ecliptic_to_icrs

            x = self.xp.asarray(self.xp.atleast_2d(params), dtype=float).copy()
            lam, beta = ecliptic_to_icrs(x[:, -2].copy(), x[:, -1].copy())
            x[:, -2], x[:, -1] = lam, beta
            params = x
        kwargs.pop("convert_to_ra_dec", None)
        return self._lookup.fill_lookup(params, templates, *args, **kwargs)

    def get_ll_wdm(self, params, wdm_holder, data_index=None, noise_index=None,
                   convert_to_ra_dec=None, **kwargs):
        """The chunked signature; ``grid_dim`` / layer-group / ``m_band_half_width`` knobs
        of the chunked kernel do not apply (the lookup band is ``lookup_num_m_layers``)."""
        if convert_to_ra_dec is None:
            convert_to_ra_dec = bool(getattr(self, "convert_to_ra_dec", False))
        x = self.xp.asarray(self.xp.atleast_2d(params), dtype=float).copy()
        if convert_to_ra_dec:
            from lisatools.response.directresponse import ecliptic_to_icrs

            lam, beta = ecliptic_to_icrs(x[:, -2].copy(), x[:, -1].copy())
            x[:, -2], x[:, -1] = lam, beta
        wdm_holder = self._as_wdm_holder(wdm_holder)
        self._lookup.d_d = float(getattr(self, "d_d", 0.0))
        ll = self._lookup.get_ll_wdm(x, wdm_holder, data_index=data_index,
                                     noise_index=noise_index)
        self.d_h_out = self._lookup.d_h_out
        self.h_h_out = self._lookup.h_h_out
        self.d_h_im_out = self._lookup.d_h_im_out
        return ll
