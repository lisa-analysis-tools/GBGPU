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
"""

from __future__ import annotations

import numpy as np

from lisatools.response.tdionfly import GBTDIonTheFly

from .parallelbase import GBGPUParallelModule


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
            import os

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
        self.Tobs = Tobs
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

    def information_matrix(self, params, wdm_holder, inds=None, param_eps=None,
                           noise_index=None, max_bytes=256 * 2**20, **kwargs):
        """Fisher matrix ``<dh_a | dh_b>`` from lookup templates -- the chunked
        delegate's definition (``chunked_het.information_matrix``), built from
        2 * len(inds) lookup FILLS per source (central differences) and ONE
        contraction against the invC rows ``noise_index`` of ``wdm_holder`` (a
        full-band ACA), instead of 4 swap launches per parameter pair. Same
        normalization as ``get_ll_wdm`` (plain sums). ICRS params."""
        xp = self.xp
        p = xp.ascontiguousarray(xp.atleast_2d(xp.asarray(params, dtype=float)))
        n, nparams = p.shape
        inds = list(range(nparams)) if inds is None else [int(i) for i in np.asarray(
            inds.get() if hasattr(inds, "get") else inds).ravel()]
        nd = len(inds)
        eps = np.asarray(self.chunked._info_matrix_param_eps(nparams, param_eps), dtype=float)
        if param_eps is None and nparams > 2:
            # f0 / fdot steps scaled to the observation: the chunked defaults (2e-14 Hz,
            # 1e-21 Hz/s) move the phase by ~1e-7 / 1e-8 rad, where the lookup's
            # interpolation noise dominates the difference (fdot-fdot came out 6x the
            # exact-template Gram at 30 d). 1e-4 / Tobs^k is flat in k over 1e-2..1e-5.
            eps[1] = 1e-4 / self.Tobs
            eps[2] = 1e-4 / self.Tobs ** 2
        Nt = self.Nt_active
        W = 2 * self.num_m_layers + 4        # carrier layer(s) +- L, plus Doppler drift
        f0 = np.asarray(p[:, 1].get() if hasattr(p, "get") else p[:, 1])
        lo = np.clip(np.floor(f0 / self.layer_df).astype(int) - self.num_m_layers - 1,
                     self.ind_min_f, self.ind_max_f - W + 1).astype(np.int32)
        ni = np.zeros(n, dtype=np.int64) if noise_index is None else np.asarray(
            noise_index.get() if hasattr(noise_index, "get") else noise_index).ravel()
        iC_all = xp.asarray(wdm_holder.linear_psd_arr[0]).reshape(
            -1, 3, 3, self.Nf_active, Nt)
        out = xp.zeros((n, nd, nd))
        per_src = 2 * nd * 3 * W * Nt * 8 + 9 * W * Nt * 8
        chunk = max(1, int(max_bytes // per_src))
        lay = np.arange(W)
        for s0 in range(0, n, chunk):
            k = min(chunk, n - s0)
            ps = p[s0:s0 + k]
            rows = []
            for a, i in enumerate(inds):
                for sgn in (1.0, -1.0):
                    q = ps.copy()
                    q[:, i] += sgn * eps[i]
                    rows.append(q)
            P = xp.concatenate(rows)                     # (2 nd k, 9): [a][sign][src]
            nr = int(P.shape[0])
            buf = xp.zeros(nr * 3 * W * Nt)
            lo_r = np.tile(lo[s0:s0 + k], 2 * nd)
            self.fill_lookup(P, buf, data_index=xp.arange(nr, dtype=xp.int32),
                             factors=xp.ones(nr), band_slab_Nf=W, slab_min_f=lo_r)
            F = buf.reshape(nd, 2, k, 3, W, Nt)
            dh = (F[:, 0] - F[:, 1]) / xp.asarray(2.0 * eps[inds])[:, None, None, None, None]
            ll = xp.asarray(lo[s0:s0 + k] - self.ind_min_f)
            iC = iC_all[xp.asarray(ni[s0:s0 + k])[:, None],
                        :, :, (ll[:, None] + xp.asarray(lay)[None, :]), :]   # (k, W, 3, 3, Nt)
            out[s0:s0 + k] = xp.einsum("akcwt,kwcdt,bkdwt->kab", dh, iC, dh)
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
