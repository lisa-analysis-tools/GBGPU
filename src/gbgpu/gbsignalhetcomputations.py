"""GBSignalHetComputations -- installed GB signal-heterodyne WDM likelihood.

The GB signal-het frontend, living in GBGPU (the GB-physics owner) alongside the
chunked-het ``GBWDMComputations``. Exposes the GBGPU backend
(``FastLISAResponseParallelModule`` -> ``self.backend`` /
``self.backend.GBComputationGroupWrap()``) and sets up ALL the reference
coefficients in ``__init__``, taken from installed infrastructure only:

* WDM data + inverse-sensitivity: ``lisatools`` (``TDSignal.transform`` +
  ``XYZ2SensitivityMatrix``).
* WDM analysis window: ``lisatools.domains.WDMSettings.window`` (NO re-implemented
  ``phitilde``).
* sparse-time grid + bin-fold coefficients: ``lisatools.signal_het``.
* reference carrier ``c0`` (sparse + dense complex WDM): the GBGPU backend producer
  ``gb_signal_het_make_reference`` (the chunked-het FD-gen + polyphase on the
  reference params) -- NOT a Python polyphase.

``get_ll(params, data_index)`` calls the backend ``gb_signal_het_get_ll_in_kernel``.
CPU-only for now (the sig-het CUDA kernels are a TODO -- construct with
``force_backend="cpu"``).
"""
from __future__ import annotations

import math
from copy import deepcopy

import numpy as np
from scipy.signal.windows import tukey as _tukey

from lisatools.domains import TDSettings, TDSignal, WDMSettings
from lisatools.analysiscontainer import AnalysisContainer
from lisatools.sensitivity import XYZ2SensitivityMatrix
from lisatools.response.tdiconfig import TDIConfig
from lisatools.response.tdionfly import GBTDIonTheFly
from lisatools.signal_het import sparse_time_grid, bin_fold_real

from .parallelbase import GBGPUParallelModule as FastLISAResponseParallelModule


def _recommended_edge_cut(Nt, tukey_alpha, margin=8):
    """Edge-cut (WDM time-layers) matching the Tukey taper: ``max(20, taper+margin)``,
    where the taper is ``ceil(0.5*alpha*Nt)`` layers. Auto-decided from the window."""
    taper = int(math.ceil(0.5 * float(tukey_alpha) * int(Nt)))
    return max(20, taper + int(margin))


class GBSignalHetComputations(FastLISAResponseParallelModule):
    """Signal-het GB likelihood. ``__init__`` builds the heterodyne reference ``c0``
    (from the backend) and the bin-fold coefficients (from lisatools); ``get_ll``
    evaluates logL for candidate params.

    Args mirror the chunked-het / dev sig-het frontends; ``ref_params`` is the
    length-9 heterodyne reference ``(amp, f0, fdot, fddot, phi0, inc, psi, lam, beta)``.
    """

    @classmethod
    def supported_backends(cls):
        return ["cpu"]           # sig-het CUDA kernels are a TODO

    def __init__(self, data_td, ref_params, *, Nf, Nt, dt, t0, t_ref,
                 orbits, tdi_config, min_freq, max_freq, sens_model="scirdv1",
                 edge_cut=None, nt_layer=64, n_sparse_fd=1024, m_active_half_width=2,
                 max_r=5.0, tukey_alpha=0.05, force_backend="cpu"):
        super().__init__(force_backend=force_backend)
        self.cpp = self.backend.GBComputationGroupWrap()

        self.taper_layers = int(math.ceil(0.5 * float(tukey_alpha) * int(Nt)))
        self.edge_cut = int(_recommended_edge_cut(Nt, tukey_alpha)
                            if edge_cut is None else edge_cut)
        self.m_half = int(m_active_half_width)

        Nobs = Nf * Nt
        Tobs = Nt * Nf * dt
        t_arr = np.arange(Nobs) * dt + t0
        # plain backend flavor for the lisatools frontends (TDIConfig / TDSettings /
        # WDMSettings / GBTDIonTheFly). self.backend is the GBGPU COMPOSITE backend
        # (name "lisatools_gbgpu_cpu") whose GBComputationGroupWrap self.cpp uses; the
        # lisatools objects want the plain flavor ("cpu").
        bname = force_backend if isinstance(force_backend, str) else "cpu"
        if isinstance(tdi_config, str):
            tdi_config = TDIConfig(tdi_config, force_backend=bname)
        td_set = TDSettings(Nobs, dt, t0=t0, force_backend=bname)
        window = (_tukey(Nobs, alpha=tukey_alpha).astype(float) if tukey_alpha > 0
                  else np.ones(Nobs))

        wdm_kw = dict(t0=t0, min_freq=min_freq, max_freq=max_freq,
                      min_time=self.edge_cut * Nf * dt,
                      max_time=(Nt - self.edge_cut) * Nf * dt, force_backend=bname)
        wdm_set_real = WDMSettings(Nf, Nt, dt, is_complex=False, **wdm_kw)
        wdm_set_complex = WDMSettings(Nf, Nt, dt, is_complex=True, **wdm_kw)

        # GB TD-on-the-fly generator: its wave_gen drives the backend FD-gen used by
        # both the reference producer and the per-call kernel.
        t_tdi = np.linspace(t_arr[0], t_arr[-1], 16384)
        gb_gen = GBTDIonTheFly(t_tdi, Tobs, t_ref, 1.0 / dt, 1,
                               tdi_config=tdi_config, orbits=orbits, tdi_chan="XYZ",
                               force_backend=bname)
        self.tdi_wrap = gb_gen.wave_gen

        # --- data on the WDM band (real + complex) + d_d (all lisatools) ---------
        data_real = TDSignal(data_td, settings=td_set).transform(wdm_set_real, window=window)
        data_complex = np.asarray(
            TDSignal(data_td, settings=td_set).transform(wdm_set_complex, window=window).arr)
        sens_real = XYZ2SensitivityMatrix(wdm_set_real, model=sens_model)
        self.analysis = AnalysisContainer(data_real, sens_real)
        self.d_d = float(np.real(self.analysis.inner_product()))
        invC_complex = np.asarray(XYZ2SensitivityMatrix(wdm_set_complex, model=sens_model).invC)

        # --- grid + window: taken DIRECTLY from lisatools -----------------------
        ind_min_t = int(wdm_set_real.ind_min_t)
        ind_min_f = int(wdm_set_real.ind_min_f)
        Nt_active = int(wdm_set_real.Nt_active)
        Nf_active = int(wdm_set_real.ind_max_f - wdm_set_real.ind_min_f + 1)
        stride, N_sparse_t, n_sparse_local = sparse_time_grid(Nt, Nt_active, nt_layer)
        window_full = np.asarray(wdm_set_real.window, dtype=np.float64)   # lisatools window

        # --- reference c0 from the GBGPU BACKEND producer (no Python polyphase) --
        ref_params = np.asarray(ref_params, dtype=float).reshape(9)
        layer_df = float(wdm_set_real.layer_df)
        c0_sparse = np.zeros((1, 3, Nf_active, N_sparse_t), dtype=np.complex128)
        c0_dense = np.zeros((1, 3, Nf_active, Nt_active), dtype=np.complex128)
        self.cpp.gb_signal_het_make_reference(
            self.tdi_wrap, c0_sparse, c0_dense, window_full, n_sparse_local,
            np.ascontiguousarray(ref_params[None]), 1, 9, 1, 2,
            Nf, Nt, Nf_active, Nt_active, nt_layer, N_sparse_t, stride,
            ind_min_t, ind_min_f, layer_df, dt, Tobs, t0,
            3, n_sparse_fd, tukey_alpha)

        # --- bin-fold coefficients from lisatools (dense c0 x data x invC) -------
        A0, A1, B0, B1, B0nc, B1nc = bin_fold_real(
            data_complex, c0_dense[0], invC_complex, n_sparse_local, stride,
            Nt_active, tdi_type="XYZ")

        self.c0_sparse_all = np.ascontiguousarray(c0_sparse)   # (1, 3, Nf_active, N_sparse_t)
        self.A0_all = np.ascontiguousarray(A0[None]); self.A1_all = np.ascontiguousarray(A1[None])
        self.B0_all = np.ascontiguousarray(B0[None]); self.B1_all = np.ascontiguousarray(B1[None])
        self.B0nc_all = np.ascontiguousarray(B0nc[None]); self.B1nc_all = np.ascontiguousarray(B1nc[None])
        self.window_full = window_full
        self.n_sparse_local = n_sparse_local
        self.params_ref_all = np.ascontiguousarray(ref_params.reshape(1, 9))

        # keep-alives: the kernel reads C++ pointers held inside self.tdi_wrap;
        # keep the Python objects that own that state alive for the object lifetime.
        self._keep_alive = dict(gb_gen=gb_gen, orbits=orbits, tdi_config=tdi_config,
                                td_set=td_set, wdm_set_real=wdm_set_real,
                                wdm_set_complex=wdm_set_complex, window=window)
        self._g = dict(Nf=Nf, Nt=Nt, Nf_active=Nf_active, Nt_active=Nt_active,
                       nt_layer=nt_layer, N_sparse_t=N_sparse_t, stride=stride,
                       ind_min_t=ind_min_t, ind_min_f=ind_min_f, layer_df=layer_df,
                       dt=dt, Tobs=Tobs, t0=t0, n_sparse_fd=n_sparse_fd,
                       tukey_alpha=tukey_alpha, max_r=max_r, m_half=self.m_half)

    @property
    def xp(self):
        return self.backend.xp

    # ------------------------------------------------------------------
    # Band-engine mode: sig-het scoring INSIDE the GB special move.
    #
    # Constructed via :meth:`for_band_engine` around a chunked-het
    # GBWDMComputations delegate. Everything the WDM band engine touches
    # (fill_global_wdm / get_ll_wdm / get_swap_ll_wdm / grads) routes to
    # the chunked delegate, EXCEPT get_ll_wdm while an in-model reference
    # is active: setup_in_model() -- called by the move once per picked
    # source batch, at the friends/info-matrix stage, AFTER the sources
    # are removed from their cell residuals -- builds the heterodyne
    # reference (backend c0 producer + lisatools bin-fold) against the
    # source-free residual slabs, and every repeat proposal then scores
    # through the fast sig-het kernel against that CONSTANT reference.
    # clear_in_model() (after the repeat block) reverts get_ll_wdm to the
    # chunked delegate. Dispatch is purely by comp type: settings hand
    # the move THIS class instead of GBWDMComputations, nothing else
    # changes (chunked comps inherit no-op hooks from
    # WDMComputationsBase).
    # ------------------------------------------------------------------

    @classmethod
    def for_band_engine(cls, chunked_comp, *, nt_layer=64, n_sparse_fd=1024,
                        m_active_half_width=2, max_r=5.0):
        """Build a data-less engine-mode instance around ``chunked_comp``.

        Grid, orbits, TDI configuration, phase reference time, window and
        taper all come from the chunked delegate's ``wdm_settings`` (the
        run's active-band WDM grid), so the two likelihoods are defined on
        identical grids. CPU-only (the sig-het kernels are CPU for now),
        like the plain constructor.
        """
        wdm = chunked_comp.wdm_settings
        self = cls.__new__(cls)
        FastLISAResponseParallelModule.__init__(self, force_backend="cpu")
        self.cpp = self.backend.GBComputationGroupWrap()
        self.chunked = chunked_comp
        self.m_half = int(m_active_half_width)

        Nf, Nt = int(wdm.Nf), int(wdm.Nt)
        dt = float(wdm.data_dt)
        t0 = float(wdm.t0)
        Tobs = float(wdm.Tobs)
        Nobs = Nf * Nt

        ind_min_t = int(wdm.ind_min_t)
        ind_min_f = int(wdm.ind_min_f)
        Nt_active = int(wdm.Nt_active)
        Nf_active = int(wdm.ind_max_f - wdm.ind_min_f + 1)
        stride, N_sparse_t, n_sparse_local = sparse_time_grid(
            Nt, Nt_active, nt_layer)
        self.window_full = np.asarray(wdm.window, dtype=np.float64)
        self.n_sparse_local = n_sparse_local

        # The chunked comp stores the raw ctor value (possibly the -1 AUTO
        # sentinel); the kernels consume the RESOLVED alpha.
        tukey_alpha = float(getattr(
            chunked_comp, "resolved_tukey_alpha",
            getattr(chunked_comp, "tukey_alpha", 0.0)))
        if tukey_alpha < 0.0:
            tukey_alpha = 0.0

        tdi_config = chunked_comp.tdi_config
        orbits = chunked_comp.orbits
        t_tdi = np.linspace(t0, t0 + (Nobs - 1) * dt, 16384)
        gb_gen = GBTDIonTheFly(t_tdi, Tobs, float(chunked_comp.t_ref),
                               1.0 / dt, 1, tdi_config=tdi_config,
                               orbits=orbits, tdi_chan="XYZ",
                               force_backend="cpu")
        self.tdi_wrap = gb_gen.wave_gen
        self._keep_alive = dict(gb_gen=gb_gen, orbits=orbits,
                                tdi_config=tdi_config)

        self._g = dict(Nf=Nf, Nt=Nt, Nf_active=Nf_active, Nt_active=Nt_active,
                       nt_layer=int(nt_layer), N_sparse_t=N_sparse_t,
                       stride=stride, ind_min_t=ind_min_t,
                       ind_min_f=ind_min_f, layer_df=float(wdm.layer_df),
                       dt=dt, Tobs=Tobs, t0=t0,
                       n_sparse_fd=int(n_sparse_fd),
                       tukey_alpha=tukey_alpha, max_r=float(max_r),
                       m_half=self.m_half)
        # Deltas are what the in-model repeats consume; keep the chunked
        # delegate's d_d convention so absolute ll values line up too.
        self.d_d = float(getattr(chunked_comp, "d_d", 0.0))
        self._in_model = None
        self._slot_to_ref = None
        return self

    def setup_in_model(self, buffer_aca, params_ref_phys, data_index,
                       N_vals=None) -> bool:
        """Build (or patch) the per-source heterodyne references.

        ``params_ref_phys`` (n, 9): the picked sources' CURRENT physical
        parameters (the heterodyne expansion points). ``data_index`` (n,):
        their buffer slots; the slot's residual slab -- which at this point
        CONTAINS the source's signal (the move removed the template from
        the model just before calling this) -- plus its inverse-PSD slab
        feed the bin-fold. Only the REAL part of the WDM data enters the
        real-projection bin-fold, so the buffer's real residual is the
        exact data-side input.

        INCREMENTAL semantics: with no reference active (block start,
        after clear_in_model) this builds fresh for all given slots. While
        a reference IS active, the given slots must be a subset of the
        existing ones and only THOSE slots' coefficient blocks are
        rebuilt in place -- the move's mid-block drift refresh uses this
        to re-anchor only the sources that walked too far from their
        expansion point. Returns True so callers can tell an active
        sig-het setup from the no-op hooks (which return None)."""
        g = self._g
        nch = 3
        slots = np.asarray(
            data_index.get() if hasattr(data_index, "get") else data_index,
            dtype=int)
        # FORCED copy: ascontiguousarray(asarray(...)) returns a VIEW for
        # float64-contiguous input, and params_ref_all must never alias
        # the caller's array (the mid-block patch writes into it).
        refs = np.array(
            params_ref_phys.get() if hasattr(params_ref_phys, "get")
            else params_ref_phys, dtype=float, copy=True
        ).reshape(-1, 9)
        n = refs.shape[0]

        slab_shape = (-1, nch, g["Nf_active"], g["Nt_active"])
        res = np.asarray(buffer_aca.linear_data_arr[0]).reshape(slab_shape)[slots]
        psd_flat = np.asarray(buffer_aca.linear_psd_arr[0])
        expected_xyz = psd_flat.size // (nch * nch * g["Nf_active"] * g["Nt_active"])
        if expected_xyz * nch * nch * g["Nf_active"] * g["Nt_active"] != psd_flat.size:
            raise NotImplementedError(
                "sig-het in-model setup currently supports the XYZ "
                "cross-channel inverse covariance layout only.")
        invC = psd_flat.reshape(-1, nch, nch, g["Nf_active"], g["Nt_active"])[slots]

        c0_sparse = np.zeros((n, nch, g["Nf_active"], g["N_sparse_t"]),
                             dtype=np.complex128)
        c0_dense = np.zeros((n, nch, g["Nf_active"], g["Nt_active"]),
                            dtype=np.complex128)
        self.cpp.gb_signal_het_make_reference(
            self.tdi_wrap, c0_sparse, c0_dense, self.window_full,
            self.n_sparse_local, refs, n, 9, 1, 2,
            g["Nf"], g["Nt"], g["Nf_active"], g["Nt_active"],
            g["nt_layer"], g["N_sparse_t"], g["stride"],
            g["ind_min_t"], g["ind_min_f"], g["layer_df"],
            g["dt"], g["Tobs"], g["t0"],
            3, g["n_sparse_fd"], g["tukey_alpha"])

        A0l, A1l, B0l, B1l, B0ncl, B1ncl = [], [], [], [], [], []
        for i in range(n):
            A0, A1, B0, B1, B0nc, B1nc = bin_fold_real(
                res[i].astype(np.complex128), c0_dense[i], invC[i],
                self.n_sparse_local, g["stride"], g["Nt_active"],
                tdi_type="XYZ")
            A0l.append(A0); A1l.append(A1); B0l.append(B0)
            B1l.append(B1); B0ncl.append(B0nc); B1ncl.append(B1nc)

        if self._in_model is not None:
            # Mid-block PATCH: re-anchor only the given slots (the move's
            # drift refresh). They must already carry a reference.
            if int(slots.max()) >= len(self._slot_to_ref):
                raise RuntimeError(
                    "sig-het in-model patch hit a slot outside the "
                    "block's reference set.")
            ref_idx = self._slot_to_ref[slots]
            if np.any(ref_idx < 0):
                raise RuntimeError(
                    "sig-het in-model patch hit a slot with no reference; "
                    "mid-block refreshes must target the block's slots.")
            self.c0_sparse_all[ref_idx] = c0_sparse
            self.A0_all[ref_idx] = np.stack(A0l)
            self.A1_all[ref_idx] = np.stack(A1l)
            self.B0_all[ref_idx] = np.stack(B0l)
            self.B1_all[ref_idx] = np.stack(B1l)
            self.B0nc_all[ref_idx] = np.stack(B0ncl)
            self.B1nc_all[ref_idx] = np.stack(B1ncl)
            self.params_ref_all[ref_idx] = refs
            return True

        self.c0_sparse_all = np.ascontiguousarray(c0_sparse)
        self.A0_all = np.ascontiguousarray(np.stack(A0l))
        self.A1_all = np.ascontiguousarray(np.stack(A1l))
        self.B0_all = np.ascontiguousarray(np.stack(B0l))
        self.B1_all = np.ascontiguousarray(np.stack(B1l))
        self.B0nc_all = np.ascontiguousarray(np.stack(B0ncl))
        self.B1nc_all = np.ascontiguousarray(np.stack(B1ncl))
        self.params_ref_all = refs

        slot_map = np.full(int(slots.max()) + 1, -1, dtype=int)
        slot_map[slots] = np.arange(n)
        self._slot_to_ref = slot_map
        self._in_model = True
        return True

    def clear_in_model(self) -> None:
        """Deactivate the in-model reference: get_ll_wdm routes back to the
        chunked delegate (RJ / removal / any out-of-block scoring)."""
        self._in_model = None
        self._slot_to_ref = None

    # ---- band-engine surface (chunked delegate + in-model routing) -------

    def fill_global_wdm(self, *args, **kwargs):
        return self.chunked.fill_global_wdm(*args, **kwargs)

    def get_swap_ll_wdm(self, *args, **kwargs):
        return self.chunked.get_swap_ll_wdm(*args, **kwargs)

    def get_ll_grad_wdm(self, *args, **kwargs):
        return self.chunked.get_ll_grad_wdm(*args, **kwargs)

    def hessian_wdm(self, *args, **kwargs):
        return self.chunked.hessian_wdm(*args, **kwargs)

    def information_matrix(self, *args, **kwargs):
        return self.chunked.information_matrix(*args, **kwargs)

    def get_ll_wdm(self, params, wdm_holder, data_index=None,
                   noise_index=None, **kwargs):
        """Chunked-het scoring, EXCEPT while an in-model reference is active:
        then candidates score through the sig-het kernel against the fixed
        reference of their buffer slot (``data_index`` maps slot -> ref)."""
        if self._in_model is None:
            ll = self.chunked.get_ll_wdm(
                params, wdm_holder, data_index=data_index,
                noise_index=noise_index, **kwargs)
            self.d_h_out = self.chunked.d_h_out
            self.h_h_out = self.chunked.h_h_out
            return ll
        slots = np.asarray(
            data_index.get() if hasattr(data_index, "get") else data_index,
            dtype=int)
        ref_idx = self._slot_to_ref[slots]
        if np.any(ref_idx < 0):
            raise RuntimeError(
                "sig-het in-model scoring hit a buffer slot with no "
                "reference; setup_in_model was not run for it.")
        p = params.get() if hasattr(params, "get") else params
        ll = self.get_ll(np.asarray(p, dtype=float), data_index=ref_idx)
        self.d_h_out = self.last_d_h
        self.h_h_out = self.last_h_h
        return ll

    def get_ll(self, params, data_index=None, phase_maximize=False):
        """logL for candidate ``params`` (length-9 or ``(N,9)``); ``data_index``
        ``(N,)`` selects the reference (default all-zero -> single reference).

        ``phase_maximize=True`` analytically maximises over the initial
        phase via the two-quadrature trick (second kernel call at
        ``phi0 + pi/2``; physical column 4): ``d_h -> |d_h_0 + i d_h_90|``,
        with the maximising PHYSICAL phi0 shift stashed on
        ``self.phase_angle`` (``None`` otherwise) and the un-maximised d_h on
        ``self.non_marg_d_h``. Same convention as
        ``gb_likelihood.TwoQuadraturePhaseMaxMixin``.
        """
        if phase_maximize:
            ll_0 = self.get_ll(params, data_index=data_index)
            d_h_0, h_h = self.last_d_h.copy(), self.last_h_h.copy()
            # FORCED copy: the quadrature shift must not mutate the
            # caller's params through an ascontiguousarray view.
            x_q = np.atleast_2d(np.array(params, dtype=float, copy=True))
            x_q[:, 4] = x_q[:, 4] + np.pi / 2
            self.get_ll(x_q, data_index=data_index)
            d_h_90 = self.last_d_h.copy()
            D = d_h_0 + 1j * d_h_90
            d_h_max = np.abs(D)
            self.non_marg_d_h = d_h_0
            self.phase_angle = np.arctan2(D.imag, D.real)
            self.last_d_h = d_h_max
            self.last_h_h = h_h
            return ll_0 + (d_h_max - d_h_0)
        self.phase_angle = None
        x = np.ascontiguousarray(np.atleast_2d(np.asarray(params, dtype=float)))
        N = x.shape[0]
        di = (np.zeros(N, dtype=np.int32) if data_index is None
              else np.asarray(data_index, dtype=np.int32))
        d_h = np.zeros(N, dtype=np.float64)
        h_h = np.zeros(N, dtype=np.float64)
        num_data = int(self.params_ref_all.shape[0])
        g = self._g
        self.cpp.gb_signal_het_get_ll_in_kernel(
            self.tdi_wrap, d_h, h_h, self.c0_sparse_all,
            self.A0_all, self.A1_all, self.B0_all, self.B1_all,
            self.B0nc_all, self.B1nc_all,
            self.window_full, self.n_sparse_local,
            x, self.params_ref_all, di,
            N, num_data, 9, 1, 2,
            g["Nf"], g["Nt"], g["Nf_active"], g["Nt_active"],
            g["nt_layer"], g["N_sparse_t"], g["stride"],
            g["ind_min_t"], g["ind_min_f"], g["m_half"],
            g["layer_df"], g["dt"], g["Tobs"], g["t0"],
            3, 0, g["n_sparse_fd"],
            g["tukey_alpha"], g["max_r"], 1)     # project_real=1
        self.last_d_h = np.asarray(d_h).copy()
        self.last_h_h = np.asarray(h_h).copy()
        return -0.5 * self.d_d + np.asarray(d_h) - 0.5 * np.asarray(h_h)
