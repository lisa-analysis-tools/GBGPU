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

Backends: the in-model band-engine path (``for_band_engine`` ->
``setup_in_model`` / ``get_ll``) runs on CPU and CUDA -- the fused
``gb_signal_het_get_ll_in_kernel`` kernel evaluates each candidate entirely in
per-block shared memory, and ``gb_signal_het_make_reference`` builds the
per-source reference coefficients on-device. The full data-loading constructor
below stays CPU-only (offline/validation use).
"""
from __future__ import annotations

import math
import os
from copy import deepcopy

import logging
import numpy as np

logger = logging.getLogger(__name__)
from scipy.signal.windows import tukey as _tukey

from lisatools.domains import TDSettings, TDSignal, WDMSettings
from lisatools.analysiscontainer import AnalysisContainer
from lisatools.sensitivity import XYZ2SensitivityMatrix
from lisatools.response.tdiconfig import TDIConfig
from lisatools.response.tdionfly import GBTDIonTheFly
from lisatools.signal_het import sparse_time_grid, bin_fold_real

from .parallelbase import GBGPUParallelModule as FastLISAResponseParallelModule


# Peak-memory cap (bytes) for the batched bin-fold in ``setup_in_model``. The
# <h|h> intermediates (Ec + En) dominate the setup's high-water mark, so the
# source axis is chunked to stay under this. 1 GiB leaves room for the
# residual/invC slabs and the reference stash on the same device while still
# batching every realistic block (a mojito-scale source costs ~13 MB here, so
# ~75 sources per chunk).
#
# GB_SIGHET_FOLD_MAX_BYTES overrides it. Lowering it is how the parity gates
# force the multi-chunk branch with only a handful of sources -- otherwise
# concatenation across chunks is never exercised at realistic block sizes.
_SIGHET_FOLD_MAX_BYTES = int(os.environ.get("GB_SIGHET_FOLD_MAX_BYTES", 1 << 30))

# Extra WDM layers each side of the reference carrier included in the WINDOWED
# reference build, beyond the make_reference kernel's own write extent of
# ceil(n_sparse_fd/Nt)+1 layers. The windowed build is EXACT for any margin
# >= 0 (outside the kernel's write window c0 is identically zero, so every
# fold coefficient is too); the default 1 only absorbs the floor-vs-C-integer
# carrier-layer rounding difference between Python and the kernel.
_SIGHET_WINDOW_MARGIN = int(os.environ.get("GB_SIGHET_WINDOW_MARGIN", 1))


def _resolve_n_cp(n_cp_build, Tobs):
    """Control-point count for the spline waveform build.

    ``n_cp_build`` semantics: 0 -> direct per-point evaluation (legacy
    path); > 1 -> explicit count; < 0 -> AUTO: one node per
    ``SIGHET_N_CP_SPACING`` seconds of Tobs (default 4 days -- keeps the
    cubic-spline phase error of the annual Doppler/response modulation
    below ~1e-4 rad across the GB band), clamped to [32, 256]. The upper
    clamp bounds the shared-memory arena (~253 B/node); the lower keeps
    the fit far inside the chunked engine's validated density.
    """
    if n_cp_build == 0:
        return 0
    if n_cp_build > 1:
        return int(n_cp_build)
    spacing = float(os.environ.get("SIGHET_N_CP_SPACING", 4.0 * 86400.0))
    return int(np.clip(int(math.ceil(float(Tobs) / spacing)) + 1, 32, 256))


def _c0_row_mask_bits(c0, xp):
    """Bit-packed |c0| row-floor mask -- the v5 scorer's only view of c0.

    ``c0`` is the expanded reference stash, ``(n, nch, Nf_active,
    N_sparse_t)``. Returns ``uint64`` of shape ``(n, nch, Nf_active,
    ceil(N_sparse_t / 64))`` with bit ``b % 64`` of word ``b // 64`` set
    where pixel ``b`` survives its row's floor.

    This is EXACTLY the test the v4 kernel runs -- ``|c0| > max(1e-12 *
    max_b |c0|, 1e-300)`` along each ``(c, m_local)`` row -- hoisted out of
    the scorer. It depends only on the reference, never on the candidate
    being scored, yet v4 recomputes it (and re-reads all of c0 twice, on
    nch*M of 128 threads) inside every candidate's block. Doing it once here
    is what lets the v5 fold rebuild ``r``/``dr`` from ``r_pix`` and one bit
    instead of from a materialised 480 B/pixel shared scratch. See the V5
    section of ``cutils/gb_tdi_on_the_fly.cu``.

    ``max`` is exact and order-independent, and the comparison is the same
    comparison, so the mask is bit-for-bit v4's -- which is what makes the
    v5 fold bit-identical rather than merely equal to round-off.
    """
    mag = xp.abs(c0)
    floor_a = 1e-12 * mag.max(axis=-1)
    floor_th = xp.where(floor_a > 1e-300, floor_a, 1e-300)
    keep = mag > floor_th[..., None]
    N = keep.shape[-1]
    nwords = (N + 63) // 64
    pad = nwords * 64 - N
    if pad:
        keep = xp.concatenate(
            [keep, xp.zeros(keep.shape[:-1] + (pad,), dtype=bool)], axis=-1)
    k = keep.reshape(keep.shape[:-1] + (nwords, 64)).astype(xp.uint64)
    # Disjoint bits, so the sum IS the bitwise or -- and it vectorises.
    bits = xp.arange(64, dtype=xp.uint64)
    return xp.ascontiguousarray((k << bits).sum(axis=-1, dtype=xp.uint64))


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
        return ["cpu", "cuda11x", "cuda12x", "cuda13x"]

    def __init__(self, data_td, ref_params, *, Nf, Nt, dt, t0, t_ref,
                 orbits, tdi_config, min_freq, max_freq, sens_model="scirdv1",
                 edge_cut=None, nt_layer=64, n_sparse_fd=1024, m_active_half_width=2,
                 max_r=0.0, tukey_alpha=0.05, n_cp_build=-1,
                 force_backend="cpu"):
        if isinstance(force_backend, str) and force_backend not in ("cpu", "gbgpu_cpu"):
            raise NotImplementedError(
                "GBSignalHetComputations' full data-loading constructor is the "
                "CPU offline/validation path (numpy TDSignal transforms). For "
                "GPU in-model scoring use for_band_engine(), which inherits "
                "the chunked delegate's backend.")
        super().__init__(force_backend=force_backend)
        self.cpp = self.backend.GBComputationGroupWrap()

        self.taper_layers = int(math.ceil(0.5 * float(tukey_alpha) * int(Nt)))
        self.edge_cut = int(_recommended_edge_cut(Nt, tukey_alpha)
                            if edge_cut is None else edge_cut)
        self.m_half = int(m_active_half_width)

        Nobs = Nf * Nt
        Tobs = Nt * Nf * dt
        self.n_cp_build = _resolve_n_cp(n_cp_build, Tobs)
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
            np.zeros(1, dtype=np.int32),          # shared-origin full band
            np.ascontiguousarray(ref_params[None]), 1, 9, 1, 2,
            Nf, Nt, Nf_active, Nt_active, nt_layer, N_sparse_t, stride,
            ind_min_t, ind_min_f, layer_df, dt, Tobs, t0,
            3, n_sparse_fd, tukey_alpha, self.n_cp_build)

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
                       tukey_alpha=tukey_alpha, max_r=max_r, m_half=self.m_half,
                       n_cp_build=self.n_cp_build)

    @property
    def xp(self):
        return self.backend.xp


    _v4_band_arrays = None

    def _make_v4_band_arrays(self):
        """Host-side cardinal weights for the V4 BANDED evaluation mode.

        The knot-values -> pixel-values map of a fixed-knot cubic spline is
        a constant linear operator whose rows decay geometrically (~0.27 per
        knot), so each pixel value is a short dot product against a window of
        node values.  ``v4_band`` (half-band) selects the truncation: 12 ->
        ~1e-7, 16 -> ~1e-9 relative, both far below the pipeline's floors.
        ``v4_band = 0`` returns empty arrays and the kernel takes the
        cooperative spline-solve path instead (both live in one binary so the
        two are directly comparable).
        """
        xp = self.xp
        g = self._g
        band = int(g.get("v4_band", 0))
        if band <= 0:
            return (xp.zeros(1, dtype=xp.float64),
                    xp.zeros(1, dtype=xp.int32), 0)
        from scipy.interpolate import CubicSpline
        import numpy as _np
        K = int(g["v4_knots"])
        tau_k = _np.linspace(0.0, g["Tobs"], K)
        # n_sparse_local lives on the DEVICE under a CUDA backend; the
        # cardinal weights are built once on the host with scipy, so pull
        # it across explicitly (cupy refuses implicit conversion).
        _nsl = self.n_sparse_local
        _nsl = _nsl.get() if hasattr(_nsl, "get") else _np.asarray(_nsl)
        n_glob = (g["ind_min_t"] + _nsl[0]
                  + _np.arange(g["N_sparse_t"]) * g["stride"])
        t_pix = n_glob * g["Nf"] * g["dt"]          # tau at the sparse pixels
        dt_k = tau_k[1] - tau_k[0]
        W = 2 * band
        w_out = _np.zeros((len(t_pix), W))
        j0_out = _np.zeros(len(t_pix), dtype=_np.int32)
        eye = _np.eye(K)
        # cardinal columns evaluated at the pixel times, then truncated to
        # the window centred on each pixel's own knot interval
        L = CubicSpline(tau_k, eye, axis=0)(t_pix)   # (n_pix, K)
        for b in range(len(t_pix)):
            seg = int(_np.clip(t_pix[b] / dt_k, 0, K - 2))
            j0 = int(_np.clip(seg - band + 1, 0, max(K - W, 0)))
            j0_out[b] = j0
            w_out[b] = L[b, j0:j0 + W]
        return (xp.asarray(_np.ascontiguousarray(w_out.ravel())),
                xp.asarray(j0_out), W)

    #: Measured ``n_r`` tiers: ``(T_obs/yr upper bound, n_r)``, ascending.
    #: A baseline at or below a bound takes that row; anything past the last
    #: row takes :attr:`_NR_DEFAULT`. Same shape as ``get_N``'s tiered
    #: lookups -- a table of measurements, deliberately NOT a fitted formula.
    #:
    #: Every row is an anchor that was actually swept with LAT
    #: ``scripts/gb_chunked_het/gb_sighet_tier_assess.py`` (its
    #: ``ii16``/``ii32``/``ii64`` columns ARE this sweep), 8 references x 5
    #: displacement directions x 5 tiers, against the tiered budget
    #: ``allowed(T) ~ max(0.1, T/100)``.
    #:
    #: **Do not interpolate between rows and do not add a row that was not
    #: measured** -- an unmeasured baseline should fall through to the safe
    #: default, not to a guess.
    _NR_TIERS = (
        # 0.25 yr (3 mo), 2026-08-09: n_r=32 is indistinguishable from 64 at
        # every tier -- worst |dLL| 0.052 vs 0.060 at T=1000, ordering
        # flipping inside noise below that, and 100% pass everywhere.
        # n_r=16 also passes 100% but leaves only 1.6x margin at T=10 -- the
        # binding tier -- versus 33x for 32, so the extra 2x is not worth it.
        (0.25, 32),
    )
    #: Baselines past the last measured tier.
    #:
    #: 1.0 yr, 2026-08-10 (same harness, 8 refs x 5 dirs): the picture
    #: INVERTS -- worst |dLL| / pass-rate at T=1000 is
    #: ``n_r=16 -> 350 / 68%``, ``32 -> 49.6 / 85%``, ``64 -> 26.5 / 98%``.
    #: So 32 genuinely IS too coarse at 1 yr, which is what the v4/v5
    #: shootout found; it never contradicted the 0.25 yr row above, because
    #: n_r is a function of T_obs and the two were measured at different
    #: ones. Medians stay small at every n_r (0.09 at T=1000 for 64) -- it is
    #: the WORST references that blow up, so median-only checks miss this.
    #:
    #: NOTE: at 1 yr even n_r=64 does not reach 100% (92-98% by tier). That
    #: residual is NOT an n_r problem -- raising nodes has clearly saturated
    #: by 64 -- it is the known hard-reference population (see the ref02
    #: builder-slip case in gb_sighet_tier_assess.py). Do not try to buy it
    #: back with more nodes.
    _NR_DEFAULT = 64

    def _resolve_v3_nodes(self, x, di):
        """Node count for the v3 ratio build.

        ``v3_n_nodes > 0`` = fixed (the explicit ``SIGHET_V3_NODES`` knob);
        ``-1`` = look the baseline up in :attr:`_NR_TIERS`.

        The lookup keys on ``T_obs`` ALONE. Two reasons, both measured:

        * **Displacement does not need a term.** The 0.25 yr sweep held
          ``n_r = 32`` across the whole displacement range the sampler can
          reach (T=1 through T=1000, the latter being the gate boundary --
          past it candidates are rejected by the trust region / prior
          anyway). So there is nothing for a displacement term to buy inside
          the supported range.
        * **It removes a device sync.** The superseded law reduced a
          per-candidate displacement array to a scalar ``d_max``, forcing one
          device sync per call. A table keyed on a scalar the settings
          already carry needs none.

        Polynomial phase (df0, dfdot) costs no extra nodes at any baseline --
        cubic splines represent it exactly -- so it never enters either way.

        TODO(node-law-snr): raw lnL error scales as SNR^2 x mismatch, so a
        loud source may need a finer fit for the same lnL accuracy. The
        0.25 yr sweep did not vary reference SNR, so there is no measured
        anchor for an SNR axis; add one to :attr:`_NR_TIERS`' shape only
        after sweeping it.
        """
        g = self._g
        v3 = int(g.get("v3_n_nodes", 0))
        if v3 > 0:
            return max(4, v3)
        _YRSID_SI = 31558149.763545603
        t_yr = float(g.get("Tobs", 0.0) or 0.0) / _YRSID_SI
        if t_yr > 0.0:
            for t_hi, n_tier in self._NR_TIERS:
                if t_yr <= t_hi:
                    return int(n_tier)
        return int(self._NR_DEFAULT)

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
                        m_active_half_width=2, max_r=0.0, n_cp_build=-1,
                        v3_n_nodes=0, v4_knots=0, v4_band=0, v5=0):
        """Build a data-less engine-mode instance around ``chunked_comp``.

        ``max_r=0`` disables the kernel's heterodyne-ratio magnitude clip.
        The clip SILENTLY saturates every candidate whose amplitude ratio
        vs the fixed reference exceeds it — at the old default 5.0 that
        corrupted the likelihood of any in-model proposal beyond
        |dlnA| ~ 1.6 (measured: 62% delta error at +2, driving cold-chain
        ll-bookkeeping runaways over long repeat blocks), while with the
        clip off the expansion is exact in amplitude to at least ratio
        e^8 (~3e-6). The FLOOR_EPS reference floor in the kernel already
        guards the divide-by-small on its own. Set ``max_r > 0`` only as
        a diagnostic.

        ``v5`` selects the V5 occupancy experiment (opt-in, default OFF --
        with ``v5=0`` nothing about the v2/v3/v4 paths changes). It only
        applies when ``v4_knots`` is set, since v5 IS v4 with the
        per-candidate fold scratch eliminated: ``r_sparse``/``dr_sparse``
        are an M-fold replication of ``r_pix`` modulated by one
        candidate-independent bit per (row, pixel), so the bit is
        precomputed in :meth:`setup_in_model` and the fold rebuilds r/dr in
        registers. Per-pixel shared cost falls 528 -> 48 B/point, and the
        arithmetic is unchanged, so v5 is BIT-identical to v4.

        * ``v5 = 0`` -- off; ``v4_knots`` routes to the v4 entry point.
        * ``v5 = 1`` -- v5 kernel with the phase-lifetime shared arena
          (regions with disjoint lifetimes overlaid). ~27.6 KB, constant in
          ``N_sparse_t`` up to N ~ 450, so an A100 should go 1 -> 5
          blocks/SM and an H100 1 -> 7. The production candidate.
        * ``v5 = 2`` -- v5 kernel with a FLAT carve: same arithmetic, same
          traffic, ~3 blocks/SM. The A/B that isolates the occupancy change
          from everything else, which is the question the experiment exists
          to answer.

        Grid, orbits, TDI configuration, phase reference time, window and
        taper all come from the chunked delegate's ``wdm_settings`` (the
        run's active-band WDM grid), so the two likelihoods are defined on
        identical grids. The compute backend is INHERITED from the chunked
        delegate (cpu or cudaXXx) so in-model scoring runs where the run
        runs; on CUDA the reference coefficients live on-device and every
        repeat proposal scores through the fused shared-memory kernel with
        zero H2D traffic.
        """
        wdm = chunked_comp.wdm_settings
        self = cls.__new__(cls)
        # backend name is "gbgpu_<flavor>"; re-passing the plain flavor
        # through our ctor re-prefixes it.
        _flavor = chunked_comp.backend.name.split("_", 1)[1]
        FastLISAResponseParallelModule.__init__(self, force_backend=_flavor)
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
        # The polyphase fold identity requires nt_layer to DIVIDE Nt exactly
        # (Nt == nt_layer * stride); the C++ wraps now hard-error otherwise.
        # Production grids need not be power-of-two-friendly (mojito Nt=2160),
        # so SNAP the requested value to the nearest divisor of Nt (ties ->
        # the larger, i.e. denser/more accurate, side).
        nt_layer = int(nt_layer)
        if Nt % nt_layer != 0:
            divisors = [d for d in range(2, Nt + 1) if Nt % d == 0]
            snapped = min(divisors, key=lambda d: (abs(d - nt_layer), -d))
            import logging
            logging.getLogger(__name__).warning(
                "sig-het nt_layer=%d does not divide Nt=%d; snapping to %d "
                "(stride %d).", nt_layer, Nt, snapped, Nt // snapped)
            nt_layer = snapped
        stride, N_sparse_t, n_sparse_local = sparse_time_grid(
            Nt, Nt_active, nt_layer)
        # window + sparse grid go straight into the kernels -> device-resident
        # on CUDA (self.xp follows the inherited backend).
        self.window_full = self.xp.asarray(
            np.asarray(wdm.window.get() if hasattr(wdm.window, "get")
                       else wdm.window, dtype=np.float64))
        self.n_sparse_local = self.xp.asarray(n_sparse_local)

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
                               force_backend=_flavor)
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
                       m_half=self.m_half,
                       n_cp_build=_resolve_n_cp(n_cp_build, Tobs),
                       v3_n_nodes=int(v3_n_nodes), v4_knots=int(v4_knots),
                       v4_band=int(v4_band), v5=int(v5))
        # Deltas are what the in-model repeats consume; keep the chunked
        # delegate's d_d convention so absolute ll values line up too.
        self.d_d = float(getattr(chunked_comp, "d_d", 0.0))
        self._in_model = None
        self._slot_to_ref = None
        self._slot_to_ref_xp = None
        self.c0_mask_all = None
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
        xp = self.xp
        # host copy of the slot ids for the slot->ref map bookkeeping.
        slots = np.asarray(
            data_index.get() if hasattr(data_index, "get") else data_index,
            dtype=int)
        # FORCED device-side copy: params_ref_all must never alias the
        # caller's array (the mid-block patch writes into it). xp.asarray
        # of a host array uploads on CUDA; xp.array(copy=True) covers the
        # already-on-device case.
        refs = xp.array(xp.asarray(params_ref_phys), dtype=float,
                        copy=True).reshape(-1, 9)
        n = refs.shape[0]

        # ---- WINDOWED reference build (EXACT, not an approximation) ------
        # The make_reference kernel only WRITES layers within
        # ceil(n_sparse_fd/Nt)+1 of each reference's carrier; c0 is
        # identically ZERO outside that window, so every bin-fold
        # coefficient outside it is exactly zero too (A ~ data*invC*c0,
        # B ~ c0*invC*c0). All heavy work therefore restricts to a
        # constant-width window of W layers per reference:
        #
        #   * the backend runs PER SOURCE with the grid's ind_min_f SHIFTED
        #     to the window start and Nf_active = W. The kernel's math is
        #     absolute (m_global = ind_min_f + m_local; j_base, kappa and
        #     the parity signs all use m_global), so this is the SAME
        #     computation writing into a compact slab -- no kernel change;
        #   * the bin-fold runs batched over the (n, ..., W, Nt_active)
        #     window slices;
        #   * results scatter into full-band, absolutely-indexed stash
        #     arrays (zeros elsewhere == exactly what the full-band build
        #     produced there), so the get_ll consumer is untouched.
        #
        # This makes the reference build Tobs-independent: full-band it
        # scaled with Nf_active*Nt_active (~2.4 s/source at 1 yr, and a
        # ~54 MB/source dense-c0 transient); windowed it scales with
        # W*Nt_active (W ~ 7-13).
        slab_Nf = getattr(buffer_aca, "band_slab_Nf", None)
        if slab_Nf is not None:
            # NARROW-SLAB buffers (task-b DEFAULT): each slot's slab already
            # IS the validated small window -- the central band plus
            # leakage/guard layers each side, with a per-slot ABSOLUTE layer
            # origin in ``slab_min_f``. The slab doubles as the fold window:
            # the data needs no layer gather at all, and reference c0 beyond
            # the slab edge truncates exactly where the chunked-het task-b
            # kernels truncate, so the two engines score the same domain.
            W = int(slab_Nf)
            _slo = buffer_aca.slab_min_f
            w_lo_host = (np.asarray(
                _slo.get() if hasattr(_slo, "get") else _slo
            )[slots] - int(g["ind_min_f"])).astype(np.int32)
        else:
            # FULL-BAND buffers: carrier-centered window (exact -- see the
            # header comment above; the kernel writes nothing outside it).
            half_ext = (int(math.ceil(g["n_sparse_fd"] / g["Nt"])) + 1
                        + _SIGHET_WINDOW_MARGIN)
            W = min(2 * half_ext + 1, g["Nf_active"])
            f0_host = np.asarray(refs[:, 1].get() if hasattr(refs, "get")
                                 else refs[:, 1])
            m_carrier = np.floor(f0_host / g["layer_df"]).astype(int)
            w_lo_host = np.clip(m_carrier - g["ind_min_f"] - half_ext,
                                0, g["Nf_active"] - W).astype(np.int32)
        w_lo = xp.asarray(w_lo_host)
        layers = w_lo[:, None] + xp.arange(W)[None, :]          # (n, W), local

        # Data-side inputs. The slabs are reshaped VIEWS of the buffer's
        # flat arrays; the per-slot layer width is the slab width on the
        # narrow layout, Nf_active on the full-band one.
        Nf_data = W if slab_Nf is not None else g["Nf_active"]
        res_full = xp.asarray(buffer_aca.linear_data_arr[0]).reshape(
            -1, nch, Nf_data, g["Nt_active"])
        psd_flat = xp.asarray(buffer_aca.linear_psd_arr[0])
        _cell = nch * nch * Nf_data * g["Nt_active"]
        if (psd_flat.size // _cell) * _cell != psd_flat.size:
            raise NotImplementedError(
                "sig-het in-model setup currently supports the XYZ "
                "cross-channel inverse covariance layout only.")
        invC_full = psd_flat.reshape(-1, nch, nch, Nf_data, g["Nt_active"])
        sl = xp.asarray(slots)
        ch = xp.arange(nch)
        if slab_Nf is not None:
            # Slab == window: a plain slot pick copies only the narrow slabs.
            res_w = res_full[sl]
            invC_w = invC_full[sl]
        else:
            # FUSED slot+layer gather: one advanced-index gather pulls only
            # the windowed rows (~W*Nt_active per source) -- the two-step
            # form (full-slab copies, then a layer gather) materialized
            # ~226 MB/source at 1 yr and dominated the windowed setup cost.
            res_w = res_full[sl[:, None, None], ch[None, :, None],
                             layers[:, None, :], :]
            invC_w = invC_full[sl[:, None, None, None], ch[None, :, None, None],
                               ch[None, None, :, None],
                               layers[:, None, None, :], :]

        # Compact windowed c0 slabs -- ONE batched backend call. The kernel
        # takes the per-ref window origins as ``w_lo_arr`` (active-local,
        # added to ind_min_f per reference inside), writing each reference's
        # compact W-wide slab. This replaced a per-source Python loop whose
        # ~8.4 ms/source of per-call GPU API overhead scaled linearly with
        # the block (50 s/iteration at nwalkers=64) and cancelled the
        # sig-het likelihood win.
        c0_sparse_w = xp.zeros((n, nch, W, g["N_sparse_t"]),
                               dtype=xp.complex128)
        c0_dense_w = xp.zeros((n, nch, W, g["Nt_active"]),
                              dtype=xp.complex128)
        self.cpp.gb_signal_het_make_reference(
            self.tdi_wrap, c0_sparse_w, c0_dense_w,
            self.window_full, self.n_sparse_local,
            xp.ascontiguousarray(xp.asarray(w_lo_host, dtype=xp.int32)),
            refs, n, 9, 1, 2,
            g["Nf"], g["Nt"], W, g["Nt_active"],
            g["nt_layer"], g["N_sparse_t"], g["stride"],
            g["ind_min_t"], g["ind_min_f"],
            g["layer_df"], g["dt"], g["Tobs"], g["t0"],
            3, g["n_sparse_fd"], g["tukey_alpha"], g["n_cp_build"])

        # Row helper for the full-band stash expansion below. The scatters
        # use direct advanced-index ASSIGNMENT, not xp.put_along_axis: cupy
        # only grew put_along_axis in v13 and the cluster envs run older
        # cupy (numpy-only local validation cannot see cupy API-surface
        # gaps -- same environment-trap family as module-level ``cp``).
        rows = xp.arange(n)

        # BATCHED bin-fold over the window slices; chunked because the
        # <h|h> intermediates (Ec + En, nch^2 * W * Nt_active complex128
        # per source) remain the setup's memory high-water mark.
        per_src_bytes = 2 * nch * nch * W * g["Nt_active"] * 16
        chunk = max(1, min(n, _SIGHET_FOLD_MAX_BYTES // max(per_src_bytes, 1)))
        folds = [
            bin_fold_real(res_w[s:s + chunk], c0_dense_w[s:s + chunk],
                          invC_w[s:s + chunk], self.n_sparse_local,
                          g["stride"], g["Nt_active"], tdi_type="XYZ")
            for s in range(0, n, chunk)
        ]
        if len(folds) == 1:
            A0s, A1s, B0s, B1s, B0ncs, B1ncs = folds[0]
        else:
            A0s, A1s, B0s, B1s, B0ncs, B1ncs = [
                xp.concatenate(parts, axis=0) for parts in zip(*folds)
            ]

        # Scatter the compact window results into full-band stash arrays
        # (absolute layer indexing, zeros outside -- the consumer kernel is
        # unchanged). A-blocks: (n, nch, Nf_active, N_sparse_t), axis 2;
        # B-blocks: (n, nch, nch, Nf_active, N_sparse_t), axis 3.
        def _expand_A(vals):
            out = xp.zeros((n, nch, g["Nf_active"], g["N_sparse_t"]),
                           dtype=xp.complex128)
            out[rows[:, None, None], ch[None, :, None],
                layers[:, None, :], :] = vals
            return out

        def _expand_B(vals):
            out = xp.zeros((n, nch, nch, g["Nf_active"], g["N_sparse_t"]),
                           dtype=xp.complex128)
            out[rows[:, None, None, None], ch[None, :, None, None],
                ch[None, None, :, None], layers[:, None, None, :], :] = vals
            return out

        c0_sparse = _expand_A(c0_sparse_w)
        A0s, A1s = _expand_A(A0s), _expand_A(A1s)
        B0s, B1s = _expand_B(B0s), _expand_B(B1s)
        B0ncs, B1ncs = _expand_B(B0ncs), _expand_B(B1ncs)
        # The |c0| row-floor mask is the ONLY thing the fold needs from c0,
        # and it depends only on the reference -- so build it here, once,
        # instead of in every candidate's block. See _c0_row_mask_bits.
        c0_mask = _c0_row_mask_bits(c0_sparse, xp)

        if self._in_model is not None:
            # Mid-block PATCH: re-anchor only the given slots (the move's
            # drift refresh). They must already carry a reference. Full-row
            # assignment (not a window write) so a shifted window cannot
            # leave stale coefficients behind.
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
            self.c0_mask_all[ref_idx] = c0_mask
            self.A0_all[ref_idx] = A0s
            self.A1_all[ref_idx] = A1s
            self.B0_all[ref_idx] = B0s
            self.B1_all[ref_idx] = B1s
            self.B0nc_all[ref_idx] = B0ncs
            self.B1nc_all[ref_idx] = B1ncs
            self.params_ref_all[ref_idx] = refs
            return True

        # The coefficient stash is the per-block CACHE: built once here (on
        # the run's device), then reused by every repeat-proposal get_ll
        # with no further host<->device traffic. (Freshly allocated by the
        # expanders above, so already contiguous.)
        self.c0_sparse_all = c0_sparse
        # v5's view of c0. Cheap (1/128 the size of c0_sparse_all) and built
        # unconditionally so the v5 A/B never carries a setup-cost asymmetry
        # against v4; v2/v3/v4 simply never read it.
        self.c0_mask_all = c0_mask
        self.A0_all = A0s
        self.A1_all = A1s
        self.B0_all = B0s
        self.B1_all = B1s
        self.B0nc_all = B0ncs
        self.B1nc_all = B1ncs
        self.params_ref_all = refs

        slot_map = np.full(int(slots.max()) + 1, -1, dtype=int)
        slot_map[slots] = np.arange(n)
        self._slot_to_ref = slot_map
        # Device mirror for the per-repeat hot path (get_ll_wdm); the host
        # copy above stays for the once-per-refresh patch bookkeeping.
        self._slot_to_ref_xp = xp.asarray(slot_map)
        self._in_model = True
        return True

    def clear_in_model(self) -> None:
        """Deactivate the in-model reference: get_ll_wdm routes back to the
        chunked delegate (RJ / removal / any out-of-block scoring)."""
        self._in_model = None
        self._slot_to_ref = None
        self._slot_to_ref_xp = None
        self.c0_mask_all = None

    # ---- band-engine surface (chunked delegate + in-model routing) -------

    def fill_global_wdm(self, *args, **kwargs):
        return self.chunked.fill_global_wdm(*args, **kwargs)

    def get_swap_ll_wdm(self, *args, **kwargs):
        return self.chunked.get_swap_ll_wdm(*args, **kwargs)

    def get_ll_grad_wdm(self, *args, **kwargs):
        return self.chunked.get_ll_grad_wdm(*args, **kwargs)

    def hessian_wdm(self, *args, **kwargs):
        return self.chunked.hessian_wdm(*args, **kwargs)

    def information_matrix(self, params, wdm_holder, inds=None, param_eps=None,
                           noise_index=None, data_index=None, **kwargs):
        """Information matrix; sig-het second-difference path when armed.

        ``SIGHET_INFOMAT=1`` routes to
        :func:`lisatools.info_matrix_ll.information_matrix_from_ll`, driven by
        THIS object's in-model scorer -- so every evaluation reuses the block
        reference ``setup_in_model`` already built and costs one fast candidate
        score instead of a chunked-het swap launch. Requires an active in-model
        block (``_in_model``); otherwise, and whenever the knob is off, falls
        through to the validated chunked delegate.

        Cost target: the point is to amortize against the block's repeats.
        Per 120-source block, ~163 evaluations/source at sig-het speed is
        ~0.3 s against ~2.3 s of repeats (~13%); the chunked path is ~5.6 s
        (~250%).
        """
        from lisatools.info_matrix_ll import (
            infomat_knob, information_matrix_from_ll,
        )

        # ``data_index`` here means BUFFER SLOT (get_ll_wdm maps it through
        # ``_slot_to_ref`` while an in-model reference is live), NOT the
        # walker. It must be supplied explicitly: falling back to
        # ``noise_index`` would silently reinterpret walker indices as slots
        # -- in range, so nothing raises, and the proposal covariance comes
        # out of the WRONG sources' references. Without it, take the
        # validated chunked path.
        if (not infomat_knob("SIGHET_INFOMAT", False)
                or not self._in_model
                or data_index is None):
            return self.chunked.information_matrix(
                params, wdm_holder, inds=inds, param_eps=param_eps,
                noise_index=noise_index, **kwargs)

        xp = self.xp
        p = xp.asarray(xp.atleast_2d(params))
        if param_eps is None:
            param_eps = self.chunked.default_param_eps(p.shape[1]) \
                if hasattr(self.chunked, "default_param_eps") \
                else xp.full(p.shape[1], 1e-7)
        eps = xp.asarray(param_eps, dtype=xp.float64) * float(
            infomat_knob("SIGHET_INFOMAT_EPS_SCALE", 1.0))

        di = xp.asarray(data_index)

        def _call_ll(rows):
            # Rows are the SAME sources repeated per corner, so the data_index
            # tiles with them.
            reps = int(rows.shape[0] // di.shape[0])
            return self.get_ll_wdm(rows, wdm_holder,
                                   data_index=xp.tile(di, reps),
                                   noise_index=xp.tile(di, reps))

        G = information_matrix_from_ll(
            _call_ll, p, xp=xp, param_eps=eps, inds=inds,
            one_sided=bool(infomat_knob("SIGHET_INFOMAT_ONESIDED", False)),
        )

        if infomat_knob("SIGHET_INFOMAT_VALIDATE", False):
            ref = self.chunked.information_matrix(
                params, wdm_holder, inds=inds, param_eps=param_eps,
                noise_index=noise_index, **kwargs)
            num = xp.abs(G - ref)
            den = xp.maximum(xp.abs(ref), 1e-30)
            logger.warning(
                "[SIGHET_INFOMAT_VALIDATE] max reldiff %.3e | median %.3e "
                "| max|ref| %.3e", float(xp.max(num / den)),
                float(xp.median(num / den)), float(xp.max(xp.abs(ref))))
        return G

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
        # Slot -> reference lookup stays ON DEVICE. The old path did
        # ``data_index.get()`` (a full D2H of the slot array, forcing a sync),
        # a numpy gather, then an H2D re-upload -- on EVERY repeat proposal.
        # The validity test would sync again.
        #
        # Instead: CLAMP to an in-range, non-negative reference so the kernel
        # can never read out of bounds, run it, and raise AFTERWARDS. The
        # kernel wrap already synchronizes, so by then the check is a single
        # scalar read off an idle stream rather than an extra round trip.
        xp = self.xp
        di = xp.asarray(data_index)
        n_slots = int(self._slot_to_ref_xp.shape[0])
        di_ok = xp.clip(di, 0, n_slots - 1)
        ref_raw = self._slot_to_ref_xp[di_ok]
        bad = (di != di_ok) | (ref_raw < 0)
        ref_idx = xp.where(bad, 0, ref_raw)
        # params stay on their resident device (cupy on the CUDA path).
        ll = self.get_ll(self.xp.asarray(params, dtype=float),
                         data_index=ref_idx)
        if bool(bad.any()):
            raise RuntimeError(
                "sig-het in-model scoring hit a buffer slot with no "
                "reference; setup_in_model was not run for it.")
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
        xp = self.xp
        if phase_maximize:
            ll_0 = self.get_ll(params, data_index=data_index)
            d_h_0, h_h = self.last_d_h.copy(), self.last_h_h.copy()
            # FORCED copy: the quadrature shift must not mutate the
            # caller's params through an ascontiguousarray view.
            x_q = xp.atleast_2d(xp.array(xp.asarray(params), dtype=float,
                                         copy=True))
            x_q[:, 4] = x_q[:, 4] + np.pi / 2
            self.get_ll(x_q, data_index=data_index)
            d_h_90 = self.last_d_h.copy()
            D = d_h_0 + 1j * d_h_90
            d_h_max = xp.abs(D)
            self.non_marg_d_h = d_h_0
            self.phase_angle = xp.arctan2(D.imag, D.real)
            self.last_d_h = d_h_max
            self.last_h_h = h_h
            return ll_0 + (d_h_max - d_h_0)
        self.phase_angle = None
        # Arrays live on the run's device (cupy on CUDA): candidate params
        # are typically already device-resident, the coefficient stash always
        # is (built by setup_in_model), so a repeat-proposal call moves
        # nothing across the PCIe bus.
        x = xp.ascontiguousarray(xp.atleast_2d(xp.asarray(params, dtype=float)))
        N = x.shape[0]
        di = (xp.zeros(N, dtype=xp.int32) if data_index is None
              else xp.ascontiguousarray(xp.asarray(data_index, dtype=xp.int32)))
        d_h = xp.zeros(N, dtype=xp.float64)
        h_h = xp.zeros(N, dtype=xp.float64)
        num_data = int(self.params_ref_all.shape[0])
        g = self._g
        if g.get("v4_knots", 0) and self._v4_band_arrays is None:
            self._v4_band_arrays = self._make_v4_band_arrays()
        # Which scorer actually ran -- logged ONCE per comp. The settings
        # guard upstream only proves the KNOBS were consistent; it cannot
        # show that the v5 branch was taken at runtime. overnight_v5 asked
        # for v5 and measured no speedup, and nothing in the log could
        # distinguish "v5 ran and did not help" from "v5 silently fell
        # through to v3/v2" -- so say it out loud.
        if not getattr(self, "_kernel_path_logged", False):
            self._kernel_path_logged = True
            if g.get("v4_knots", 0) and int(g.get("v5", 0)):
                _p = f"v5 (mode {1 if int(g['v5']) == 1 else 2})"
            elif g.get("v4_knots", 0):
                _p = "v4"
            elif g.get("v3_n_nodes", 0):
                _p = "v3"
            else:
                _p = "v2"
            logger.info(
                "sig-het scorer: %s  [v3_n_nodes=%s v4_knots=%s v4_band=%s "
                "v5=%s]", _p, g.get("v3_n_nodes", 0), g.get("v4_knots", 0),
                g.get("v4_band", 0), g.get("v5", 0),
            )
        if g.get("v4_knots", 0) and int(g.get("v5", 0)):
            # V5: v4 with the per-candidate fold scratch (r_sparse /
            # dr_sparse) ELIMINATED -- they are an M-fold replication of
            # r_pix modulated by the candidate-independent |c0| row-floor
            # mask that setup_in_model already built, so the scorer rebuilds
            # r/dr in registers and reads ``c0_mask_all`` instead of
            # ``c0_sparse_all``. Identical algebra, identical accumulation
            # order -> bit-identical to v4. ``v5 == 1`` -> phase-aliased
            # shared arena (~5 blocks/SM on an A100 against v4's 1);
            # ``v5 == 2`` -> flat carve at the same arithmetic and traffic
            # (~3 blocks/SM), the A/B that isolates occupancy. Opt-in;
            # default 0 never reaches here.
            _v5_mode = 1 if int(g["v5"]) == 1 else 2
            if self.c0_mask_all is None:
                raise RuntimeError(
                    "sig-het v5 needs the |c0| row-floor mask, which "
                    "setup_in_model builds alongside the coefficient stash; "
                    "no in-model reference is active.")
            self.cpp.gb_signal_het_v5_get_ll(
                self.tdi_wrap, d_h, h_h, self.c0_mask_all,
                self.A0_all, self.A1_all, self.B0_all, self.B1_all,
                self.B0nc_all, self.B1nc_all,
                self.n_sparse_local,
                self._v4_band_arrays[0], self._v4_band_arrays[1],
                self._v4_band_arrays[2],
                x, self.params_ref_all, di,
                N, num_data,
                self._resolve_v3_nodes(x, di), int(g["v4_knots"]),
                9, 1, 2,
                g["Nf"], g["Nt"], g["Nf_active"], g["Nt_active"],
                g["nt_layer"], g["N_sparse_t"], g["stride"],
                g["ind_min_t"], g["ind_min_f"], g["m_half"],
                g["layer_df"], g["dt"], g["Tobs"], g["t0"],
                3, 0, 1,                      # XYZ, project_real=1
                _v5_mode)
            self.last_d_h = d_h.copy()
            self.last_h_h = h_h.copy()
            return -0.5 * self.d_d + d_h - 0.5 * h_h
        if g.get("v4_knots", 0):
            # V4: fixed-knot ratio evaluation. Same node-eval + log-polar
            # spline FIT as v3, but the fitted ratio is RESAMPLED onto
            # n_knots FIXED (candidate-independent) uniform knot times as
            # linear complex values, and the pixel-time evaluation goes
            # through the fixed-knot not-a-knot cubic spline whose Thomas
            # factorization + per-pixel segment map are precomputed once
            # per call. Consumes the SAME stash as v2/v3. Takes precedence
            # over v3 when both knobs are set (the fit node count still
            # honors v3_n_nodes / the adaptive resolve).
            self.cpp.gb_signal_het_v4_get_ll(
                self.tdi_wrap, d_h, h_h, self.c0_sparse_all,
                self.A0_all, self.A1_all, self.B0_all, self.B1_all,
                self.B0nc_all, self.B1nc_all,
                self.n_sparse_local,
                self._v4_band_arrays[0], self._v4_band_arrays[1],
                self._v4_band_arrays[2],
                x, self.params_ref_all, di,
                N, num_data,
                self._resolve_v3_nodes(x, di), int(g["v4_knots"]),
                9, 1, 2,
                g["Nf"], g["Nt"], g["Nf_active"], g["Nt_active"],
                g["nt_layer"], g["N_sparse_t"], g["stride"],
                g["ind_min_t"], g["ind_min_f"], g["m_half"],
                g["layer_df"], g["dt"], g["Tobs"], g["t0"],
                3, 0, 1)                      # XYZ, project_real=1
            self.last_d_h = d_h.copy()
            self.last_h_h = h_h.copy()
            return -0.5 * self.d_d + d_h - 0.5 * h_h
        if g.get("v3_n_nodes", 0):
            # V3: ratio-spline candidate build straight into the bin-fold
            # (no FFT / polyphase / division). Consumes the SAME stash as
            # v2 -- setup_in_model needs no v3-specific work.
            self.cpp.gb_signal_het_v3_get_ll(
                self.tdi_wrap, d_h, h_h, self.c0_sparse_all,
                self.A0_all, self.A1_all, self.B0_all, self.B1_all,
                self.B0nc_all, self.B1nc_all,
                self.n_sparse_local,
                x, self.params_ref_all, di,
                N, num_data,
                self._resolve_v3_nodes(x, di), 9, 1, 2,
                g["Nf"], g["Nt"], g["Nf_active"], g["Nt_active"],
                g["nt_layer"], g["N_sparse_t"], g["stride"],
                g["ind_min_t"], g["ind_min_f"], g["m_half"],
                g["layer_df"], g["dt"], g["Tobs"], g["t0"],
                3, 0, 1)                      # XYZ, project_real=1
            self.last_d_h = d_h.copy()
            self.last_h_h = h_h.copy()
            return -0.5 * self.d_d + d_h - 0.5 * h_h
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
            g["tukey_alpha"], g["max_r"], 1,     # project_real=1
            g["n_cp_build"])
        self.last_d_h = d_h.copy()
        self.last_h_h = h_h.copy()
        return -0.5 * self.d_d + d_h - 0.5 * h_h
