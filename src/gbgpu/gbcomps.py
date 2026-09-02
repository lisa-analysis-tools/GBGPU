# Phase 3L.7i (2026-06-04): GB chunked-heterodyne Python frontend moved
# from lisa-on-gpu's fastlisaresponse package to GBGPU as part of the
# lisa-on-gpu full-retirement work. lisa-on-gpu keeps a thin deprecation
# shim at fastlisaresponse.gbcomps that re-exports from here.
#
# Phase 3L.7p-followup (2026-06-06): the shared chunked-heterodyne
# Python frontend (the ~885-line ``GBWDMComputations`` class body) and
# its helper functions moved further to ``lisatools.chunked_het`` and
# ``lisatools.wdm_het`` so the SOBBH side (which now lives in
# ``bbhx.sobbhcomps``) can share them without introducing a BBHx ->
# GBGPU import dependency. ``GBWDMComputations`` here is now a thin
# GB-physics sub-class that only sets the routing constants.
#
# Architectural rule (sprint-wide, user direction 2026-06-06):
#   * GB-specific code     -> GBGPU
#   * SOBBH-specific code  -> BBHx
#   * Shared GB <-> SOBBH  -> LISAanalysistools (lisatools)
import os
from copy import deepcopy
from typing import Optional

import numpy as np

from lisatools.chunked_het import WDMComputationsBase
from lisatools.detector import EqualArmlengthOrbits, Orbits
from lisatools.domains import WDMSettings
from lisatools.response.directresponse import (
    ecliptic_to_icrs,
    warn_deprecated_frame_conversion,
)
from lisatools.response.tdiconfig import TDIConfig

# Phase 3L.7k (2026-06-04): GBFDComputations (the FD chunked-heterodyne
# analog of WDMComputationsBase) still dispatches through gbgpu's own
# ParallelModule (prefix gbgpu_) because it needs the GB-specific Wraps
# (GBTDIonTheFlyWrap, GBComputationGroupWrap) that only the
# gbgpu_<flavor> composed backend carries. ``GBWDMComputations`` inherits
# from the lisatools shared base which already supports per-class
# ``_BACKEND_PREFIX`` overrides, so it no longer needs the gbgpu-local
# alias -- it just sets ``_BACKEND_PREFIX = "gbgpu"`` directly.
from .parallelbase import GBGPUParallelModule as FastLISAResponseParallelModule

from .wdm_het import USE_RECOMMENDED_TUKEY

# Index-bound sanity asserts (debugging aids): each ``int(arr.max())`` is a
# full device sync per engine call on CuPy. Default OFF in production;
# opt back in with GB_INDEX_ASSERTS=1. Checked ONCE at import.
_GB_INDEX_ASSERTS = os.environ.get("GB_INDEX_ASSERTS", "0") == "1"

# Fused phase-max sign contract for the FD family (see the chunked-family
# constants in lisatools.chunked_het and _QUAD_SIGN_SIGHET in
# gbsignalhetcomputations for the full story). Normalize the raw kernel Im
# outputs to EXACTLY "the same inner product at phi0 + pi/2". Pinned
# EMPIRICALLY by tests/test_phase_max_fused.py; the swap cross term
# <h_add|h_rem> holds the add template in the CONJUGATED slot, so its
# phasor turns opposite to <d|h_add>'s -- hence its own constant.
_QUAD_SIGN_FD = +1.0
_QUAD_SIGN_FD_SWAP_ADD = +1.0
# Pinned -1 by test_phase_max_fused (2026-08-27): the conjugated add slot
# flips the phasor exactly as predicted.
_QUAD_SIGN_FD_SWAP_CROSS = -1.0


class GBWDMComputations(WDMComputationsBase):
    """Source-side WDM-domain GB likelihood (chunked-heterodyne path).

    Routes through the chunked-heterodyne kernel set
    ``gb_wdm_het_{fill_global, get_ll, swap_ll}`` on the backend (C++
    on CPU / CUDA, JAX on the ``jax`` backend).

    See :class:`lisatools.chunked_het.WDMComputationsBase` for the
    full method surface (``fill_global_wdm`` / ``get_ll_wdm`` /
    ``get_swap_ll_wdm`` / ``get_ll_grad_wdm`` / ``get_fstat_ll_wdm`` /
    ``get_swap_ll_grad_wdm``). This sub-class only specifies the
    GB-specific routing constants below; everything else is inherited.

    Param order (matches ``GBTDIonTheFly``):
        ``(amp, f0, fdot, fddot, phi0, inc, psi, lam, beta)``.
    """

    # GB chunked-het uses gbgpu_<flavor> backends (carry
    # GBComputationGroupWrap + the gb_wdm_het_* kernel family).
    _BACKEND_PREFIX = "gbgpu"
    # GBGPU's gb_wdm_het_{get_ll, swap_ll} bindings carry the trailing fused
    # phase-max quadrature arrays (see WDMComputationsBase._FUSED_QUAD_KERNELS).
    _FUSED_QUAD_KERNELS = True
    # GBGPU's gb_wdm_het_get_fstat_ll binding carries the trailing basis-filter
    # fold argument (see WDMComputationsBase._FSTAT_FOLD_KERNELS). The fold
    # itself now defaults ON (2026-08-28); GB_FSTAT_FOLD=0 restores the
    # 4-filter path bit-for-bit.
    #
    # This flag is a hard constant, NOT probed from the compiled binding, so a
    # STALE .so is a loud TypeError at the first F-stat call (the trailing arg
    # is passed unconditionally, fold on or off) rather than a silent no-op.
    _FSTAT_FOLD_KERNELS = True
    # Same contract for the trailing N_cp_orbit arg (orbit spline cache,
    # 2026-09-01): hard constant, passed unconditionally (cache on or off), so
    # a stale .so is again a loud TypeError -- which also makes THIS rebuild
    # boundary loud for the whole 09-01 kernel batch (per-m fold completion +
    # invC hoist ride the same recompile).
    _FSTAT_ORBIT_CACHE_KERNELS = True
    _WRAP_ATTR = "GBComputationGroupWrap"
    _METHOD_PREFIX = "gb_wdm_het"
    _NPARAMS = 9
    _F0_PARAM_INDEX = 1   # GBTDIonTheFly: params[1] = f0


class _FDHolderBinding:
    """One holder's live FD-domain binding (see ``GBFDComputations._fd_binding``).

    Keeps the C-side :class:`FDDomainWrap` alive together with every array it
    holds by pointer (data, invC, per-row window starts). Cached on the holder
    object itself so repeated calls against the same buffers are zero-cost.
    """

    __slots__ = ("cpp_fd", "data_flat", "invc_flat", "starts",
                 "num_rows", "num_noise", "n_rfft", "windowed", "token")


class GBFDComputations(FastLISAResponseParallelModule):
    """Frequency-domain heterodyne analog of :class:`GBWDMComputations`.

    Mirrors the WDM ``get_ll_wdm`` / ``get_swap_ll_wdm`` / ``fill_global``
    surface for the sparse FD heterodyne kernel.  All inner products are
    accumulated in C using the standard lisatools FD inner product
    ``(a|b) = 4 Re sum_{c1,c2} sum_k conj(a_c1[k]) b_c2[k] invC[c1,c2][k] * df``,
    so calling :func:`lisatools.diagnostic.inner_product` on the same data /
    invC / template gives an identical answer up to floating-point round-off.
    """

    def __init__(self, fd_settings, t_ref,
                 N_sparse=256,
                 orbits=None, tdi_config=None, force_backend=None,
                 d_d=0.0, tdi_type="XYZ", nchannels=3,
                 tukey_alpha=0.0, edge_frac=0.0, t_start=None):
        """Args:
            fd_settings: :class:`lisatools.domains.FDSettings` instance
                describing the rfft grid (``N``, ``df``) and the active
                band (``ind_min`` / ``ind_max`` from ``min_freq`` /
                ``max_freq``). Mirrors ``GBWDMComputations(wdm_settings,
                t_ref, ...)`` -- the comp is CONFIG-ONLY at construction;
                data / inverse-covariance holders (an
                :class:`AnalysisContainerArray`, a single
                :class:`AnalysisContainer`, or -- for ``fill_global``
                targets -- a :class:`DomainBase` signal) are passed to
                each compute method at call time.
            t_ref: source-phase reference time in seconds.
            N_sparse: default heterodyne FFT length (power of two);
                per-call ``N_sparse=`` overrides remain available.
            orbits, tdi_config, tdi_type, d_d, force_backend: as on
                :class:`GBWDMComputations`.
            nchannels: TDI channel count of the holders this comp will
                be called with (default 3).
            tukey_alpha, edge_frac: FD-het window controls (see the
                attribute comments below).
            t_start: heterodyne start time; must equal ``t_ref`` (the
                kernels assume a unity phase factor). Default ``None``
                means ``t_ref``.
        """
        super().__init__(force_backend=force_backend)
        # Multi-GPU replica metadata -- the WDM twin's contract verbatim
        # (see ``lisatools.chunked_het.WDMComputationsBase.__init__``): the
        # ``cpp_orbits`` / ``cpp_tdi_config`` wraps this constructor builds
        # hold device pointers fixed at the device current right now, so
        # record that device plus the arguments needed to rebuild an
        # equivalent comp inside another shard's device context.
        from lisatools.utils.device import current_device as _current_device

        self._build_device = _current_device(self.xp)
        self._ctor_args = (fd_settings, t_ref)
        self._ctor_kwargs = dict(
            N_sparse=N_sparse, orbits=orbits, tdi_config=tdi_config,
            force_backend=force_backend, d_d=d_d, tdi_type=tdi_type,
            nchannels=nchannels, tukey_alpha=tukey_alpha,
            edge_frac=edge_frac, t_start=t_start,
        )
        from lisatools.domains import FDSettings as _FDSettings
        if not isinstance(fd_settings, _FDSettings):
            raise TypeError(
                "GBFDComputations now takes an FDSettings instance as its "
                "first argument and receives data at call time (2026-07 GB "
                "rework). The legacy constructor "
                "GBFDComputations(T, t_ref, t_start, N_sparse, df, data_fd, "
                "invC, ...) is retired: build "
                "FDSettings(N=n_rfft, df=df, min_freq=..., max_freq=...) and "
                "pass the data holder (AnalysisContainerArray / "
                "AnalysisContainer) to get_ll_fd / fill_global / "
                f"information_matrix instead. Got {type(fd_settings).__name__}."
            )
        if N_sparse < 1 or (N_sparse & (N_sparse - 1)) != 0:
            raise ValueError("N_sparse must be a power of two.")
        if t_start is None:
            t_start = t_ref
        if abs(float(t_start) - float(t_ref)) > 1e-9:
            raise ValueError("GBFDComputations requires t_start == t_ref so "
                             "the heterodyne phase factor is unity.")
        if tdi_type not in {"XYZ", "AET", "AE"}:
            raise ValueError("tdi_type must be one of 'XYZ', 'AET', 'AE'.")

        self.fd_settings = fd_settings
        self.df = float(fd_settings.df)
        self.T = 1.0 / self.df
        self.ind_min = int(fd_settings.ind_min)
        self.ind_max = int(fd_settings.ind_max)
        self.nchannels = int(nchannels)

        self.t_ref = float(t_ref)
        self.t_start = float(t_start)
        self.N_sparse = int(N_sparse)
        self.d_d = float(d_d)
        self.tdi_type = tdi_type
        # Tukey taper (scipy.signal.windows.tukey alpha) applied to the slow
        # signal before the sparse-FD FFT, mirroring rfft(Tukey*td). MUST match
        # the window used to build the data rows -- with the same alpha the FD
        # heterodyne reproduces the WDM-domain mismatch (mm ~ 1e-11); the
        # default 0.0 (rectangular) matches lisatools' unwindowed FD reference.
        self.tukey_alpha = float(tukey_alpha)
        # Edge-cut fraction (EC/Nt): zero the first/last edge_frac of the sparse
        # grid so the FD-het template analyses the SAME time region as a WDM grid
        # with min_time=EC*layer, max_time=(Nt-EC)*layer. Set this (and window the
        # data the same way) to make the FD-het off-source logL match the WDM.
        self.edge_frac = float(edge_frac)

        self.orbits = orbits
        self.tdi_config = tdi_config

    @property
    def xp(self): return self.backend.xp

    @property
    def args(self) -> tuple:
        """Positional arguments for recreating this comp (``(fd_settings, t_ref)``).

        Same contract as :attr:`lisatools.chunked_het.WDMComputationsBase.args`:
        the per-device replica helper rebuilds
        ``type(comp)(*comp.args, **comp.kwargs)`` inside the owning device
        context so the ``*Wrap`` pointer fields are device-local.
        """
        return self._ctor_args

    @property
    def kwargs(self) -> dict:
        """Keyword arguments for recreating this comp (see :attr:`args`)."""
        return dict(self._ctor_kwargs)

    @property
    def orbits(self): return self._orbits
    @orbits.setter
    def orbits(self, o):
        if o is None:
            o = EqualArmlengthOrbits()
        elif not isinstance(o, Orbits) and issubclass(o, Orbits):
            o = o()
        else:
            assert isinstance(o, Orbits)
        self._orbits = deepcopy(o)
        # pycppdetector_args triggers lazy configuration if needed.
        self.cpp_orbits = self.backend.OrbitsWrap(
            *self._orbits.pycppdetector_args)

    @property
    def tdi_config(self): return self._tdi_config
    @tdi_config.setter
    def tdi_config(self, tc):
        if tc is None:
            tc = TDIConfig("1st generation")
        elif isinstance(tc, str):
            tc = TDIConfig(tc)
        elif not isinstance(tc, TDIConfig):
            raise ValueError("tdi_config must be TDIConfig, str, or None.")
        self._tdi_config = tc
        self.cpp_tdi_config = self.backend.TDIConfigWrap(
            *self._tdi_config.pytdiconfig_args)

    @classmethod
    def supported_backends(cls):
        return [cls._BACKEND_PREFIX + "_" + _t for _t in cls.GPU_RECOMMENDED()]

    def _prep_params(self, params, convert_to_ra_dec):
        p = self.xp.asarray(self.xp.atleast_2d(params)).copy()
        if convert_to_ra_dec:
            # Deprecated legacy path: sky coords are consumed in the
            # orbits frame directly (matching the TDI-on-the-fly handling).
            warn_deprecated_frame_conversion()
            lam = p[:, -2].copy(); beta = p[:, -1].copy()
            lam, beta = ecliptic_to_icrs(lam, beta)
            p[:, -2] = lam; p[:, -1] = beta
        return p

    # ---------- call-time data binding ------------------------------------
    #
    # The comp is config-only; every compute method takes an ``fd_holder``
    # (AnalysisContainerArray or AnalysisContainer) whose live linear
    # buffers are bound zero-copy into a per-holder FDDomainWrap. Mirrors
    # ``WDMComputationsBase._as_wdm_holder``.

    # Effective ind_max for windowed-row holders: the per-row starts already
    # restrict the analysed bins, so the absolute band bounds must not clip
    # the +/- N/2 template margins at the window edges.
    _WINDOW_IND_MAX = 2 ** 30

    def _as_fd_holder(self, fd_holder):
        """Normalise ``fd_holder`` to something exposing the flat buffers."""
        if hasattr(fd_holder, "linear_data_arr"):
            return fd_holder
        from lisatools.analysiscontainer import (
            AnalysisContainer, AnalysisContainerArray)
        if isinstance(fd_holder, AnalysisContainer):
            arr = getattr(getattr(fd_holder, "data_res_arr", None), "arr", None)
            dev = getattr(getattr(arr, "device", None), "id", None)
            gpus = None if dev is None else [int(dev)]
            return AnalysisContainerArray(fd_holder, gpus=gpus)
        raise TypeError(
            "fd_holder must be an AnalysisContainerArray or "
            f"AnalysisContainer; got {type(fd_holder).__name__}."
        )

    def _holder_starts(self, holder, num_rows):
        """Per-row absolute window-start bins ``(starts_int32, windowed)``.

        Sub-band buffers expose ``min_freq_inds`` -- a persistent int32
        store the buffer updates IN PLACE on cell swaps; binding it by
        pointer keeps the wrap current forever. Bare ACAs fall back to
        their per-container ``start_freq_ind`` (one shared global-grid
        offset per row); anything unresolvable starts at bin 0.
        """
        starts = getattr(holder, "min_freq_inds", None)
        if starts is not None:
            starts = self.xp.ascontiguousarray(starts, dtype=self.xp.int32)
            if starts.shape != (num_rows,):
                raise ValueError(
                    f"holder.min_freq_inds has shape {starts.shape}; "
                    f"expected ({num_rows},).")
            return starts, True
        sfi = getattr(holder, "start_freq_ind", None)
        try:
            starts = self.xp.ascontiguousarray(
                self.xp.asarray(sfi), dtype=self.xp.int32)
        except (TypeError, ValueError):
            starts = self.xp.zeros(num_rows, dtype=self.xp.int32)
        if starts.ndim == 0:
            starts = self.xp.full(num_rows, int(starts), dtype=self.xp.int32)
        if starts.shape != (num_rows,):
            raise ValueError(
                f"holder.start_freq_ind resolves to shape {starts.shape}; "
                f"expected ({num_rows},) or a scalar.")
        return starts, False

    def _fd_binding(self, fd_holder) -> _FDHolderBinding:
        """The :class:`FDDomainWrap` bound to this holder's live rows.

        Cached on the holder; the token guards the identity of the flat
        buffers, so in-place updates (residual fills, ``min_freq_inds``
        window moves) never invalidate it while a swapped-out array does.
        Complex parent PSDs are snapshotted to their real part (the gb_fd
        kernels consume real invC); real PSDs (sub-band buffers) bind
        zero-copy and stay live.
        """
        holder = self._as_fd_holder(fd_holder)
        if len(holder.linear_data_arr) != 1:
            raise NotImplementedError(
                "GBFDComputations is single-shard by contract. Multi-shard "
                "holders must go through the LAT shard router "
                "(lisatools.globalfit.moves.gbbands._RoutedBandEngine), "
                "which presents one per-split view per call.")
        data_flat = holder.linear_data_arr[0]
        psd_flat = holder.linear_psd_arr[0]
        num_rows = int(holder.acs_total_entries)
        n_rfft = int(data_flat.size) // (num_rows * self.nchannels)
        if num_rows * self.nchannels * n_rfft != int(data_flat.size):
            raise ValueError(
                f"holder data size {int(data_flat.size)} is not "
                f"num_rows({num_rows}) * nchannels({self.nchannels}) * "
                "n_bins.")

        token = (id(data_flat), id(psd_flat), num_rows, n_rfft,
                 self.nchannels, self.tdi_type, self.ind_min, self.ind_max,
                 float(self.df))
        b = getattr(holder, "_gb_fd_binding", None)
        if b is not None and b.token == token:
            return b

        per_noise = (self.nchannels * self.nchannels
                     if self.tdi_type == "XYZ" else self.nchannels) * n_rfft
        num_noise = int(psd_flat.size) // per_noise
        if num_noise * per_noise != int(psd_flat.size):
            raise ValueError(
                f"holder psd size {int(psd_flat.size)} is not a multiple of "
                f"the per-noise-row size {per_noise} for tdi_type="
                f"{self.tdi_type}.")

        if data_flat.dtype != self.xp.complex128:
            raise ValueError(
                "holder data rows must be complex128 (bound zero-copy into "
                f"the C-side FDDomain); got {data_flat.dtype}.")

        if self.xp.iscomplexobj(psd_flat):
            invc_flat = self.xp.ascontiguousarray(
                psd_flat.real.reshape(-1).astype(self.xp.float64))
        else:
            invc_flat = self.xp.ascontiguousarray(
                psd_flat.reshape(-1), dtype=self.xp.float64)

        starts, windowed = self._holder_starts(holder, num_rows)
        ind_min_eff, ind_max_eff = (
            (0, self._WINDOW_IND_MAX) if windowed
            else (self.ind_min, self.ind_max)
        )

        b = _FDHolderBinding()
        b.cpp_fd = self.backend.FDDomainWrap(
            data_flat.reshape(-1), invc_flat,
            n_rfft, self.nchannels, num_rows, num_noise,
            ind_min_eff, ind_max_eff, self.df,
            starts,
        )
        b.data_flat = data_flat
        b.invc_flat = invc_flat
        b.starts = starts
        b.num_rows = num_rows
        b.num_noise = num_noise
        b.n_rfft = n_rfft
        b.windowed = windowed
        b.token = token
        holder._gb_fd_binding = b
        return b

    def get_ll_fd(self, params, fd_holder, data_index=None, noise_index=None,
                  convert_to_ra_dec: Optional[bool] = None,
                  N_sparse: Optional[int] = None):
        b = self._fd_binding(fd_holder)
        p = self._prep_params(params, convert_to_ra_dec)
        num_bin = p.shape[0]
        d_h_out = self.xp.zeros(num_bin)
        h_h_out = self.xp.zeros(num_bin)
        d_h_im = self.xp.zeros(num_bin)   # fused phase-max quadrature
        if data_index is None:
            data_index = self.xp.zeros(num_bin, dtype=self.xp.int32)
        else:
            data_index = self.xp.asarray(data_index).astype(self.xp.int32)
        if noise_index is None:
            noise_index = self.xp.zeros(num_bin, dtype=self.xp.int32)
        else:
            noise_index = self.xp.asarray(noise_index).astype(self.xp.int32)
        if _GB_INDEX_ASSERTS:
            assert int(data_index.max()) < b.num_rows
            assert int(noise_index.max()) < b.num_noise

        self.backend.GBComputationGroupWrap().gb_fd_get_ll(
            d_h_out, h_h_out,
            self.cpp_orbits, self.cpp_tdi_config, b.cpp_fd,
            p.flatten().copy(),
            data_index, noise_index,
            num_bin, 9, self.T, self.t_start, self.t_ref,
            int(N_sparse) if N_sparse is not None else self.N_sparse,
            self.nchannels,
            self.backend.TDITypeDict[self.tdi_type],
            self.tukey_alpha, self.edge_frac,
            d_h_im,
        )
        self.d_h_out = d_h_out
        self.h_h_out = h_h_out
        self.d_h_im_out = _QUAD_SIGN_FD * d_h_im
        return -0.5 * (self.d_d + h_h_out - 2.0 * d_h_out)

    def get_swap_ll_fd(self, params_add, params_remove, fd_holder,
                       data_index=None, noise_index=None,
                       convert_to_ra_dec: Optional[bool] = None,
                       N_sparse: Optional[int] = None):
        b = self._fd_binding(fd_holder)
        pa = self._prep_params(params_add, convert_to_ra_dec)
        pr = self._prep_params(params_remove, convert_to_ra_dec)
        num_bin = pa.shape[0]
        assert pr.shape[0] == num_bin

        d_h_a = self.xp.zeros(num_bin); d_h_r = self.xp.zeros(num_bin)
        aa    = self.xp.zeros(num_bin); rr    = self.xp.zeros(num_bin)
        ar    = self.xp.zeros(num_bin)
        d_h_a_im = self.xp.zeros(num_bin)   # fused phase-max quadratures
        ar_im    = self.xp.zeros(num_bin)   #   (ADD-linear terms)

        if data_index is None:
            data_index = self.xp.zeros(num_bin, dtype=self.xp.int32)
        else:
            data_index = self.xp.asarray(data_index).astype(self.xp.int32)
        if noise_index is None:
            noise_index = self.xp.zeros(num_bin, dtype=self.xp.int32)
        else:
            noise_index = self.xp.asarray(noise_index).astype(self.xp.int32)

        if _GB_INDEX_ASSERTS:
            assert int(data_index.max()) < b.num_rows
            assert int(noise_index.max()) < b.num_noise

        self.backend.GBComputationGroupWrap().gb_fd_swap_ll(
            d_h_a, d_h_r, aa, rr, ar,
            self.cpp_orbits, self.cpp_tdi_config, b.cpp_fd,
            pa.flatten().copy(), pr.flatten().copy(),
            data_index, noise_index,
            num_bin, 9, self.T, self.t_start, self.t_ref,
            int(N_sparse) if N_sparse is not None else self.N_sparse,
            self.nchannels,
            self.backend.TDITypeDict[self.tdi_type],
            self.tukey_alpha, self.edge_frac,
            d_h_a_im, ar_im,
        )
        self.d_h_add_im_out = _QUAD_SIGN_FD_SWAP_ADD * d_h_a_im
        self.add_remove_im_out = _QUAD_SIGN_FD_SWAP_CROSS * ar_im
        like_add = -0.5 * (self.d_d + aa - 2.0 * d_h_a)
        like_rem = -0.5 * (self.d_d + rr - 2.0 * d_h_r)
        return like_add, like_rem, d_h_a, d_h_r, aa, rr, ar

    def setup_in_model(self, buffer_aca, params_ref_phys, data_index,
                       N_vals=None) -> None:
        """No-op per-source in-model setup hook (mirrors
        WDMComputationsBase.setup_in_model; the FD kernels score directly
        against the residual, no per-source preparation needed)."""
        return None

    def clear_in_model(self) -> None:
        """No-op in-model teardown hook."""
        return None

    def get_fstat_ll_fd(self, params, fd_holder, data_index=None,
                        noise_index=None,
                        convert_to_ra_dec: Optional[bool] = None):
        """F-statistic over the FD domain (Cornish & Crowder '05 basis).

        Single-template parity with :meth:`GBGPU.get_fstat_ll` and
        ``GBWDMComputations.get_fstat_ll_wdm``. The 4 basis filters share the
        binary's ``(f0, fdot, fddot, lam, beta)`` with fixed
        ``(A, iota, psi, phi0) = (2, pi/2, {0, pi/4, 0, pi/4},
        {0, pi, 3 pi/2, pi/2})``.

        Returns
        -------
        N_arr : ndarray, shape ``(num_bin, 4)``
            Per-binary ``<d | A_i>``.
        M_mat : ndarray, shape ``(num_bin, 10)``
            Per-binary upper-triangle ``<A_i | A_j>`` (i <= j), row-major
            ``[M00, M01, M02, M03, M11, M12, M13, M22, M23, M33]`` -- the
            SAME flat layout as ``get_fstat_ll_wdm``. Compute
            ``F = N^T M^{-1} N / 2`` on the Python side.

        Composed from the existing FD shared-memory kernels so the inner
        products match the FD likelihood exactly: ``N_i`` / ``M_ii`` come from
        :meth:`get_ll_fd` (``d_h`` / ``h_h``) and ``M_ij`` (i<j) from
        :meth:`get_swap_ll_fd`'s ``<A_i|A_j>`` (``ar``) cross term. A dedicated
        ``gb_fd_get_fstat_ll`` shared-memory kernel (mirroring
        ``wdm_het_get_fstat_ll_kernel``, building all 4 filters once per block)
        can replace this for throughput.
        """
        p = self._prep_params(params, convert_to_ra_dec)
        num_bin = p.shape[0]

        N_FILTERS = 4
        psi_arr  = [0.0, np.pi / 4.0, 0.0, np.pi / 4.0]
        phi0_arr = [0.0, np.pi, 3.0 * np.pi / 2.0, np.pi / 2.0]
        IDX_A, IDX_PHI0, IDX_IOTA, IDX_PSI = 0, 4, 5, 6

        def _filter(i):
            pf = p.copy()
            pf[:, IDX_A]    = 2.0
            pf[:, IDX_IOTA] = np.pi / 2.0
            pf[:, IDX_PSI]  = psi_arr[i]
            pf[:, IDX_PHI0] = phi0_arr[i]
            return pf

        def _m_idx(i, j):  # (i,j) -> flat upper-triangle index (i<=j)
            return i * N_FILTERS - (i * (i + 1)) // 2 + j

        filters = [_filter(i) for i in range(N_FILTERS)]
        N_arr = self.xp.zeros((num_bin, N_FILTERS), dtype=self.xp.float64)
        M_mat = self.xp.zeros(
            (num_bin, (N_FILTERS * (N_FILTERS + 1)) // 2), dtype=self.xp.float64)

        # Filters are already in the orbits frame; do not re-convert.
        for i in range(N_FILTERS):
            self.get_ll_fd(filters[i], fd_holder, data_index=data_index,
                           noise_index=noise_index, convert_to_ra_dec=False)
            N_arr[:, i] = self.d_h_out
            M_mat[:, _m_idx(i, i)] = self.h_h_out
            for j in range(i + 1, N_FILTERS):
                ar = self.get_swap_ll_fd(
                    filters[i], filters[j], fd_holder,
                    data_index=data_index, noise_index=noise_index,
                    convert_to_ra_dec=False)[6]
                M_mat[:, _m_idx(i, j)] = ar

        self.N_arr = N_arr
        self.M_mat = M_mat
        return N_arr, M_mat

    def _resolve_fill_target(self, templates, template_start_inds):
        """Normalise a ``fill_global`` target to its flat rows + addressing wrap.

        Accepts, per the sprint-wide call-time-holder rule:

        * :class:`AnalysisContainerArray` (incl. sub-band buffers) or a lone
          :class:`AnalysisContainer` -- templates accumulate into the
          holder's live ``linear_data_arr[0]`` rows; window starts default to
          the holder's own row starts (``min_freq_inds`` /
          ``start_freq_ind``).
        * a :class:`~lisatools.domains.DomainBase` child (e.g. ``FDSignal``)
          -- one template row written into ``domain.arr`` in place; the row
          start defaults to 0 on a full ``N``-bin grid, ``ind_min`` on an
          active-band slice.
        * a raw complex ndarray ``(num_templates, nchannels, n_bins)``
          (legacy layout) -- zeros starts by default.

        Returns ``(target_flat, cpp_fd, num_templates, n_bins,
        default_starts)``. ``cpp_fd`` only supplies row addressing to the
        fill kernel (its invC is never read), so non-holder targets get a
        throwaway wrap with a 1-element dummy invC.
        """
        from lisatools.analysiscontainer import AnalysisContainer
        from lisatools.domains import DomainBase

        if hasattr(templates, "linear_data_arr") or isinstance(
                templates, AnalysisContainer):
            b = self._fd_binding(templates)
            return b.data_flat, b.cpp_fd, b.num_rows, b.n_rfft, b.starts

        if isinstance(templates, DomainBase):
            arr = templates.arr
            if arr.ndim != 2 or arr.shape[0] != self.nchannels:
                raise ValueError(
                    f"DomainBase fill target must expose arr of shape "
                    f"({self.nchannels}, n_bins); got {arr.shape}.")
            n_bins = int(arr.shape[-1])
            start = 0 if n_bins == int(self.fd_settings.N) else self.ind_min
            default_starts = self.xp.full(1, start, dtype=self.xp.int32)
            arr_target, num_templates = arr, 1
        else:
            if templates.ndim != 3 or templates.shape[1] != self.nchannels:
                raise ValueError(
                    f"ndarray fill target must be (num_templates, "
                    f"{self.nchannels}, n_bins) complex; got "
                    f"{templates.shape}.")
            n_bins = int(templates.shape[-1])
            num_templates = int(templates.shape[0])
            default_starts = self.xp.zeros(num_templates, dtype=self.xp.int32)
            arr_target = templates

        if arr_target.dtype != self.xp.complex128:
            raise ValueError(
                f"fill target must be complex128; got {arr_target.dtype}.")
        if not arr_target.flags["C_CONTIGUOUS"]:
            raise ValueError(
                "fill target must be C-contiguous (the kernel accumulates "
                "into it in place).")

        # Explicit per-row starts signal windowed rows; the band bounds must
        # not clip the +/- N/2 template margins inside those windows. The
        # default (whole-grid rows) keeps the settings' active-band clip.
        starts = (default_starts if template_start_inds is None
                  else self.xp.ascontiguousarray(
                      template_start_inds, dtype=self.xp.int32))
        ind_min_eff, ind_max_eff = (
            (self.ind_min, self.ind_max) if template_start_inds is None
            else (0, self._WINDOW_IND_MAX)
        )
        target_flat = arr_target.reshape(-1)
        cpp_fd = self.backend.FDDomainWrap(
            target_flat, self.xp.zeros(1, dtype=self.xp.float64),
            n_bins, self.nchannels, num_templates, 1,
            ind_min_eff, ind_max_eff, self.df,
            starts,
        )
        # The wrap holds `starts` by pointer; keep the pair alive together.
        self._last_fill_wrap = (cpp_fd, starts, target_flat)
        return target_flat, cpp_fd, num_templates, n_bins, default_starts

    def fill_global(self, params, templates, data_index=None, factors=None,
                    convert_to_ra_dec: Optional[bool] = None,
                    N_sparse: Optional[int] = None,
                    template_start_inds=None):
        p = self._prep_params(params, convert_to_ra_dec)
        num_bin = p.shape[0]
        # Empty batch (e.g. a sub-band cell with no live sources): nothing
        # to accumulate.
        if num_bin == 0 or p.size == 0:
            return
        (target_flat, cpp_fd, num_templates, n_bins,
         default_starts) = self._resolve_fill_target(
            templates, template_start_inds)
        if data_index is None:
            data_index = self.xp.zeros(num_bin, dtype=self.xp.int32)
        else:
            data_index = self.xp.asarray(data_index).astype(self.xp.int32)
        if factors is None:
            factors = self.xp.ones(num_bin, dtype=float)
        else:
            factors = self.xp.asarray(factors, dtype=float)
        if _GB_INDEX_ASSERTS:
            assert int(data_index.max()) < num_templates

        if template_start_inds is None:
            template_start_inds = default_starts
        else:
            template_start_inds = self.xp.ascontiguousarray(
                template_start_inds, dtype=self.xp.int32)
            assert template_start_inds.shape == (num_templates,)

        self.backend.GBComputationGroupWrap().gb_fd_fill_global(
            target_flat,
            self.cpp_orbits, self.cpp_tdi_config, cpp_fd,
            p.flatten().copy(), data_index, factors,
            template_start_inds,
            num_bin, 9, self.T, self.t_start, self.t_ref,
            int(N_sparse) if N_sparse is not None else self.N_sparse,
            self.nchannels,
            self.tukey_alpha, self.edge_frac,
        )

    # ------------------------------------------------------------------
    # Chain-rule gradients of gb_fd_get_ll / gb_fd_swap_ll
    #
    # Mirrors GBWDMComputations.get_ll_grad_wdm / get_swap_ll_grad_wdm at the
    # API surface: same _DEFAULT_PARAM_EPS table, same _resolve_eps_and_scales
    # logic (param_eps + param_scales + param_eps_relative), same convention
    # that supplying param_scales returns the gradient in rescaled coordinates
    # eta = theta / Delta_theta.
    # ------------------------------------------------------------------
    _DEFAULT_PARAM_EPS = (
        1.0e-25,   # amp
        2.0e-14,   # f0    (Hz)
        1.0e-21,   # fdot  (Hz/s)
        1.0e-28,   # fddot (Hz/s^2)
        1.0e-6,    # phi0
        1.0e-6,    # iota
        1.0e-6,    # psi
        1.0e-6,    # lam (or RA after convert)
        1.0e-6,    # beta (or DEC after convert)
    )

    def _default_param_eps(self, nparams=9):
        eps = self.xp.asarray(self._DEFAULT_PARAM_EPS[:nparams],
                              dtype=self.xp.float64)
        if eps.shape[0] != nparams:
            extra = self.xp.full(nparams - eps.shape[0],
                                 eps[-1].item(), dtype=self.xp.float64)
            eps = self.xp.concatenate([eps, extra])
        return eps

    def _resolve_eps_and_scales(self, nparams,
                                param_eps, param_scales, param_eps_relative):
        if param_scales is not None:
            scales = self.xp.asarray(param_scales, dtype=self.xp.float64)
            assert scales.shape[0] == nparams, (
                f"param_scales length {scales.shape[0]} != nparams {nparams}"
            )
            if param_eps is None:
                eps_theta = scales * float(param_eps_relative)
            else:
                eps_theta = self.xp.asarray(param_eps, dtype=self.xp.float64)
                assert eps_theta.shape[0] == nparams
            return eps_theta, scales

        if param_eps is None:
            eps_theta = self._default_param_eps(nparams)
        else:
            eps_theta = self.xp.asarray(param_eps, dtype=self.xp.float64)
            assert eps_theta.shape[0] == nparams, (
                f"param_eps length {eps_theta.shape[0]} != nparams {nparams}"
            )
        return eps_theta, None

    def get_ll_grad_fd(self, params, fd_holder,
                       param_eps=None,
                       param_scales=None,
                       param_eps_relative=1.0e-6,
                       data_index=None, noise_index=None,
                       convert_to_ra_dec: Optional[bool] = None):
        """Chain-rule gradient of :meth:`get_ll_fd`.

        See :meth:`GBWDMComputations.get_ll_grad_wdm` for the meaning of
        ``param_scales`` / ``param_eps_relative``: with ``param_scales``
        provided the returned gradient is in rescaled coordinates
        ``eta_k = theta_k / Delta_theta_k`` and the kernel uses a uniform FD
        step ``eps_theta_k = param_eps_relative * Delta_theta_k``.

        Returns
        -------
        grad : (num_bin, nparams) xp.ndarray
            ``dL/dtheta_k`` (default) or ``dL/d(eta_k)`` when
            ``param_scales`` is supplied.
        """
        b = self._fd_binding(fd_holder)
        p = self._prep_params(params, convert_to_ra_dec)
        num_bin, nparams = p.shape

        if data_index is None:
            data_index = self.xp.zeros(num_bin, dtype=self.xp.int32)
        else:
            data_index = self.xp.asarray(data_index).astype(self.xp.int32)
        if noise_index is None:
            noise_index = self.xp.zeros(num_bin, dtype=self.xp.int32)
        else:
            noise_index = self.xp.asarray(noise_index).astype(self.xp.int32)

        eps_theta, scales = self._resolve_eps_and_scales(
            nparams, param_eps, param_scales, param_eps_relative,
        )

        grad_out = self.xp.zeros(num_bin * nparams, dtype=self.xp.float64)
        self.backend.GBComputationGroupWrap().gb_fd_get_ll_grad(
            grad_out,
            self.cpp_orbits, self.cpp_tdi_config, b.cpp_fd,
            p.flatten().copy(),
            data_index, noise_index,
            eps_theta,
            num_bin, nparams, self.T, self.t_start, self.t_ref,
            self.N_sparse, self.nchannels,
            self.backend.TDITypeDict[self.tdi_type],
        )
        grad = grad_out.reshape(num_bin, nparams)
        if scales is not None:
            grad = grad * scales[None, :]
        return grad

    def get_swap_ll_grad_fd(self, params_add, params_remove, fd_holder,
                            param_eps_add=None, param_eps_remove=None,
                            param_scales_add=None, param_scales_remove=None,
                            param_eps_relative=1.0e-6,
                            data_index=None, noise_index=None,
                            convert_to_ra_dec: Optional[bool] = None):
        """Chain-rule gradient of :meth:`get_swap_ll_fd`.

        Returns ``(grad_add, grad_remove)``, the per-binary derivatives of
        ``ll_diff = L(after swap) - L(before swap)`` with respect to
        ``theta_add`` and ``theta_remove`` respectively.  Rescaling
        semantics match :meth:`get_ll_grad_fd`.
        """
        b = self._fd_binding(fd_holder)
        pa = self._prep_params(params_add, convert_to_ra_dec)
        pr = self._prep_params(params_remove, convert_to_ra_dec)
        assert pa.shape == pr.shape, (
            f"params_add {pa.shape} != params_remove {pr.shape}"
        )
        num_bin, nparams = pa.shape

        if data_index is None:
            data_index = self.xp.zeros(num_bin, dtype=self.xp.int32)
        else:
            data_index = self.xp.asarray(data_index).astype(self.xp.int32)
        if noise_index is None:
            noise_index = self.xp.zeros(num_bin, dtype=self.xp.int32)
        else:
            noise_index = self.xp.asarray(noise_index).astype(self.xp.int32)

        eps_theta_add, scales_add = self._resolve_eps_and_scales(
            nparams, param_eps_add, param_scales_add, param_eps_relative,
        )
        eps_theta_remove, scales_remove = self._resolve_eps_and_scales(
            nparams, param_eps_remove, param_scales_remove, param_eps_relative,
        )

        grad_add_out    = self.xp.zeros(num_bin * nparams, dtype=self.xp.float64)
        grad_remove_out = self.xp.zeros(num_bin * nparams, dtype=self.xp.float64)
        self.backend.GBComputationGroupWrap().gb_fd_swap_ll_grad(
            grad_add_out, grad_remove_out,
            self.cpp_orbits, self.cpp_tdi_config, b.cpp_fd,
            pa.flatten().copy(), pr.flatten().copy(),
            data_index, noise_index,
            eps_theta_add, eps_theta_remove,
            num_bin, nparams, self.T, self.t_start, self.t_ref,
            self.N_sparse, self.nchannels,
            self.backend.TDITypeDict[self.tdi_type],
        )
        grad_add = grad_add_out.reshape(num_bin, nparams)
        grad_remove = grad_remove_out.reshape(num_bin, nparams)
        if scales_add is not None:
            grad_add = grad_add * scales_add[None, :]
        if scales_remove is not None:
            grad_remove = grad_remove * scales_remove[None, :]
        return grad_add, grad_remove

    def information_matrix(self, params, fd_holder, inds=None,
                           param_eps=None, param_scales=None,
                           param_eps_relative=1.0e-6,
                           easy_central_difference: bool = False,
                           noise_index=None,
                           convert_to_ra_dec: Optional[bool] = None,
                           batch_size: Optional[int] = None):
        """Per-source Fisher information matrix over the FD domain.

        Single-template parity with ``GBGPU.information_matrix`` (Python path):
        the waveform derivative ``dh/dtheta_i`` is built by central differences
        of the on-the-fly FD template (2nd order by default; 1st order with
        ``easy_central_difference=True``) and folded into the FD inner product

        ``Gamma_ij = 4 df Re sum_k dh_i^*(k) invC(k) dh_j(k)``

        which is exactly the normalization ``GBGPU.information_matrix`` uses
        (``4 df einsum(dh*, invC, dh)``; XYZ uses the full cross-channel invC,
        AE/AET the per-channel diagonal). Per-parameter steps reuse the same
        ``_resolve_eps_and_scales`` table as :meth:`get_ll_grad_fd`; passing
        ``param_scales`` returns the matrix in rescaled coordinates
        ``eta = theta / Delta_theta`` (``Gamma_eta = S Gamma_theta S``).

        Args:
            params: ``(num_bin, nparams)`` GB parameters (1D auto-promoted).
            fd_holder: the data holder (:class:`AnalysisContainerArray` or
                :class:`AnalysisContainer`) whose inverse-covariance rows
                (keyed by ``noise_index``) the derivative templates are
                contracted against. The ``dh`` workspace rows share the
                holder rows' window starts so both sides of the contraction
                are bin-aligned.
            inds: parameter indices to include (default all ``nparams``).
            param_eps / param_scales / param_eps_relative: finite-difference
                steps, mirroring :meth:`get_ll_grad_fd`.
            easy_central_difference: 1st-order derivative when ``True``.
            noise_index: per-binary index into the invC slabs.
            batch_size: source-batch size to bound the ``dh`` workspace
                (default: all sources at once).

        Returns:
            ``(num_bin, num_derivs, num_derivs)`` real Fisher matrices.
        """
        b = self._fd_binding(fd_holder)
        n_rfft = b.n_rfft
        invc_rows = b.invc_flat.reshape(
            (b.num_noise, self.nchannels, self.nchannels, n_rfft)
            if self.tdi_type == "XYZ"
            else (b.num_noise, self.nchannels, n_rfft)
        )

        p = self._prep_params(params, convert_to_ra_dec)
        num_bin, nparams = p.shape
        if inds is None:
            inds = list(range(nparams))
        else:
            inds = [int(i) for i in self.xp.asarray(inds).tolist()]
        num_derivs = len(inds)

        eps_theta, scales = self._resolve_eps_and_scales(
            nparams, param_eps, param_scales, param_eps_relative,
        )

        if noise_index is None:
            noise_index = self.xp.zeros(num_bin, dtype=self.xp.int32)
        else:
            noise_index = self.xp.asarray(noise_index).astype(self.xp.int32)
        if _GB_INDEX_ASSERTS:
            assert int(noise_index.max()) < b.num_noise

        if batch_size is None or batch_size <= 0:
            batch_size = num_bin

        info = self.xp.zeros((num_bin, num_derivs, num_derivs),
                             dtype=self.xp.float64)

        def _templates(pp, row_starts):
            nb = pp.shape[0]
            buf = self.xp.zeros((nb, self.nchannels, n_rfft),
                                dtype=complex)
            self.fill_global(
                pp, buf,
                data_index=self.xp.arange(nb, dtype=self.xp.int32),
                template_start_inds=row_starts,
            )
            return buf

        for s in range(0, num_bin, batch_size):
            e = min(s + batch_size, num_bin)
            pb = p[s:e]
            nb = e - s
            # dh rows must share the window offset of the invC rows they
            # are contracted with below.
            row_starts = self.xp.ascontiguousarray(
                b.starts[noise_index[s:e]], dtype=self.xp.int32)
            dh = self.xp.zeros(
                (nb, num_derivs, self.nchannels, n_rfft), dtype=complex)
            for di, ind in enumerate(inds):
                ei = float(eps_theta[ind])
                up1 = pb.copy(); up1[:, ind] += ei
                dn1 = pb.copy(); dn1[:, ind] -= ei
                h_up1 = _templates(up1, row_starts)
                h_dn1 = _templates(dn1, row_starts)
                if easy_central_difference:
                    dh[:, di] = (h_up1 - h_dn1) / (2.0 * ei)
                else:
                    up2 = pb.copy(); up2[:, ind] += 2.0 * ei
                    dn2 = pb.copy(); dn2[:, ind] -= 2.0 * ei
                    h_up2 = _templates(up2, row_starts)
                    h_dn2 = _templates(dn2, row_starts)
                    dh[:, di] = (
                        -h_up2 + h_dn2 + 8.0 * (h_up1 - h_dn1)
                    ) / (12.0 * ei)

            invc = invc_rows[noise_index[s:e]]
            if self.tdi_type == "XYZ":
                # dh: (b, i, c, k); invC: (b, c, d, k)
                info[s:e] = (4.0 * self.df * self.xp.einsum(
                    "bick,bcdk,bjdk->bij", dh.conj(), invc, dh)).real
            else:
                # invC diagonal in channels: (b, c, k)
                info[s:e] = (4.0 * self.df * self.xp.einsum(
                    "bick,bjck,bck->bij", dh.conj(), dh, invc)).real

        if scales is not None:
            scales_sel = self.xp.asarray([scales[i] for i in inds],
                                         dtype=self.xp.float64)
            info = info * scales_sel[None, :, None] * scales_sel[None, None, :]
        return info
