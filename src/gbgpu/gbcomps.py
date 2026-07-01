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
    _WRAP_ATTR = "GBComputationGroupWrap"
    _METHOD_PREFIX = "gb_wdm_het"
    _NPARAMS = 9
    _F0_PARAM_INDEX = 1   # GBTDIonTheFly: params[1] = f0


class _GBGradEpsMixin:
    """Shared finite-difference-step machinery for the GB gradient methods
    (``GBFDComputations`` FD path and ``STFTGBComputations`` STFT path).

    ``_DEFAULT_PARAM_EPS`` is the per-parameter central-FD step. Supplying
    ``param_scales`` switches the returned gradient to rescaled coordinates
    ``eta_k = theta_k / Delta_theta_k`` with a uniform step
    ``eps_theta_k = param_eps_relative * Delta_theta_k`` (matching
    ``GBWDMComputations.get_ll_grad_wdm``). Requires ``self.xp``.
    """

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


class GBFDComputations(_GBGradEpsMixin, FastLISAResponseParallelModule):
    """Frequency-domain heterodyne analog of :class:`GBWDMComputations`.

    Mirrors the WDM ``get_ll_wdm`` / ``get_swap_ll_wdm`` / ``fill_global``
    surface for the sparse FD heterodyne kernel.  All inner products are
    accumulated in C using the standard lisatools FD inner product
    ``(a|b) = 4 Re sum_{c1,c2} sum_k conj(a_c1[k]) b_c2[k] invC[c1,c2][k] * df``,
    so calling :func:`lisatools.diagnostic.inner_product` on the same data /
    invC / template gives an identical answer up to floating-point round-off.
    """

    def __init__(self, T, t_ref, t_start, N_sparse, df,
                 data_fd, invC,
                 orbits=None, tdi_config=None, force_backend=None,
                 d_d=0.0, tdi_type="XYZ", ind_min=None, ind_max=None,
                 tukey_alpha=0.0, edge_frac=0.0):
        super().__init__(force_backend=force_backend)
        if N_sparse < 1 or (N_sparse & (N_sparse - 1)) != 0:
            raise ValueError("N_sparse must be a power of two.")
        if abs(float(t_start) - float(t_ref)) > 1e-9:
            raise ValueError("GBFDComputations requires t_start == t_ref so "
                             "the heterodyne phase factor is unity.")
        if tdi_type not in {"XYZ", "AET", "AE"}:
            raise ValueError("tdi_type must be one of 'XYZ', 'AET', 'AE'.")

        self.T = float(T)
        self.t_ref = float(t_ref)
        self.t_start = float(t_start)
        self.N_sparse = int(N_sparse)
        self.df = float(df)
        self.d_d = float(d_d)
        self.tdi_type = tdi_type
        # Tukey taper (scipy.signal.windows.tukey alpha) applied to the slow
        # signal before the sparse-FD FFT, mirroring rfft(Tukey*td). MUST match
        # the window used to build ``data_fd`` -- with the same alpha the FD
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

        data_fd = self.xp.ascontiguousarray(data_fd)
        if data_fd.ndim != 3:
            raise ValueError(
                "data_fd must have shape (num_data, nchannels, n_rfft).")
        self.num_data, self.nchannels, self.n_rfft = data_fd.shape

        invC = self.xp.ascontiguousarray(invC, dtype=float)
        if tdi_type == "XYZ":
            if invC.ndim != 4 or invC.shape[1:] != (
                    self.nchannels, self.nchannels, self.n_rfft):
                raise ValueError(
                    f"For tdi_type=XYZ, invC must have shape "
                    f"(num_noise, {self.nchannels}, {self.nchannels}, "
                    f"{self.n_rfft}); got {invC.shape}.")
        else:
            if invC.ndim != 3 or invC.shape[1:] != (
                    self.nchannels, self.n_rfft):
                raise ValueError(
                    f"For tdi_type={tdi_type}, invC must have shape "
                    f"(num_noise, {self.nchannels}, {self.n_rfft}); "
                    f"got {invC.shape}.")
        self.num_noise = invC.shape[0]

        if ind_min is None: ind_min = 0
        if ind_max is None: ind_max = self.n_rfft - 1
        self.ind_min = int(ind_min)
        self.ind_max = int(ind_max)

        self._data_fd = data_fd
        self._invC    = invC

        self.cpp_fd = self.backend.FDDomainWrap(
            data_fd.reshape(-1),
            invC.reshape(-1),
            self.n_rfft, self.nchannels,
            self.num_data, self.num_noise,
            self.ind_min, self.ind_max, self.df,
        )

    @property
    def xp(self): return self.backend.xp

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
        if not self._orbits.configured:
            self._orbits.configure(linear_interp_setup=True)
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

    def get_ll_fd(self, params, data_index=None, noise_index=None,
                  convert_to_ra_dec: Optional[bool] = None):
        p = self._prep_params(params, convert_to_ra_dec)
        num_bin = p.shape[0]
        d_h_out = self.xp.zeros(num_bin)
        h_h_out = self.xp.zeros(num_bin)
        if data_index is None:
            data_index = self.xp.zeros(num_bin, dtype=self.xp.int32)
        else:
            data_index = self.xp.asarray(data_index).astype(self.xp.int32)
        if noise_index is None:
            noise_index = self.xp.zeros(num_bin, dtype=self.xp.int32)
        else:
            noise_index = self.xp.asarray(noise_index).astype(self.xp.int32)

        self.backend.GBComputationGroupWrap().gb_fd_get_ll(
            d_h_out, h_h_out,
            self.cpp_orbits, self.cpp_tdi_config, self.cpp_fd,
            p.flatten().copy(),
            data_index, noise_index,
            num_bin, 9, self.T, self.t_start, self.t_ref,
            self.N_sparse, self.nchannels,
            self.backend.TDITypeDict[self.tdi_type],
            self.tukey_alpha, self.edge_frac,
        )
        self.d_h_out = d_h_out
        self.h_h_out = h_h_out
        return -0.5 * (self.d_d + h_h_out - 2.0 * d_h_out)

    def get_swap_ll_fd(self, params_add, params_remove,
                       data_index=None, noise_index=None,
                       convert_to_ra_dec: Optional[bool] = None):
        pa = self._prep_params(params_add, convert_to_ra_dec)
        pr = self._prep_params(params_remove, convert_to_ra_dec)
        num_bin = pa.shape[0]
        assert pr.shape[0] == num_bin

        d_h_a = self.xp.zeros(num_bin); d_h_r = self.xp.zeros(num_bin)
        aa    = self.xp.zeros(num_bin); rr    = self.xp.zeros(num_bin)
        ar    = self.xp.zeros(num_bin)

        if data_index is None:
            data_index = self.xp.zeros(num_bin, dtype=self.xp.int32)
        else:
            data_index = self.xp.asarray(data_index).astype(self.xp.int32)
        if noise_index is None:
            noise_index = self.xp.zeros(num_bin, dtype=self.xp.int32)
        else:
            noise_index = self.xp.asarray(noise_index).astype(self.xp.int32)

        self.backend.GBComputationGroupWrap().gb_fd_swap_ll(
            d_h_a, d_h_r, aa, rr, ar,
            self.cpp_orbits, self.cpp_tdi_config, self.cpp_fd,
            pa.flatten().copy(), pr.flatten().copy(),
            data_index, noise_index,
            num_bin, 9, self.T, self.t_start, self.t_ref,
            self.N_sparse, self.nchannels,
            self.backend.TDITypeDict[self.tdi_type],
            self.tukey_alpha, self.edge_frac,
        )
        like_add = -0.5 * (self.d_d + aa - 2.0 * d_h_a)
        like_rem = -0.5 * (self.d_d + rr - 2.0 * d_h_r)
        return like_add, like_rem, d_h_a, d_h_r, aa, rr, ar

    def fill_global(self, params, templates, data_index=None, factors=None,
                    convert_to_ra_dec: Optional[bool] = None):
        p = self._prep_params(params, convert_to_ra_dec)
        num_bin = p.shape[0]
        if templates.ndim != 3 or templates.shape[1:] != (
                self.nchannels, self.n_rfft):
            raise ValueError(
                f"templates must be (num_templates, {self.nchannels}, "
                f"{self.n_rfft}) complex; got {templates.shape}.")
        num_templates = templates.shape[0]
        if data_index is None:
            data_index = self.xp.zeros(num_bin, dtype=self.xp.int32)
        else:
            data_index = self.xp.asarray(data_index).astype(self.xp.int32)
        if factors is None:
            factors = self.xp.ones(num_bin, dtype=float)
        else:
            factors = self.xp.asarray(factors, dtype=float)
        assert int(data_index.max()) < num_templates

        self.backend.GBComputationGroupWrap().gb_fd_fill_global(
            templates.reshape(-1),
            self.cpp_orbits, self.cpp_tdi_config, self.cpp_fd,
            p.flatten().copy(), data_index, factors,
            num_bin, 9, self.T, self.t_start, self.t_ref,
            self.N_sparse, self.nchannels,
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
    def get_ll_grad_fd(self, params,
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
            self.cpp_orbits, self.cpp_tdi_config, self.cpp_fd,
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

    def get_swap_ll_grad_fd(self, params_add, params_remove,
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
            self.cpp_orbits, self.cpp_tdi_config, self.cpp_fd,
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


class STFTGBComputations(_GBGradEpsMixin, FastLISAResponseParallelModule):
    """Source-side STFT/Fresnel-domain GB likelihood (Fresnel transform path).

    On-the-fly analog of :class:`GBFDComputations` / :class:`GBWDMComputations`
    for the STFT (Fresnel) domain. Routes through the ``gb_stft_{get_ll,
    fill_global}`` kernels on the backend (templated
    ``stft_*_impl<GBTDIonTheFly>`` in LAT's ``lat_stft_kernels.hh``), reusing the
    ``STFTFresnel`` / ``STFTDomain`` device primitives. The domain objects
    (``cpp_fresnel`` / ``cpp_domain``) and the ``<d|d>`` term are taken from a
    ``lisatools.domaincomputation.STFTComputationGroup``.

    Param order (matches ``GBTDIonTheFly``):
        ``(amp, f0, fdot, fddot, phi0, iota, psi, lam, beta)`` -- 9 params.

    ``freq_from_tdi_phase`` (default ``True``) derives the per-bin Fresnel chirp
    frequency / rate from the TDI phase, so the LISA orbital Doppler (whose rate
    typically exceeds the astrophysical fdot) is included; set ``False`` to
    recover the legacy astrophysical-``get_f``/``get_fdot`` behaviour.
    """

    _BACKEND_PREFIX = "gbgpu"

    def __init__(self, stft_comps, T, t_ref=0.0, orbits=None, tdi_config=None,
                 force_backend=None, n_side_bins=2, window_factor=1.0,
                 freq_from_tdi_phase=True):
        super().__init__(force_backend=force_backend)
        self.stft_comps = stft_comps
        self.T = float(T)
        self.t_ref = float(t_ref)
        self.n_side_bins = int(n_side_bins)
        self.window_factor = float(window_factor)
        self.freq_from_tdi_phase = bool(freq_from_tdi_phase)
        self.orbits = orbits
        self.tdi_config = tdi_config

    @property
    def xp(self): return self.backend.xp

    @property
    def num_params(self): return 9

    @property
    def stft_comps(self): return self._stft_comps
    @stft_comps.setter
    def stft_comps(self, sc): self._stft_comps = sc

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
        if not self._orbits.configured:
            self._orbits.configure(linear_interp_setup=True)
        self.cpp_orbits = self.backend.OrbitsWrap(*self._orbits.pycppdetector_args)

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
        self.cpp_tdi_config = self.backend.TDIConfigWrap(*self._tdi_config.pytdiconfig_args)

    @classmethod
    def supported_backends(cls):
        return [cls._BACKEND_PREFIX + "_" + _t for _t in cls.GPU_RECOMMENDED()]

    def _prep_params(self, params):
        return self.xp.asarray(self.xp.atleast_2d(params)).copy()

    def _resolve_indices(self, num_bin, data_index, noise_index):
        if data_index is None:
            data_index = self.xp.zeros(num_bin, dtype=self.xp.int32)
        else:
            data_index = self.xp.asarray(data_index).astype(self.xp.int32)
        if noise_index is None:
            noise_index = self.xp.zeros(num_bin, dtype=self.xp.int32)
        else:
            noise_index = self.xp.asarray(noise_index).astype(self.xp.int32)
        return data_index, noise_index

    def get_ll_stft(self, params, data_index=None, noise_index=None,
                    phase_maximize=False):
        """Log-likelihood ``-0.5*(<d|d> + <h|h> - 2<d|h>)`` per binary.

        Also stores the raw complex ``self.d_h_out`` / ``self.h_h_out`` (handy
        for cross-checks against the template-based STFTComputationGroup path).
        """
        if phase_maximize:
            raise NotImplementedError("Phase maximization not implemented for STFT GB yet.")
        p = self._prep_params(params)
        num_bin = p.shape[0]
        d_h_out = self.xp.zeros(num_bin, dtype=self.xp.complex128)
        h_h_out = self.xp.zeros(num_bin, dtype=self.xp.complex128)
        data_index, noise_index = self._resolve_indices(num_bin, data_index, noise_index)

        self.backend.GBComputationGroupWrap().gb_stft_get_ll(
            d_h_out, h_h_out,
            self.cpp_orbits, self.cpp_tdi_config,
            self.stft_comps.cpp_fresnel, self.stft_comps.cpp_domain,
            p.flatten().copy(), data_index, noise_index,
            num_bin, self.num_params, self.T, self.t_ref,
            self.n_side_bins, self.window_factor, self.freq_from_tdi_phase,
        )
        self.d_h_out = d_h_out
        self.h_h_out = h_h_out
        d_d_arr = self.stft_comps.d_d
        d_d = d_d_arr[data_index] if d_d_arr is not None else 0.0
        return (-0.5 * (d_d + h_h_out - 2.0 * d_h_out)).real

    def get_ll_stft_fft(self, params, data_index=None, noise_index=None,
                        n_sub=32, phase_maximize=False):
        """FFT-per-column log-likelihood, the throughput-oriented variant of
        :meth:`get_ll_stft`. Each STFT segment's template is a targeted DFT of the
        response sub-sampled at ``n_sub`` points (design 2026-07-01), instead of
        the analytic Fresnel per-pixel value. Converges to :meth:`get_ll_stft` as
        ``n_sub`` grows. Stores raw ``self.d_h_out_fft`` / ``self.h_h_out_fft``.

        Window: the FFT kernel applies the analysis window as a plain per-sample
        multiply from ``stft_comps.cpp_fresnel.window_alpha`` -- Tukey taper
        (taper_duration = alpha*dt/2, matching ``scipy.signal.windows.tukey`` and the
        Fresnel ``get_windowed_fourier_value``) when alpha>0, else a flat window scaled
        by ``window_factor``. This matches the windowed data STFT + the Fresnel path.

        Note (design 2026-07-01, get_ll-only phase): **``n_sub`` must satisfy
        ``n_sub >~ 2*n_side_bins + 1``** to resolve the requested band; below that the
        far bins alias. Default n_sub=32 pairs with the default n_side_bins=2 (5 bins);
        raise n_sub for wide side-bands.
        """
        if phase_maximize:
            raise NotImplementedError("Phase maximization not implemented for STFT GB FFT yet.")
        p = self._prep_params(params)
        num_bin = p.shape[0]
        d_h_out = self.xp.zeros(num_bin, dtype=self.xp.complex128)
        h_h_out = self.xp.zeros(num_bin, dtype=self.xp.complex128)
        data_index, noise_index = self._resolve_indices(num_bin, data_index, noise_index)

        self.backend.GBComputationGroupWrap().gb_stft_get_ll_fft(
            d_h_out, h_h_out,
            self.cpp_orbits, self.cpp_tdi_config,
            self.stft_comps.cpp_fresnel, self.stft_comps.cpp_domain,
            p.flatten().copy(), data_index, noise_index,
            num_bin, self.num_params, self.T, self.t_ref,
            self.n_side_bins, int(n_sub), self.window_factor, self.freq_from_tdi_phase,
        )
        self.d_h_out_fft = d_h_out
        self.h_h_out_fft = h_h_out
        d_d_arr = self.stft_comps.d_d
        d_d = d_d_arr[data_index] if d_d_arr is not None else 0.0
        return (-0.5 * (d_d + h_h_out - 2.0 * d_h_out)).real

    def get_swap_ll_stft(self, params_add, params_remove,
                         data_index=None, noise_index=None):
        """The 5 RJMCMC source-swap inner-product terms per binary.

        On-the-fly STFT analog of :meth:`GBFDComputations.get_swap_ll_fd`.
        Returns ``(like_add, like_remove, d_h_add, d_h_remove, add_add,
        remove_remove, add_remove)``. ``like_add`` / ``like_remove`` are the
        per-binary log-likelihoods of the add / remove template against the
        (shared) data; the five trailing complex arrays are the raw inner
        products an RJMCMC swap combines as
        ``2*Re[(d|h_add) - (d|h_remove)] - [(h_add|h_add) - (h_remove|h_remove)]
        - 2*Re[(h_add|h_remove)]``. They are also stored on ``self`` as
        ``d_h_add`` / ``d_h_remove`` / ``add_add`` / ``remove_remove`` /
        ``add_remove``. With ``params_add == params_remove`` every term collapses
        to the corresponding :meth:`get_ll_stft` ``(d|h)`` / ``(h|h)``.
        """
        pa = self._prep_params(params_add)
        pr = self._prep_params(params_remove)
        num_bin = pa.shape[0]
        assert pr.shape[0] == num_bin

        d_h_a = self.xp.zeros(num_bin, dtype=self.xp.complex128)
        d_h_r = self.xp.zeros(num_bin, dtype=self.xp.complex128)
        aa = self.xp.zeros(num_bin, dtype=self.xp.complex128)
        rr = self.xp.zeros(num_bin, dtype=self.xp.complex128)
        ar = self.xp.zeros(num_bin, dtype=self.xp.complex128)
        data_index, noise_index = self._resolve_indices(num_bin, data_index, noise_index)

        self.backend.GBComputationGroupWrap().gb_stft_swap_ll(
            d_h_a, d_h_r, aa, rr, ar,
            self.cpp_orbits, self.cpp_tdi_config,
            self.stft_comps.cpp_fresnel, self.stft_comps.cpp_domain,
            pa.flatten().copy(), pr.flatten().copy(),
            data_index, noise_index,
            num_bin, self.num_params, self.T, self.t_ref,
            self.n_side_bins, self.window_factor, self.freq_from_tdi_phase,
        )
        self.d_h_add, self.d_h_remove = d_h_a, d_h_r
        self.add_add, self.remove_remove, self.add_remove = aa, rr, ar
        d_d_arr = self.stft_comps.d_d
        d_d = d_d_arr[data_index] if d_d_arr is not None else 0.0
        like_add = (-0.5 * (d_d + aa - 2.0 * d_h_a)).real
        like_rem = (-0.5 * (d_d + rr - 2.0 * d_h_r)).real
        return like_add, like_rem, d_h_a, d_h_r, aa, rr, ar

    def get_fstat_ll_stft(self, params, data_index=None, noise_index=None):
        """F-statistic per binary over the STFT/Fresnel grid.

        Builds the 4 Cornish & Crowder '05 basis filters ``A_i`` at the binary's
        intrinsic ``(f0, fdot, fddot, lam, beta)`` with fixed
        ``(A, iota, psi, phi0) = (2, pi/2, {0, pi/4, 0, pi/4},
        {0, pi, 3*pi/2, pi/2})``.

        Returns
        -------
        N_arr : ndarray, shape ``(num_bin, 4)``
            Per-binary ``Re<d | A_i>``.
        M_mat : ndarray, shape ``(num_bin, 10)``
            Per-binary upper-triangle ``Re<A_i | A_j>`` (i <= j), flattened
            ``[M00, M01, M02, M03, M11, M12, M13, M22, M23, M33]``.

        Compute ``2F = N^T M^{-1} N`` from these via :meth:`fstat_2F`.

        STFT inner products are complex; the F-stat uses the real part (the same
        convention :meth:`get_ll_stft`'s logL uses). The kernel is a thin
        orchestration over the validated Stage-1/2 helpers (``stft_eval_block_ll``
        x4 + ``stft_eval_block_swap`` x6), so ``N`` / ``M`` are byte-identical to
        the matching :meth:`get_ll_stft` (d|A_i),(A_i|A_i) and
        :meth:`get_swap_ll_stft` (A_i|A_j) terms. The raw complex re+im outputs
        are stored on ``self`` as ``N_arr_cmplx`` / ``M_mat_cmplx`` (imag is a
        near-zero diagnostic).
        """
        p = self._prep_params(params)
        num_bin = p.shape[0]
        N_re = self.xp.zeros((num_bin, 4))
        N_im = self.xp.zeros((num_bin, 4))
        M_re = self.xp.zeros((num_bin, 10))
        M_im = self.xp.zeros((num_bin, 10))
        data_index, noise_index = self._resolve_indices(num_bin, data_index, noise_index)

        self.backend.GBComputationGroupWrap().gb_stft_get_fstat_ll(
            N_re.reshape(-1), N_im.reshape(-1),
            M_re.reshape(-1), M_im.reshape(-1),
            self.cpp_orbits, self.cpp_tdi_config,
            self.stft_comps.cpp_fresnel, self.stft_comps.cpp_domain,
            p.flatten().copy(), data_index, noise_index,
            num_bin, self.num_params, self.T, self.t_ref,
            self.n_side_bins, self.window_factor, self.freq_from_tdi_phase,
        )
        self.N_arr = N_re
        self.M_mat = M_re
        self.N_arr_cmplx = N_re + 1j * N_im
        self.M_mat_cmplx = M_re + 1j * M_im
        return N_re, M_re

    @staticmethod
    def fstat_2F(N_arr, M_mat):
        """Assemble ``2F = N^T M^{-1} N`` per binary from the F-stat outputs.

        Args:
            N_arr: ``(num_bin, 4)`` real ``<d|A_i>``.
            M_mat: ``(num_bin, 10)`` real upper-triangle ``<A_i|A_j>`` (i <= j,
                row-major) as returned by :meth:`get_fstat_ll_stft`. The flatten
                matches ``numpy.triu_indices(4)`` (= the kernel's ``m_idx``).

        Returns:
            ``(num_bin,)`` numpy array of ``2F`` values (host-side linear algebra;
            cupy inputs are pulled to host via ``.get()``).
        """
        N_arr = N_arr.get() if hasattr(N_arr, "get") else np.asarray(N_arr)
        M_mat = M_mat.get() if hasattr(M_mat, "get") else np.asarray(M_mat)
        N_arr = np.atleast_2d(N_arr)
        M_mat = np.atleast_2d(M_mat)
        iu = np.triu_indices(4)
        two_F = np.zeros(N_arr.shape[0])
        for b in range(N_arr.shape[0]):
            M = np.zeros((4, 4))
            M[iu] = M_mat[b]
            M = M + M.T - np.diag(np.diag(M))      # symmetrize from upper triangle
            two_F[b] = float(N_arr[b] @ np.linalg.solve(M, N_arr[b]))
        return two_F

    def fill_global_stft(self, params, templates, data_index=None, factors=None,
                         active_band=True):
        """Scatter ``0.5 * factor * fourier_value`` per (time, side-freq, channel)
        pixel into ``templates`` (shape ``(num_templates, nchannels, NT,
        NF_active)`` complex, the layout STFTComputationGroup consumes).
        Accumulates in place (caller pre-zeroes / accumulates)."""
        assert isinstance(templates, self.xp.ndarray)
        p = self._prep_params(params)
        num_bin = p.shape[0]
        if data_index is None:
            data_index = self.xp.zeros(num_bin, dtype=self.xp.int32)
        else:
            data_index = self.xp.asarray(data_index).astype(self.xp.int32)
        if factors is None:
            factors = self.xp.ones(num_bin, dtype=self.xp.float64)
        else:
            factors = self.xp.asarray(factors, dtype=self.xp.float64)

        self.backend.GBComputationGroupWrap().gb_stft_fill_global(
            templates.reshape(-1),
            self.cpp_orbits, self.cpp_tdi_config,
            self.stft_comps.cpp_fresnel, self.stft_comps.cpp_domain,
            p.flatten().copy(), data_index, factors,
            num_bin, self.num_params, self.T, self.t_ref,
            self.n_side_bins, self.window_factor, self.freq_from_tdi_phase,
            active_band,
        )

    # ------------------------------------------------------------------
    # Gradients (Stage 3): central finite difference over the 9 params,
    # reusing get_ll_stft / get_swap_ll_stft's exact forward evaluation.
    # Same _resolve_eps_and_scales convention as GBFDComputations (shared
    # via _GBGradEpsMixin): supplying param_scales returns the gradient in
    # rescaled coordinates eta = theta / Delta_theta.
    # ------------------------------------------------------------------
    def get_ll_grad_stft(self, params,
                         param_eps=None,
                         param_scales=None,
                         param_eps_relative=1.0e-6,
                         data_index=None, noise_index=None):
        """Per-parameter central finite-difference gradient of
        :meth:`get_ll_stft` (logL = Re(d|h) - 0.5*(h|h); the -0.5*(d|d) constant
        cancels in the difference).

        Returns ``grad`` of shape ``(num_bin, nparams)`` holding ``dL/dtheta_k``
        (default) or ``dL/d(eta_k)`` when ``param_scales`` is supplied (see
        :meth:`GBFDComputations.get_ll_grad_fd`). ``param_eps[k] <= 0`` freezes
        parameter ``k``. The kernel reuses ``get_ll_stft``'s exact forward
        evaluation, so it reproduces a host-side central difference of
        ``get_ll_stft`` to machine precision.
        """
        p = self._prep_params(params)
        num_bin = p.shape[0]
        nparams = self.num_params
        data_index, noise_index = self._resolve_indices(num_bin, data_index, noise_index)
        eps_theta, scales = self._resolve_eps_and_scales(
            nparams, param_eps, param_scales, param_eps_relative,
        )

        grad_out = self.xp.zeros(num_bin * nparams, dtype=self.xp.float64)
        self.backend.GBComputationGroupWrap().gb_stft_get_ll_grad(
            grad_out,
            self.cpp_orbits, self.cpp_tdi_config,
            self.stft_comps.cpp_fresnel, self.stft_comps.cpp_domain,
            p.flatten().copy(), data_index, noise_index,
            eps_theta,
            num_bin, nparams, self.T, self.t_ref,
            self.n_side_bins, self.window_factor, self.freq_from_tdi_phase,
        )
        grad = grad_out.reshape(num_bin, nparams)
        if scales is not None:
            grad = grad * scales[None, :]
        return grad

    def get_swap_ll_grad_stft(self, params_add, params_remove,
                              param_eps_add=None, param_eps_remove=None,
                              param_scales_add=None, param_scales_remove=None,
                              param_eps_relative=1.0e-6,
                              data_index=None, noise_index=None):
        """Central finite-difference gradients of the swap scalar
        ``S = Re(d|h_add) - Re(d|h_remove) - 0.5*(h_add|h_add)
        - 0.5*(h_remove|h_remove) + Re(h_add|h_remove)`` -- the STFT analog of
        :meth:`GBFDComputations.get_swap_ll_grad_fd` (``S = -0.5*||d - h_add +
        h_remove||^2`` up to the param-independent ``-0.5*(d|d)``).

        Returns ``(grad_add, grad_remove)``, each ``(num_bin, nparams)``:
        ``grad_add`` differentiates ``S`` w.r.t. ``theta_add`` (``theta_remove``
        held fixed); ``grad_remove`` w.r.t. ``theta_remove``. Rescaling
        semantics match :meth:`get_ll_grad_stft`; ``eps_k <= 0`` freezes the
        component.
        """
        pa = self._prep_params(params_add)
        pr = self._prep_params(params_remove)
        num_bin = pa.shape[0]
        assert pr.shape[0] == num_bin, (
            f"params_add num_bin {num_bin} != params_remove {pr.shape[0]}"
        )
        nparams = self.num_params
        data_index, noise_index = self._resolve_indices(num_bin, data_index, noise_index)
        eps_theta_add, scales_add = self._resolve_eps_and_scales(
            nparams, param_eps_add, param_scales_add, param_eps_relative,
        )
        eps_theta_remove, scales_remove = self._resolve_eps_and_scales(
            nparams, param_eps_remove, param_scales_remove, param_eps_relative,
        )

        grad_add_out    = self.xp.zeros(num_bin * nparams, dtype=self.xp.float64)
        grad_remove_out = self.xp.zeros(num_bin * nparams, dtype=self.xp.float64)
        self.backend.GBComputationGroupWrap().gb_stft_swap_ll_grad(
            grad_add_out, grad_remove_out,
            self.cpp_orbits, self.cpp_tdi_config,
            self.stft_comps.cpp_fresnel, self.stft_comps.cpp_domain,
            pa.flatten().copy(), pr.flatten().copy(),
            data_index, noise_index,
            eps_theta_add, eps_theta_remove,
            num_bin, nparams, self.T, self.t_ref,
            self.n_side_bins, self.window_factor, self.freq_from_tdi_phase,
        )
        grad_add = grad_add_out.reshape(num_bin, nparams)
        grad_remove = grad_remove_out.reshape(num_bin, nparams)
        if scales_add is not None:
            grad_add = grad_add * scales_add[None, :]
        if scales_remove is not None:
            grad_remove = grad_remove * scales_remove[None, :]
        return grad_add, grad_remove
