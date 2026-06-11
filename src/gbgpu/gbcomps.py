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


class GBFDComputations(FastLISAResponseParallelModule):
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
                 d_d=0.0, tdi_type="XYZ", ind_min=None, ind_max=None):
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
