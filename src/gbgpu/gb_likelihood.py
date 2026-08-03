"""Domain-agnostic band likelihood engines for the GB special moves.

Moved from ``lisatools.globalfit.moves.gb_likelihood`` (2026-07 GB-proposal
rework, per the fast-computation-wrap layering rule): this layer wraps the
GB fast-waveform computation objects (:class:`gbgpu.gbcomps.GBFDComputations`
/ :class:`GBWDMComputations`) and therefore lives with them in GBGPU. The
old lisatools path re-exports these names through a deprecation shim.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Protocol

import numpy as np

try:
    import cupy as cp
except (ImportError, ModuleNotFoundError):
    import numpy as cp

from lisatools.analysiscontainer import AnalysisContainerArray
from lisatools.domains import DomainSettingsBase, FDSettings, WDMSettings


@dataclass
class SwapLLResult:
    """Outputs of one batched swap-likelihood call.

    All arrays are 1-D, length ``num_proposals``. Out-of-bounds proposals get
    ``ll_diff = -1e300`` (callers don't need to mask separately).

    Attributes
    ----------
    ll_diff
        Posterior log-likelihood difference for the swap proposal, summed over
        bands. For FD this is the legacy
        ``gbgpu.swap_likelihood_difference`` output. For WDM it is
        ``-0.5 * ( <h_add|h_add> - 2<d|h_add> ) - ( -0.5 * ( <h_remove|h_remove> - 2<d|h_remove> ) )``
        which is the standard
        ``(d_h_add - d_h_remove) - 0.5*(h_add_h_add - h_remove_h_remove) - cross_term``
        algebra, computed from the five swap inner products.
    d_h_add, d_h_remove, hh_add, hh_remove, hh_cross
        The raw inner products. WDM populates all five; FD currently returns
        only those it computes natively, leaving the rest as ``None``.
    opt_snr_add
        Optimal SNR of the proposed (added) template: ``sqrt(<h_add|h_add>)``.
        Used by :meth:`Buffer.get_swap_ll` for the rejection-sampling clamp.
    phase_angle
        When ``phase_maximize=True``, the per-proposal phase rotation that
        the engine applied to maximise over phase. Otherwise ``None``. Callers
        subtract this from the proposed ``phi0`` parameter when an accept lands
        on a phase-maximised draw.
    kept
        Boolean mask (length ``num_proposals``) identifying the proposals that
        passed engine-internal bounds checks. Rejected proposals already have
        ``ll_diff = -1e300`` and ``opt_snr_add = 0``.
    """

    ll_diff: Any
    d_h_add: Any
    d_h_remove: Any
    hh_add: Any
    hh_remove: Any
    hh_cross: Any
    opt_snr_add: Any
    phase_angle: Optional[Any]
    kept: Any


# Physical GB parameter layout shared by every fast-computation wrap:
# (amp, f0[Hz], fdot, fddot, phi0, inc, psi, lam, beta).
PHYS_IDX_PHI0 = 4


class TwoQuadraturePhaseMaxMixin:
    """Analytic phase maximisation for engines whose kernels return REAL d_h.

    The GB waveform depends on the initial phase ``phi0`` as a pure carrier
    rotation, so for any linear-in-template functional ``g`` (``<d|h>``,
    ``<h_add|h_remove>``, ...) the dependence on a physical phase shift
    ``delta`` is sinusoidal: ``g(phi0 + delta) = R cos(delta ± theta)``.
    Two evaluations at ``phi0`` and ``phi0 + pi/2`` therefore determine the
    whole curve:

        D = g(phi0) + 1j * g(phi0 + pi/2)
        max_delta g = |D|      at    delta* = arg(D)
        g(delta*)   = Re[conj(D) * exp(1j * delta*)]   (any other functional)

    and -- the key property -- ``delta* = arg(D)`` holds for EITHER carrier
    sign convention (``cos(delta + theta)`` or ``cos(delta - theta)``), so
    this mixin needs no knowledge of the kernel's internal phase convention.

    ``phase_angle`` returned to callers is ``delta*``: the shift to ADD to
    the PHYSICAL ``phi0`` to sit at the maximum. The sampling basis stores
    ``-phi0`` (the transform container flips the sign, JaxGB convention), so
    the move-side contract "subtract ``phase_angle`` from sampling column 3"
    lands on exactly this maximum. Validated end-to-end by
    ``LISAanalysistools/scripts/gb_chunked_het/gb_phase_max_validate.py``.

    Cost: one extra kernel call per phase-maximised batch (the quadrature
    evaluation); ``h_h`` is phase-invariant so it is not recomputed.
    """

    def _get_ll_phase_max(
        self,
        buffer_aca,
        params_phys,
        *,
        data_index,
        noise_index,
        N_vals,
        return_inner_products,
        waveform_kwargs,
    ):
        xp = self.xp
        ll_0 = self.get_ll(
            buffer_aca, params_phys,
            data_index=data_index, noise_index=noise_index, N_vals=N_vals,
            phase_maximize=False, waveform_kwargs=waveform_kwargs,
        )
        d_h_0 = self.d_h_out.copy()
        h_h = self.h_h_out.copy()
        kept = self.kept_out.copy()

        params_q = params_phys.copy()
        params_q[:, PHYS_IDX_PHI0] = params_q[:, PHYS_IDX_PHI0] + np.pi / 2
        self.get_ll(
            buffer_aca, params_q,
            data_index=data_index, noise_index=noise_index, N_vals=N_vals,
            phase_maximize=False, waveform_kwargs=waveform_kwargs,
        )
        d_h_90 = self.d_h_out.copy()

        D = d_h_0 + 1j * d_h_90
        d_h_max = xp.abs(D)
        # The likelihood is linear in d_h (ll = -0.5*(d_d + h_h - 2 d_h)),
        # so the maximised ll is the base ll shifted by the d_h gain. The
        # -1e300 bounds sentinel stays put (gain is 0 on rejected rows).
        ll = xp.where(ll_0 > -1e290, ll_0 + (d_h_max - d_h_0), ll_0)

        self.non_marg_d_h = d_h_0
        self.d_h_out = d_h_max
        self.h_h_out = h_h
        self.phase_angle = xp.arctan2(D.imag, D.real)
        self.kept_out = kept
        if return_inner_products:
            return ll, self.d_h_out, self.h_h_out, self.phase_angle
        return ll

    def _get_swap_ll_phase_max(
        self,
        buffer_aca,
        params_remove_phys,
        params_add_phys,
        *,
        data_index,
        noise_index,
        N_vals,
        waveform_kwargs,
    ) -> "SwapLLResult":
        xp = self.xp
        res_0 = self.get_swap_ll(
            buffer_aca, params_remove_phys, params_add_phys,
            data_index=data_index, noise_index=noise_index, N_vals=N_vals,
            phase_maximize=False, waveform_kwargs=waveform_kwargs,
        )
        params_q = params_add_phys.copy()
        params_q[:, PHYS_IDX_PHI0] = params_q[:, PHYS_IDX_PHI0] + np.pi / 2
        res_90 = self.get_swap_ll(
            buffer_aca, params_remove_phys, params_q,
            data_index=data_index, noise_index=noise_index, N_vals=N_vals,
            phase_maximize=False, waveform_kwargs=waveform_kwargs,
        )

        # Only the ADD template's phase is maximised; the terms linear in
        # h_add are d_h_add and hh_cross, entering ll_diff as
        # G = d_h_add - hh_cross. Maximise G, then evaluate each linear
        # functional at the same delta*.
        D_g = (res_0.d_h_add - res_0.hh_cross) + 1j * (
            res_90.d_h_add - res_90.hh_cross
        )
        delta = xp.arctan2(D_g.imag, D_g.real)
        rot = xp.exp(1j * delta)

        def _at_max(g_0, g_90):
            return (xp.conj(g_0 + 1j * g_90) * rot).real

        d_h_add = _at_max(res_0.d_h_add, res_90.d_h_add)
        hh_cross = _at_max(res_0.hh_cross, res_90.hh_cross)
        gain = (d_h_add - hh_cross) - (res_0.d_h_add - res_0.hh_cross)
        ll_diff = xp.where(
            res_0.ll_diff > -1e290, res_0.ll_diff + gain, res_0.ll_diff
        )

        return SwapLLResult(
            ll_diff=ll_diff,
            d_h_add=d_h_add,
            d_h_remove=res_0.d_h_remove,
            hh_add=res_0.hh_add,
            hh_remove=res_0.hh_remove,
            hh_cross=hh_cross,
            opt_snr_add=res_0.opt_snr_add,
            phase_angle=delta,
            kept=res_0.kept,
        )


class BandLikelihoodEngine(Protocol):
    """Interface implemented by both the FD and WDM band engines.

    Implementations take an :class:`AnalysisContainerArray` (the per-band
    residual + inverse-PSD buffers) and the move's physical-parameter arrays;
    they return either inner products (:meth:`get_ll`) or a
    :class:`SwapLLResult` (:meth:`get_swap_ll`). The :meth:`fill_template`
    method writes (or removes) templates into the linear data buffer of the
    same ACA.

    All engines must expose ``basis_settings``, ``nchannels``, and
    ``tdi_channel_setup`` so the Buffer can sanity-check shapes at construction
    time.
    """

    basis_settings: DomainSettingsBase
    nchannels: int
    tdi_channel_setup: str

    def fill_template(
        self,
        buffer_aca: AnalysisContainerArray,
        params_phys,
        params_index,
        N_vals,
        *,
        factor: int,
        waveform_kwargs: dict,
    ) -> None:
        """Add (``factor=+1``) or remove (``factor=-1``) sources from
        ``buffer_aca.linear_data_arr[0]`` in-place.

        ``params_phys`` are already transformed to physical units.
        ``params_index`` maps each source to a band buffer index.
        """
        ...

    def get_ll(
        self,
        buffer_aca: AnalysisContainerArray,
        params_phys,
        *,
        data_index,
        noise_index,
        N_vals,
        phase_maximize: bool = False,
        return_inner_products: bool = False,
        waveform_kwargs: dict,
    ):
        """Compute the per-source log-likelihood against the per-band data
        in ``buffer_aca``.

        Returns a length-``len(params_phys)`` array of
        ``-0.5 * (d_d + h_h - 2 d_h)`` on the engine's xp module (``d_d``
        follows the underlying computation object's convention -- 0 unless
        configured, making the return the source-only piece). With
        ``return_inner_products=True`` returns the tuple
        ``(ll, d_h, h_h, phase_angle)`` instead.

        The raw inner products are also stashed on the engine as
        ``d_h_out`` / ``h_h_out``, and ``phase_angle`` holds the per-source
        maximising rotation when ``phase_maximize=True`` (``None``
        otherwise).
        """
        ...

    def get_swap_ll(
        self,
        buffer_aca: AnalysisContainerArray,
        params_remove_phys,
        params_add_phys,
        *,
        data_index,
        noise_index,
        N_vals,
        phase_maximize: bool,
        waveform_kwargs: dict,
    ) -> SwapLLResult:
        """Compute the five swap inner products and the resulting
        ``ll_diff`` for each row of the two parameter arrays.
        """
        ...

    def get_ll_grad(
        self,
        buffer_aca: AnalysisContainerArray,
        params_phys,
        *,
        data_index,
        noise_index,
        N_vals,
        param_eps=None,
        chunk: Optional[int] = None,
        waveform_kwargs: dict | None = None,
    ):
        """Per-source gradient of ``L = <d|h> - 0.5<h|h>``.

        Returns ``(num_proposals, nparams)``. Only the chunked-het
        backend implements this; the FD legacy path raises. The
        compute backend (C++ central-FD vs JAX autograd) is fixed at
        engine construction; per the sprint-wide rule there is no
        runtime ``backend=`` kwarg.
        """
        ...

    def hessian(
        self,
        buffer_aca: AnalysisContainerArray,
        params_phys,
        *,
        data_index,
        noise_index,
        N_vals,
        chunk: Optional[int] = None,
        psd_fix: bool = False,
        psd_floor_rel: float = 1e-30,
        waveform_kwargs: dict | None = None,
    ):
        """Per-source Hessian of ``L = <d|h> - 0.5<h|h>``.

        Returns ``(num_proposals, nparams, nparams)``. When
        ``psd_fix=True`` returns ``M = |-H|`` ready to feed to a NUTS
        sampler as a mass matrix. Only the chunked-het backend
        implements this; the FD legacy path raises. The compute
        backend (currently JAX autograd only) is fixed at engine
        construction.
        """
        ...


# ---------------------------------------------------------------------------
# Frequency-domain engine (wraps gbgpu.GBGPU)
# ---------------------------------------------------------------------------


class FDBandLikelihoodEngine(TwoQuadraturePhaseMaxMixin):
    """FD engine wrapping :class:`gbgpu.gbcomps.GBFDComputations`.

    2026-07 rework: the legacy ``SharedMemory*`` GBGPU kernel calls
    (``gb.get_ll`` / ``gb.generate_global_template`` /
    ``gb.swap_likelihood_difference``) are fully replaced by the new GB FD
    heterodyne kernel family (``gb_fd_get_ll`` / ``gb_fd_fill_global`` /
    ``gb_fd_swap_ll``), making this engine the exact FD mirror of
    :class:`WDMBandLikelihoodEngine` around ``GBWDMComputations``.

    The caller passes a CONFIG-ONLY :class:`GBFDComputations`
    (``GBFDComputations(fd_settings, t_ref, ...)``); every call forwards
    the buffer ACA as the ``fd_holder`` and the comp binds the holder's
    live linear data / inverse-covariance rows zero-copy at call time
    (cached on the holder; window starts come from
    ``buffer.min_freq_inds``, which the buffer updates IN PLACE on cell
    swaps, so the binding never goes stale).

    Per-source waveform lengths (``N_vals``) are handled by grouping calls
    over unique N (the kernels take one ``N_sparse`` per launch).
    """

    def __init__(
        self,
        gb_fd_comp,
        basis_settings: FDSettings,
        nchannels: int,
        tdi_channel_setup: str,
        df: float,
        start_freq_inds=None,
        data_length: Optional[int] = None,
        opt_snr_rej_samp_limit: float = 5.0,
        snr_rej_detected: bool = False,
    ):
        self.gb_fd_comp = gb_fd_comp          # config-only comp
        self.basis_settings = basis_settings
        self.nchannels = nchannels
        self.tdi_channel_setup = tdi_channel_setup
        self.df = df
        # Fallback window starts for holders exposing neither
        # ``min_freq_inds`` nor a resolvable ``start_freq_ind``.
        self.start_freq_inds = start_freq_inds
        self.data_length = data_length
        self.opt_snr_rej_samp_limit = opt_snr_rej_samp_limit
        # Optional companion test on the DETECTED snr d_h/sqrt(h_h) against
        # the same limit (default OFF): the optimal-SNR boundary is a
        # property of the template, the detected one fluctuates with the
        # noise realization.
        self.snr_rej_detected = bool(snr_rej_detected)

    @property
    def xp(self):
        return self.gb_fd_comp.xp

    # ---------- per-holder window geometry ---------------------------------

    def _min_freq_inds(self, buffer_aca: AnalysisContainerArray):
        """Per-slot window start indices, read off the holder at call time.

        Mirrors the comp's own resolution order (``min_freq_inds`` ->
        ``start_freq_ind``) so the engine-level bounds mask agrees with the
        rows the kernel actually analyses; the ctor's ``start_freq_inds``
        sits between the two as an explicit override.
        """
        min_freq_inds = getattr(buffer_aca, "min_freq_inds", None)
        if min_freq_inds is None:
            min_freq_inds = self.start_freq_inds
        if min_freq_inds is None:
            sfi = getattr(buffer_aca, "start_freq_ind", None)
            try:
                min_freq_inds = self.xp.asarray(sfi).astype(self.xp.int32)
                if min_freq_inds.ndim == 0:
                    min_freq_inds = self.xp.full(
                        int(buffer_aca.acs_total_entries),
                        int(min_freq_inds), dtype=self.xp.int32)
            except (TypeError, ValueError):
                min_freq_inds = None
        if min_freq_inds is None:
            raise ValueError(
                "FDBandLikelihoodEngine needs per-slot window start indices: "
                "the holder must expose ``min_freq_inds`` or "
                "``start_freq_ind``, or the engine must be constructed with "
                "``start_freq_inds``."
            )
        return min_freq_inds

    @staticmethod
    def _data_length_bins(buffer_aca: AnalysisContainerArray) -> int:
        """Per-slot frequency-bin count, read off the buffer at call time."""
        return buffer_aca.data_shaped[0].shape[-1]

    def _unique_N_groups(self, N_vals, num: int):
        xp = self.xp
        if N_vals is None:
            return [(xp.arange(num), self.gb_fd_comp.N_sparse)]
        N_arr = xp.asarray(N_vals)
        out = []
        for n in xp.unique(N_arr):
            rows = xp.arange(num)[N_arr == n]
            out.append((rows, int(n)))
        return out

    # ---------- fill_template ---------------------------------------------

    def fill_template(
        self,
        buffer_aca: AnalysisContainerArray,
        params_phys,
        params_index,
        N_vals,
        *,
        factor: int,
        waveform_kwargs: dict,
    ) -> None:
        assert factor in (-1, +1)
        comp = self.gb_fd_comp
        xp = self.xp

        params_index = xp.asarray(params_index).astype(xp.int32)
        factors = float(factor) * xp.ones(params_phys.shape[0], dtype=float)
        for rows, n in self._unique_N_groups(N_vals, params_phys.shape[0]):
            comp.fill_global(
                params_phys[rows], buffer_aca,
                data_index=params_index[rows],
                factors=factors[rows],
                N_sparse=n,
            )

    # ---------- get_ll ------------------------------------------------------

    def get_ll(
        self,
        buffer_aca: AnalysisContainerArray,
        params_phys,
        *,
        data_index,
        noise_index,
        N_vals,
        phase_maximize: bool = False,
        return_inner_products: bool = False,
        waveform_kwargs: dict,
    ):
        if phase_maximize:
            # Two-quadrature analytic maximisation (Python-level; the gb_fd
            # kernels return real d_h). See TwoQuadraturePhaseMaxMixin.
            return self._get_ll_phase_max(
                buffer_aca, params_phys,
                data_index=data_index, noise_index=noise_index, N_vals=N_vals,
                return_inner_products=return_inner_products,
                waveform_kwargs=waveform_kwargs,
            )
        comp = self.gb_fd_comp
        xp = self.xp
        num = params_phys.shape[0]

        min_freq_inds = self._min_freq_inds(buffer_aca)
        data_length_bins = self._data_length_bins(buffer_aca)
        data_index = xp.asarray(data_index).astype(xp.int32)
        noise_index = xp.asarray(noise_index).astype(xp.int32)
        N_arr = xp.asarray(N_vals) if N_vals is not None else xp.full(
            num, self.gb_fd_comp.N_sparse
        )

        # Reject sources whose ±N/2 window is not fully inside the per-slot
        # buffer (partial windows would silently truncate inner products).
        f_bin = (params_phys[:, 1] / self.df).astype(int) - min_freq_inds[data_index]
        keep = (f_bin - N_arr / 2 >= 0) & (f_bin + N_arr / 2 <= data_length_bins)
        self.kept_out = keep

        ll = xp.full(num, -1e300)
        self.d_h_out = xp.zeros(num)
        self.h_h_out = xp.zeros(num)
        self.phase_angle = None

        rows_keep = xp.arange(num)[keep]
        if len(rows_keep):
            for rows, n in self._unique_N_groups(N_arr[keep], len(rows_keep)):
                r = rows_keep[rows]
                ll_r = comp.get_ll_fd(
                    params_phys[r], buffer_aca,
                    data_index=data_index[r],
                    noise_index=noise_index[r],
                    N_sparse=n,
                )
                ll[r] = ll_r
                self.d_h_out[r] = comp.d_h_out
                self.h_h_out[r] = comp.h_h_out
        if return_inner_products:
            return ll, self.d_h_out, self.h_h_out, self.phase_angle
        return ll

    # ---------- in-model likelihood setup hooks ------------------------------

    def setup_in_model(self, buffer_aca, params_phys, data_index,
                       N_vals=None) -> None:
        """Per-source in-model setup hook (no-op for the FD comps; see the
        WDM engine's version for the sig-het behavior)."""
        return self.gb_fd_comp.setup_in_model(
            buffer_aca, params_phys, data_index, N_vals=N_vals)

    def clear_in_model(self) -> None:
        return self.gb_fd_comp.clear_in_model()

    # ---------- get_swap_ll --------------------------------------------------

    def get_swap_ll(
        self,
        buffer_aca: AnalysisContainerArray,
        params_remove_phys,
        params_add_phys,
        *,
        data_index,
        noise_index,
        N_vals,
        phase_maximize: bool,
        waveform_kwargs: dict,
    ) -> SwapLLResult:
        if phase_maximize:
            return self._get_swap_ll_phase_max(
                buffer_aca, params_remove_phys, params_add_phys,
                data_index=data_index, noise_index=noise_index, N_vals=N_vals,
                waveform_kwargs=waveform_kwargs,
            )
        comp = self.gb_fd_comp
        xp = self.xp
        num = params_add_phys.shape[0]

        min_freq_inds = self._min_freq_inds(buffer_aca)
        data_length_bins = self._data_length_bins(buffer_aca)
        data_index = xp.asarray(data_index).astype(xp.int32)
        noise_index = xp.asarray(noise_index).astype(xp.int32)
        N_arr = xp.asarray(N_vals) if N_vals is not None else xp.full(
            num, self.gb_fd_comp.N_sparse
        )

        f_bin_a = (params_add_phys[:, 1] / self.df).astype(int) - min_freq_inds[data_index]
        f_bin_r = (params_remove_phys[:, 1] / self.df).astype(int) - min_freq_inds[data_index]
        keep = (
            (f_bin_a - N_arr / 2 >= 0) & (f_bin_a + N_arr / 2 <= data_length_bins)
            & (f_bin_r - N_arr / 2 >= 0) & (f_bin_r + N_arr / 2 <= data_length_bins)
        )

        ll_diff = xp.full(num, -1e300)
        d_h_add = xp.zeros(num)
        d_h_remove = xp.zeros(num)
        hh_add = xp.zeros(num)
        hh_remove = xp.zeros(num)
        hh_cross = xp.zeros(num)
        opt_snr = xp.zeros(num)

        rows_keep = xp.arange(num)[keep]
        if len(rows_keep):
            for rows, n in self._unique_N_groups(N_arr[keep], len(rows_keep)):
                r = rows_keep[rows]
                (_, _, d_h_a, d_h_r, aa, rr, ar) = comp.get_swap_ll_fd(
                    params_add_phys[r], params_remove_phys[r], buffer_aca,
                    data_index=data_index[r], noise_index=noise_index[r],
                    N_sparse=n,
                )
                # standard swap algebra (see WDM engine for the derivation)
                ll_diff[r] = (d_h_a - d_h_r) - 0.5 * (aa - rr) - (ar - rr)
                d_h_add[r] = d_h_a
                d_h_remove[r] = d_h_r
                hh_add[r] = aa
                hh_remove[r] = rr
                hh_cross[r] = ar
                opt_snr[r] = xp.sqrt(xp.maximum(aa, 0.0))

        return SwapLLResult(
            ll_diff=ll_diff,
            d_h_add=d_h_add,
            d_h_remove=d_h_remove,
            hh_add=hh_add,
            hh_remove=hh_remove,
            hh_cross=hh_cross,
            opt_snr_add=opt_snr,
            phase_angle=None,
            kept=keep,
        )

    # ---------- get_ll_grad / hessian ---------------------------------------

    def get_ll_grad(
        self,
        buffer_aca: AnalysisContainerArray,
        params_phys,
        *,
        data_index,
        noise_index,
        N_vals,
        param_eps=None,
        chunk: Optional[int] = None,
        waveform_kwargs: dict | None = None,
    ):
        """Per-source gradient through ``GBFDComputations.get_ll_grad_fd``."""
        del N_vals, waveform_kwargs, chunk
        return self.gb_fd_comp.get_ll_grad_fd(
            params_phys, buffer_aca,
            data_index=data_index, noise_index=noise_index,
            param_eps=param_eps,
        )

    def hessian(self, *_args, **_kwargs):
        raise NotImplementedError(
            "FDBandLikelihoodEngine.hessian is not implemented yet; use "
            "GBFDComputations.information_matrix for proposal shaping."
        )


# ---------------------------------------------------------------------------
# WDM-domain engine (wraps gbgpu.gbcomps.GBWDMComputations)
# ---------------------------------------------------------------------------


class WDMBandLikelihoodEngine(TwoQuadraturePhaseMaxMixin):
    """WDM engine wrapping a
    :class:`gbgpu.gbcomps.GBWDMComputations` instance.

    Three calls all route through the same ``gb_comps`` object: fill, get_ll,
    and get_swap_ll. The per-band ACA is forwarded directly -- the GB WDM
    Python API already speaks AnalysisContainerArray.

    The bounds-keep mask is WDM-flavoured: each source's
    ``layer_m = int(f0 / layer_df)`` must land within the WDM
    ``[ind_min_f, ind_max_f]`` active band.
    """

    def __init__(
        self,
        gb_comps,
        basis_settings: WDMSettings,
        nchannels: int,
        tdi_channel_setup: str,
        opt_snr_rej_samp_limit: float = 5.0,
        snr_rej_detected: bool = False,
    ):
        self.gb_comps = gb_comps
        self.basis_settings = basis_settings
        self.nchannels = nchannels
        self.tdi_channel_setup = tdi_channel_setup
        self.opt_snr_rej_samp_limit = opt_snr_rej_samp_limit
        # Optional companion test on the DETECTED snr d_h/sqrt(h_h) against
        # the same limit (default OFF): the optimal-SNR boundary is a
        # property of the template, the detected one fluctuates with the
        # noise realization.
        self.snr_rej_detected = bool(snr_rej_detected)

    @property
    def xp(self):
        return self.gb_comps.xp

    # ---------- fill_template ------------------------------------------------

    def fill_template(
        self,
        buffer_aca: AnalysisContainerArray,
        params_phys,
        params_index,
        N_vals,
        *,
        factor: int,
        waveform_kwargs: dict,
        band_slab_Nf=None,
        slab_min_f=None,
    ) -> None:
        assert factor in (-1, +1)
        # GBWDMComputations.fill_global_wdm needs the params in the layout it
        # expects (num_bin, 9). data_index controls which per-band buffer each
        # source writes to. The factors argument was added to the kernel for
        # this purpose so add/remove is one C call.
        xp = self.gb_comps.xp
        num_bin = params_phys.shape[0]
        factors_arr = xp.full(num_bin, float(factor), dtype=xp.float64)

        # Post-Phase-3L.7p, ``fill_global_wdm`` signature is
        # ``(params, templates, ...)`` -- the flat WDM-template buffer is the
        # only positional after params, no third ``wdm_holder`` slot.
        # Task-b: band_slab_Nf / slab_min_f (sourced from the SubBandBuffer,
        # which owns both the residual + template-twin slabs) size the narrow
        # per-band write; None/None is the full-active-band (off) default.
        self.gb_comps.fill_global_wdm(
            params_phys,
            buffer_aca.linear_data_arr[0],
            data_index=params_index,
            factors=factors_arr,
            band_slab_Nf=band_slab_Nf,
            slab_min_f=slab_min_f,
        )

    # ---------- get_ll -------------------------------------------------------

    def get_ll(
        self,
        buffer_aca: AnalysisContainerArray,
        params_phys,
        *,
        data_index,
        noise_index,
        N_vals,
        phase_maximize: bool = False,
        return_inner_products: bool = False,
        waveform_kwargs: dict,
    ):
        if phase_maximize:
            # Two-quadrature analytic maximisation (Python-level; the WDM
            # chunked-het kernel returns real d_h). See
            # TwoQuadraturePhaseMaxMixin.
            return self._get_ll_phase_max(
                buffer_aca, params_phys,
                data_index=data_index, noise_index=noise_index, N_vals=N_vals,
                return_inner_products=return_inner_products,
                waveform_kwargs=waveform_kwargs,
            )
        ll = self.gb_comps.get_ll_wdm(
            params_phys,
            buffer_aca,
            data_index=data_index,
            noise_index=noise_index,
        )
        ll = self.xp.asarray(ll)
        # Inner products stashed for callers that need them (mirrors the
        # gb_comps d_h_out / h_h_out convention). The WDM kernel internally
        # skips out-of-band layers, so kept_out is all-True here.
        self.d_h_out = self.gb_comps.d_h_out.copy()
        self.h_h_out = self.gb_comps.h_h_out.copy()
        self.phase_angle = None
        self.kept_out = self.gb_comps.xp.ones(params_phys.shape[0], dtype=bool)
        if return_inner_products:
            return ll, self.d_h_out, self.h_h_out, self.phase_angle
        return ll

    # ---------- in-model likelihood setup hooks ------------------------------

    def setup_in_model(self, buffer_aca, params_phys, data_index,
                       N_vals=None) -> None:
        """Per-source in-model setup: forwards to the computation object.

        Chunked-het comps inherit the no-op from WDMComputationsBase;
        sig-het comps (GBSignalHetComputations.for_band_engine) build the
        heterodyne reference against the source-free cell residuals here
        and hold it constant until :meth:`clear_in_model`."""
        return self.gb_comps.setup_in_model(
            buffer_aca, params_phys, data_index, N_vals=N_vals)

    def clear_in_model(self) -> None:
        return self.gb_comps.clear_in_model()

    # ---------- get_swap_ll --------------------------------------------------

    def get_swap_ll(
        self,
        buffer_aca: AnalysisContainerArray,
        params_remove_phys,
        params_add_phys,
        *,
        data_index,
        noise_index,
        N_vals,
        phase_maximize: bool,
        waveform_kwargs: dict,
    ) -> SwapLLResult:
        xp = self.gb_comps.xp

        if phase_maximize:
            return self._get_swap_ll_phase_max(
                buffer_aca, params_remove_phys, params_add_phys,
                data_index=data_index, noise_index=noise_index, N_vals=N_vals,
                waveform_kwargs=waveform_kwargs,
            )

        # WDM bounds-keep: each source's central frequency layer must sit in
        # the active band. The kernel internally skips OOB layers, but the
        # move needs to set ll_diff = -1e300 on those proposals.
        layer_df = self.basis_settings.layer_df
        m_min = self.basis_settings.ind_min_f
        m_max = self.basis_settings.ind_max_f
        layer_add = (params_add_phys[:, 1] / layer_df).astype(int)
        layer_remove = (params_remove_phys[:, 1] / layer_df).astype(int)
        keep = (
            (layer_add >= m_min)
            & (layer_add <= m_max)
            & (layer_remove >= m_min)
            & (layer_remove <= m_max)
        )

        num_prop = keep.shape[0]
        ll_diff = xp.full(num_prop, -1e300, dtype=xp.float64)
        opt_snr = xp.zeros(num_prop, dtype=xp.float64)
        d_h_add = xp.zeros(num_prop, dtype=xp.float64)
        d_h_remove = xp.zeros(num_prop, dtype=xp.float64)
        hh_add = xp.zeros(num_prop, dtype=xp.float64)
        hh_remove = xp.zeros(num_prop, dtype=xp.float64)
        hh_cross = xp.zeros(num_prop, dtype=xp.float64)

        if bool(keep.any() if hasattr(keep, "any") else keep.any()):
            keep_idx = keep
            (
                like_add,
                like_remove,
                d_h_a,
                d_h_r,
                aa,
                rr,
                ar,
            ) = self.gb_comps.get_swap_ll_wdm(
                params_add_phys[keep_idx],
                params_remove_phys[keep_idx],
                buffer_aca,
                data_index=data_index[keep_idx],
                noise_index=noise_index[keep_idx],
            )

            # ll_diff = like_add - like_remove + cross-term correction.
            # The standard swap-likelihood difference is
            #   Δ log L = (<d|h_add> - <d|h_remove>)
            #            - 0.5 * (<h_add|h_add> - <h_remove|h_remove>)
            #            - (<h_add|h_remove> - <h_remove|h_remove>)
            # In the presence of an existing remove-template that is already
            # part of the residual, the cross term is what survives. For the
            # simplest case (independent proposal) it reduces to:
            #   Δ log L = (d_h_add - d_h_remove) - 0.5*(hh_add - hh_remove)
            #            - (hh_cross - hh_remove)
            # which is the form gbgpu's swap_likelihood_difference returns.
            ll_diff_keep = (
                (d_h_a - d_h_r)
                - 0.5 * (aa - rr)
                - (ar - rr)
            )
            ll_diff[keep_idx] = ll_diff_keep
            d_h_add[keep_idx] = d_h_a
            d_h_remove[keep_idx] = d_h_r
            hh_add[keep_idx] = aa
            hh_remove[keep_idx] = rr
            hh_cross[keep_idx] = ar
            opt_snr[keep_idx] = xp.sqrt(xp.maximum(aa, 0.0))

        return SwapLLResult(
            ll_diff=ll_diff,
            d_h_add=d_h_add,
            d_h_remove=d_h_remove,
            hh_add=hh_add,
            hh_remove=hh_remove,
            hh_cross=hh_cross,
            opt_snr_add=opt_snr,
            phase_angle=None,
            kept=keep,
        )

    # ---------- get_ll_grad / hessian (chunked-het backends only) ----------
    #
    # These two methods are present on ``gb_comps`` only when the
    # underlying generator is :class:`GBWDMComputations` (the chunked-het
    # backend). The legacy lookup-table ``GBWDMComputations`` does not
    # expose ``hessian_wdm`` (its FD ``get_ll_grad_wdm`` is implemented
    # but the in-the-kernel info-matrix path it served has been retired).
    # We probe at call time so a global fit running on the legacy
    # generator fails loudly rather than silently masking a config bug.

    def _require_chunked_het(self, method_name: str):
        if not hasattr(self.gb_comps, method_name):
            raise NotImplementedError(
                f"WDMBandLikelihoodEngine.{method_name.replace('_wdm','')}: "
                f"underlying gb_comps ({type(self.gb_comps).__name__}) "
                f"does not expose {method_name!r}. The gradient / Hessian "
                "paths require a GBWDMComputations backend; rebuild the "
                "Buffer with the chunked-het generator."
            )

    def get_ll_grad(
        self,
        buffer_aca: AnalysisContainerArray,
        params_phys,
        *,
        data_index,
        noise_index,
        N_vals,
        param_eps=None,
        chunk: Optional[int] = None,
        waveform_kwargs: dict | None = None,
    ):
        """Per-source gradient of ``L = <d|h> - 0.5 <h|h>`` w.r.t. params.

        Returns ``(num_proposals, nparams)`` -- one row per source.
        The compute backend (C++ central-FD vs JAX autograd) is fixed
        at the ``GBWDMComputations``'s construction-time
        ``force_backend``; per the sprint-wide rule no ``backend=``
        runtime kwarg is taken.
        """
        self._require_chunked_het("get_ll_grad_wdm")
        # Post-Phase-3L.7p, ``get_ll_grad_wdm`` no longer accepts
        # ``param_eps`` / ``chunk``; the JAX-autograd backend handles the
        # full-precision derivative internally and the C++ kernel does
        # not (yet) split the gradient over sub-chunks. Engine swallows
        # the kwargs for caller backward-compat.
        del N_vals, waveform_kwargs, param_eps, chunk
        return self.gb_comps.get_ll_grad_wdm(
            params_phys, buffer_aca,
            data_index=data_index, noise_index=noise_index,
        )

    def hessian(
        self,
        buffer_aca: AnalysisContainerArray,
        params_phys,
        *,
        data_index,
        noise_index,
        N_vals,
        chunk: Optional[int] = None,
        psd_fix: bool = False,
        psd_floor_rel: float = 1e-30,
        waveform_kwargs: dict | None = None,
    ):
        """Per-source Hessian of ``L = <d|h> - 0.5 <h|h>``.

        Returns ``(num_proposals, nparams, nparams)``. When
        ``psd_fix=True`` returns ``M = |−H|`` (eigendecompose +
        ``|lambda|``), ready to feed to ``NUTSSampler(metric=M)``. The
        compute backend (JAX autograd only at present) is fixed at
        ``GBWDMComputations`` construction; calling this on a non-JAX
        chunked-het instance raises ``NotImplementedError`` inside
        ``hessian_wdm``.
        """
        self._require_chunked_het("hessian_wdm")
        del N_vals, waveform_kwargs
        return self.gb_comps.hessian_wdm(
            params_phys, buffer_aca,
            data_index=data_index, noise_index=noise_index,
            chunk=chunk, psd_fix=psd_fix, psd_floor_rel=psd_floor_rel,
        )


def make_band_likelihood_engine(
    basis_settings: DomainSettingsBase,
    *,
    gb=None,
    gb_fd_comp=None,
    gb_wdm_comp=None,
    nchannels: int,
    tdi_channel_setup: str,
    df: Optional[float] = None,
    start_freq_inds=None,
    data_length: Optional[int] = None,
    opt_snr_rej_samp_limit: float = 5.0,
    snr_rej_detected: bool = False,
) -> BandLikelihoodEngine:
    """Construct the right engine for the supplied basis-domain settings.

    Dispatch keys off ``isinstance(basis_settings, ...)`` -- no string-level
    mode flag. The caller passes whichever ``gb_fd_comp`` / ``gb_wdm_comp``
    is appropriate for the domain it has selected; the missing one stays
    ``None``. ``gb`` (the legacy GBGPU handle) is accepted for backward
    compatibility but no longer consumed -- the FD engine runs on the new
    ``gb_fd_*`` kernels, not the SharedMemory family.
    """
    if isinstance(basis_settings, FDSettings):
        if gb_fd_comp is None:
            raise ValueError(
                "FDBandLikelihoodEngine requires a "
                "gbgpu.gbcomps.GBFDComputations prototype (pass "
                "gb_fd_comp=...); the legacy SharedMemory gb= path was "
                "retired in the 2026-07 rework."
            )
        if df is None:
            raise ValueError("FDBandLikelihoodEngine requires df.")
        # start_freq_inds / data_length are constructor-time fallbacks only;
        # the live values are read off the buffer ACA at call time
        # (min_freq_inds / data_shaped bin count).
        return FDBandLikelihoodEngine(
            gb_fd_comp=gb_fd_comp,
            basis_settings=basis_settings,
            nchannels=nchannels,
            tdi_channel_setup=tdi_channel_setup,
            df=df,
            start_freq_inds=start_freq_inds,
            data_length=data_length,
            opt_snr_rej_samp_limit=opt_snr_rej_samp_limit,
            snr_rej_detected=snr_rej_detected,
        )
    if isinstance(basis_settings, WDMSettings):
        if gb_wdm_comp is None:
            raise ValueError(
                "WDMBandLikelihoodEngine requires a "
                "gbgpu.gbcomps.GBWDMComputations instance (pass "
                "gb_wdm_comp=...)."
            )
        return WDMBandLikelihoodEngine(
            gb_comps=gb_wdm_comp,
            basis_settings=basis_settings,
            nchannels=nchannels,
            tdi_channel_setup=tdi_channel_setup,
            opt_snr_rej_samp_limit=opt_snr_rej_samp_limit,
            snr_rej_detected=snr_rej_detected,
        )
    raise NotImplementedError(
        f"No BandLikelihoodEngine for basis domain {type(basis_settings).__name__}."
    )
