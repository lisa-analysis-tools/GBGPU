"""JAX mirror of the C++ signal-het v2 polyphase + bin-fold pipeline.

Stage 2a consumer: takes a precomputed sparse heterodyne FD ``X_het`` (the
output of ``gb_run_fd_wave_tdi`` after the fftshift + 1/dt conversion to
the centered-slice convention -- see
``[stage2b-in-kernel-conversion]`` memory) and produces per-binary
``<d|h>`` and ``<h|h>`` via the same polyphase fold + bin-fold formula as
the C++ kernel.

This sits between two future pieces:
  * ``gb_signal_het_fd_jax`` (TBD) -- JAX FD generation equivalent of
    ``gb_run_fd_wave_tdi``, would feed X_het into this consumer.
  * ``gb_signal_het_get_ll_jax`` (TBD) -- end-to-end (params -> logL)
    wrapper, supports ``jax.grad`` through the whole pipeline.

The forward path validates bit-for-bit against C++
``gb_signal_het_get_ll_sparse_wrap``; ``jax.grad`` through this consumer
validates against C++ central-difference grad (the C++ grad pulls
through the FD generation too, so for now we can only compare the
"polyphase + bin-fold" piece -- a partial-chain-rule comparison).

The ``max_r`` clip mirrors the C++ ``max_r`` parameter: it caps |r| per
channel-cell to prevent the positive-logL blowup at angle excursions
where the channel covariance matrix is rank-degenerate. See
``[signal-het-amp-phase-redesign]`` memory.

References
----------
- C++:  ``lisa-on-gpu/src/fastlisaresponse/cutils/TDIonTheFly.cu``
        ``GBComputationGroup::gb_signal_het_get_ll_sparse_wrap`` and
        ``gb_signal_het_get_ll_wrap``.
- Python mirror: ``LISAanalysistools/scripts/gb_chunked_het/
                  gb_signal_het_v2_sparse_mirror.py``.
- Stage 2b in-kernel conversion (fftshift + 1/dt): memory
  ``project_stage2b_in_kernel_conversion.md``.
- V2 plan: ``~/.claude/plans/yes-find-and-read-sprightly-garden.md``.
"""

from __future__ import annotations

from functools import partial
from typing import Tuple

import jax
import jax.numpy as jnp

from lisatools.jax.response.base import JaxAmpPhaseSource
from lisatools.jax.orbits import OrbitsWrapJAX
from lisatools.jax.response.projection import (
    get_phase_ref, get_tdi_Xf_single, get_sky_vectors,
)
from lisatools.jax.response.tdi_config import TDIConfigWrapJAX

from .fast_inner_heterodyne import _tukey_window, _resolve_alpha


def _safe_divide_with_clip(c1: jnp.ndarray, c0: jnp.ndarray,
                            floor_eps: float, max_r: float
                            ) -> jnp.ndarray:
    """Compute r = c1 / c0 with the C++ safe-divide floor AND |r| clip.

    Matches the C++ behaviour:
      * floor_th = max(floor_eps * max|c0| per (c, m_active), 1e-300)
      * cells with |c0| <= floor_th -> r = 0
      * cells with |c0| > floor_th  -> r = c1 / c0, then clip |r| <= max_r
    All operations are JAX-friendly (no booleans, only jnp.where).
    """
    abs_c0 = jnp.abs(c0)
    max_c0 = jnp.max(abs_c0, axis=-1, keepdims=True)
    floor_th = jnp.maximum(floor_eps * max_c0, 1e-300)
    safe = abs_c0 > floor_th
    denom = jnp.where(safe, c0, 1.0 + 0.0j)
    r = jnp.where(safe, c1 / denom, 0.0 + 0.0j)
    # Apply |r| clip: only when max_r > 0; preserve direction.
    abs_r = jnp.abs(r)
    scale = jnp.where(
        (max_r > 0.0) & (abs_r > max_r),
        max_r / jnp.maximum(abs_r, 1e-300),
        1.0,
    )
    return r * scale


def _polyphase_fold_one_bin(
    X_het_bin: jnp.ndarray,     # (nch, N_sparse_fd) cmplx
    k_f0: int,                   # scalar int
    m_active: jnp.ndarray,       # (M,) int
    wdm_window: jnp.ndarray,     # (Nt,) double
    n_sparse_local_arr: jnp.ndarray,   # (N_sparse_t,) int
    Nf: int, Nt: int, Nt_layer: int, N_sparse_t: int,
    N_sparse_fd: int, stride: int, ind_min_t: int,
    dt: float,
) -> jnp.ndarray:
    """Polyphase fold -> iFFT -> apply lisatools coef.

    Returns c1_sparse shape (nchannels, M, N_sparse_t).

    For each absolute FD bin in the N_sparse_fd window around k_f0, check
    which active m-layer it falls into (j in [0, Nt)), apply the window +
    prephase, accumulate into the polyphase fold buffer of length Nt_layer
    via j % Nt_layer, then iFFT.
    """
    nchannels = X_het_bin.shape[0]
    M = m_active.shape[0]
    half_Nt = Nt // 2
    half_NS = N_sparse_fd // 2
    TWO_PI = 2.0 * jnp.pi
    kappa = 2.0 * jnp.sqrt(jnp.pi * dt) / float(Nf)
    n_start = ind_min_t + n_sparse_local_arr[0]

    # Polyphase fold via a vectorized scatter.  For each (i, im, c) we need
    # to add X[c, i] * window_pref to fold[c, im, j % Nt_layer], BUT only if
    # j = k_abs - m_active[im] * half_Nt + half_Nt is in [0, Nt).
    i_arr     = jnp.arange(N_sparse_fd)                # (N_sparse_fd,)
    k_abs     = k_f0 + (i_arr - half_NS)                # (N_sparse_fd,)
    # j[im, i] = k_abs[i] - m_active[im] * half_Nt + half_Nt
    j_arr     = k_abs[None, :] - m_active[:, None] * half_Nt + half_Nt
    valid     = (j_arr >= 0) & (j_arr < Nt)             # (M, N_sparse_fd)
    j_clamped = jnp.clip(j_arr, 0, Nt - 1)              # safe indexing
    j_off     = j_clamped - half_Nt
    phase_arg = TWO_PI * j_off * n_start / Nt           # (M, N_sparse_fd)
    prephase  = jnp.exp(1j * phase_arg)
    w_j       = wdm_window[j_clamped]                   # (M, N_sparse_fd)
    weighting = jnp.where(valid, w_j * prephase, 0.0)   # (M, N_sparse_fd)
    r_slot    = j_clamped % Nt_layer                    # (M, N_sparse_fd)

    # X_het_bin: (nch, N_sparse_fd); broadcast to (nch, M, N_sparse_fd)
    weighted = X_het_bin[:, None, :] * weighting[None, :, :]

    # Scatter (c, im, r_slot[im, i]) += weighted[c, im, i] for valid entries.
    # JAX scatter:  use jax.ops.segment_sum on flattened (im, i) -> (im, r_slot)
    # with segment_ids = im * Nt_layer + r_slot.
    seg_ids = m_active[:, None] * 0 + (jnp.arange(M)[:, None] * Nt_layer
                                          + r_slot)    # (M, N_sparse_fd)
    seg_ids_flat = seg_ids.reshape(-1)                  # (M*N_sparse_fd,)

    # Flatten per channel and segment-sum
    fold = jnp.zeros((nchannels, M * Nt_layer), dtype=jnp.complex128)
    weighted_flat = weighted.reshape(nchannels, -1)     # (nch, M*N_sparse_fd)
    for c in range(nchannels):
        fold = fold.at[c].set(
            jax.ops.segment_sum(weighted_flat[c], seg_ids_flat,
                                 num_segments=M * Nt_layer)
        )
    fold = fold.reshape(nchannels, M, Nt_layer)

    # iFFT length Nt_layer along last axis
    ifft_full = jnp.fft.ifft(fold, n=Nt_layer, axis=-1)  # (nch, M, Nt_layer)

    # Keep first N_sparse_t outputs, multiply by lisatools per-pixel coef.
    n_layer_arr = jnp.arange(N_sparse_t)                # (N_sparse_t,)
    n_global    = n_start + n_layer_arr * stride        # (N_sparse_t,)
    sign_scale  = ((-1.0) ** n_global) / float(stride)  # (N_sparse_t,)
    after = ifft_full[:, :, :N_sparse_t] * sign_scale[None, None, :]

    # Per (m_active, n_layer) coefficient (matches the C++ kernel layout).
    m_global    = m_active                               # (M,)
    m_plus_n    = (m_global[:, None] + n_global[None, :]) & 1   # (M, Nsp)
    conj_cmn    = jnp.where(m_plus_n == 0,
                             1.0 + 0.0j, 0.0 - 1.0j)     # (M, Nsp)
    sign_mn_int = ((m_global[:, None] + 1) * n_global[None, :]) & 1
    sign_mn     = jnp.where(sign_mn_int == 0, 1.0, -1.0)
    coef        = kappa * sign_mn * conj_cmn             # (M, Nsp)

    c1_sparse = after * coef[None, :, :]                 # (nch, M, Nsp)
    return c1_sparse


def gb_signal_het_get_ll_sparse_jax(
    X_het_all: jnp.ndarray,         # (num_bin, nch, N_sparse_fd) cmplx
    k_f0_all: jnp.ndarray,          # (num_bin,) int
    c0_sparse_all: jnp.ndarray,     # (num_data, nch, Nf_active, N_sparse_t) cmplx
    A0_all: jnp.ndarray,            # same shape as c0_sparse_all
    A1_all: jnp.ndarray,            # same
    B0_all: jnp.ndarray,            # (num_data, nch, nch, Nf_active, N_sparse_t)
    B1_all: jnp.ndarray,            # same
    wdm_window: jnp.ndarray,        # (Nt,) float
    n_sparse_local_arr: jnp.ndarray, # (N_sparse_t,) int
    params_cand_all: jnp.ndarray,    # (num_bin, nparams) float
    data_index_all: jnp.ndarray,     # (num_bin,) int
    *,
    nparams: int, f0_idx: int,
    Nf: int, Nt: int, Nf_active: int, Nt_layer: int, N_sparse_t: int,
    stride: int, ind_min_t: int, ind_min_f: int,
    m_active_half_width: int,
    layer_df: float, dt: float,
    nchannels: int, tdi_type: int,
    N_sparse_fd: int,
    max_r: float = 0.0,
    floor_eps: float = 1e-12,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """JAX mirror of GBComputationGroup::gb_signal_het_get_ll_sparse_wrap.

    Returns (d_h, h_h) arrays of length num_bin. logL = -0.5 d_d + d_h - 0.5 h_h
    is computed by the caller (we don't have d_d here).

    The function is jit-friendly except for the inner loop over m_active
    layers which has dynamic indexing (m_active depends on params_cand
    through f0). We unroll the M=5 layers via a Python loop (M is static).

    max_r > 0 caps |r| per channel-cell; max_r <= 0 disables the clip
    (matches C++ semantics).
    """
    M = 2 * m_active_half_width + 1

    def per_bin(args):
        X_het_bin, k_f0, params_cand, data_idx = args

        # m_active = clip(m_floor + arange(-h, h+1), ind_min_f, ind_min_f + Nf_active - 1)
        f0_cand = params_cand[f0_idx]
        m_floor = jnp.floor(f0_cand / layer_df).astype(jnp.int32)
        m_active = jnp.clip(
            m_floor + jnp.arange(-m_active_half_width,
                                  m_active_half_width + 1),
            ind_min_f, ind_min_f + Nf_active - 1,
        )
        m_local = m_active - ind_min_f                  # (M,) int

        # 1) Polyphase fold + iFFT + lisatools per-pixel coef.
        c1_sparse = _polyphase_fold_one_bin(
            X_het_bin, k_f0, m_active, wdm_window,
            n_sparse_local_arr,
            Nf, Nt, Nt_layer, N_sparse_t, N_sparse_fd, stride,
            ind_min_t, dt,
        )                                                # (nch, M, Nsp)

        # 2) r = c1/c0 (safe divide + max_r clip).
        # Gather c0 slices per m_local using dynamic_slice (static M).
        # c0_sparse_all has shape (num_data, nch, Nf_active, Nsp).
        c0_data = c0_sparse_all[data_idx]                # (nch, Nf_active, Nsp)
        # Gather along axis 1 by m_local (M small, unroll)
        def gather_m(arr, m_idx):
            # arr: (..., Nf_active, Nsp); pick m_idx along Nf_active.
            return jax.lax.dynamic_slice_in_dim(arr, m_idx, 1, axis=-2)[..., 0, :]
        c0_active = jnp.stack(
            [gather_m(c0_data, m_local[k]) for k in range(M)], axis=1
        )                                                 # (nch, M, Nsp)
        r = _safe_divide_with_clip(c1_sparse, c0_active, floor_eps, max_r)

        # 3) dr/dn central FD with stride spacing.
        Dn = float(stride)
        dr = jnp.zeros_like(r)
        # interior: (r[+1] - r[-1]) / (2 Dn)
        dr = dr.at[..., 1:-1].set((r[..., 2:] - r[..., :-2]) / (2.0 * Dn))
        dr = dr.at[..., 0].set((r[..., 1] - r[..., 0]) / Dn)
        dr = dr.at[..., -1].set((r[..., -1] - r[..., -2]) / Dn)

        # 4) Bin-fold inner products.
        A0_data = A0_all[data_idx]                       # (nch, Nf_active, Nsp)
        A1_data = A1_all[data_idx]
        # Gather A0, A1 per m_local (M static).
        A0_active = jnp.stack(
            [gather_m(A0_data, m_local[k]) for k in range(M)], axis=1)
        A1_active = jnp.stack(
            [gather_m(A1_data, m_local[k]) for k in range(M)], axis=1)
        d_h_c = jnp.sum(A0_active * r + A1_active * dr)

        if tdi_type == 0:   # XYZ -- full cross-channel
            B0_data = B0_all[data_idx]    # (nch, nch, Nf_active, Nsp)
            B1_data = B1_all[data_idx]
            # Gather along Nf_active dim
            def gather_m_5d(arr, m_idx):
                return jax.lax.dynamic_slice_in_dim(arr, m_idx, 1,
                                                     axis=-2)[..., 0, :]
            B0_active = jnp.stack(
                [gather_m_5d(B0_data, m_local[k]) for k in range(M)],
                axis=-2)              # (nch, nch, M, Nsp)
            B1_active = jnp.stack(
                [gather_m_5d(B1_data, m_local[k]) for k in range(M)],
                axis=-2)
            # r outer: conj(r_c) * r_c2  -> (nch, nch, M, Nsp)
            r_outer  = jnp.conj(r[:, None, :, :]) * r[None, :, :, :]
            cross_rd = (jnp.conj(r[:, None, :, :])  * dr[None, :, :, :]
                       + jnp.conj(dr[:, None, :, :]) * r[None, :, :, :])
            h_h_c = jnp.sum(B0_active * r_outer + B1_active * cross_rd)
        else:               # AE / AET diag
            B0_data = B0_all[data_idx]   # (nch, Nf_active, Nsp)
            B1_data = B1_all[data_idx]
            B0_active = jnp.stack(
                [gather_m(B0_data, m_local[k]) for k in range(M)], axis=1)
            B1_active = jnp.stack(
                [gather_m(B1_data, m_local[k]) for k in range(M)], axis=1)
            rsq      = (jnp.conj(r) * r).real
            cross_rd = jnp.conj(r) * dr + jnp.conj(dr) * r
            h_h_c = jnp.sum(B0_active * rsq + B1_active * cross_rd)

        return 0.5 * d_h_c.real, 0.5 * h_h_c.real

    # vmap over num_bin
    d_h_arr, h_h_arr = jax.vmap(per_bin)(
        (X_het_all, k_f0_all, params_cand_all, data_index_all)
    )
    return d_h_arr, h_h_arr


# ============================================================================
# FD generation (mirror of C++ gb_run_fd_wave_tdi + Stage 2b conversion)
# ============================================================================
#
# C++ gb_run_fd_wave_tdi produces X_het in FFT order, scaled by 0.5*dts
# (where dts = T_obs/N_sparse_fd). Stage 2b's CPU branch then converts to
# the centered-slice / dense-rfft convention via fftshift + (1/dt). We
# build the centered output directly in JAX so the downstream consumer
# can be jax.grad'd through this.
#
# Conversion: dense_rfft[k_f0 + d] = (1/dt) * fftshift(X_het_raw)[d + N/2]
#           = (1/dt) * X_het_raw[(d + N/2) mod N]  for d in [-N/2, N/2-1]
# ============================================================================


def gb_signal_het_fd_centered_jax(
    params: jnp.ndarray,                  # (9,) float
    t_start: float, T_obs: float,
    source: JaxAmpPhaseSource,
    orbits: OrbitsWrapJAX,
    tdi_config: TDIConfigWrapJAX,
    k_sky: jnp.ndarray, u_sky: jnp.ndarray, v_sky: jnp.ndarray,
    N_sparse_fd: int, nchannels: int,
    dt: float,
    tukey_alpha: float,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """JAX FD generation for one binary, centered-slice convention.

    Returns:
        X_het: complex (nchannels, N_sparse_fd) -- centered slice ready
               for the polyphase consumer. Bin i corresponds to absolute
               dense-rfft bin k_f0 + (i - N_sparse_fd/2).
        k_f0:  scalar int32 -- absolute dense-rfft bin index of the
               snapped carrier f0_grid = round(f0/df_abs) * df_abs.

    Time origin: ``t_start``, matching the C++ kernel. Both this output
    and the dense rfft of (Tukey * td) sliced around k_f0 use the same
    time origin, so no extra linear-phase factor is needed (the (1/dt)
    scale absorbs the sparse-to-dense Riemann conversion -- see memory
    project_stage2b_in_kernel_conversion.md).
    """
    f0 = params[1]
    dt_sparse = T_obs / N_sparse_fd
    t_sparse  = t_start + jnp.arange(N_sparse_fd) * dt_sparse

    def _per_t(t):
        M = get_tdi_Xf_single(t, params, source, orbits, tdi_config,
                              k_sky, u_sky, v_sky)
        phi = get_phase_ref(t, params, source, orbits)
        return M, phi
    Ms, phis = jax.vmap(_per_t)(t_sparse)
    tdi_amp   = jnp.abs(Ms)
    tdi_phase = -jnp.angle(Ms) - phis[:, None]

    df_abs = 1.0 / T_obs
    k_f0   = jnp.round(f0 / df_abs).astype(jnp.int32)
    f0_grid = k_f0.astype(jnp.float64) * df_abs

    tau     = t_sparse - t_start
    carrier = 2.0 * jnp.pi * f0_grid * tau
    slow    = tdi_amp * jnp.exp(
        +1j * (tdi_phase + phis[:, None] - carrier[:, None])
    )                                                   # (N_sparse_fd, nch)

    alpha_eff = _resolve_alpha(tukey_alpha, N_sparse_fd)
    if alpha_eff > 0.0:
        slow = slow * _tukey_window(N_sparse_fd, alpha_eff)[:, None]

    X_het_raw = 0.5 * dt_sparse * jnp.fft.fft(slow, axis=0)  # (N_sparse_fd, nch)

    # Convert FFT order + 0.5*dts scale -> centered slice + (1/dt) scale.
    # = (1/dt) * fftshift(X_het_raw, axis=0)
    X_het_shifted = jnp.fft.fftshift(X_het_raw, axes=0) / dt   # (N_sparse_fd, nch)
    # Return as (nch, N_sparse_fd) to match consumer convention.
    return X_het_shifted.T, k_f0


def gb_signal_het_get_ll_in_kernel_jax(
    params_batch: jnp.ndarray,           # (num_bin, 9)
    c0_sparse_all: jnp.ndarray,          # (num_data, nch, Nf_active, N_sparse_t)
    A0_all: jnp.ndarray, A1_all: jnp.ndarray,
    B0_all: jnp.ndarray, B1_all: jnp.ndarray,
    wdm_window: jnp.ndarray,             # (Nt,)
    n_sparse_local_arr: jnp.ndarray,     # (N_sparse_t,)
    params_ref_all: jnp.ndarray,         # (num_data, 9)
    data_index_all: jnp.ndarray,         # (num_bin,)
    source: JaxAmpPhaseSource,
    orbits: OrbitsWrapJAX,
    tdi_config: TDIConfigWrapJAX,
    *,
    nparams: int, f0_idx: int,
    Nf: int, Nt: int, Nf_active: int, Nt_active: int,
    Nt_layer: int, N_sparse_t: int, stride: int,
    ind_min_t: int, ind_min_f: int, m_active_half_width: int,
    layer_df: float, dt: float,
    T_obs: float, t_start: float,
    nchannels: int, tdi_type: int,
    N_sparse_fd: int,
    tukey_alpha: float,
    max_r: float = 0.0,
    floor_eps: float = 1e-12,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """End-to-end JAX Stage 2b: params -> logL.

    Mirrors C++ ``gb_signal_het_get_ll_in_kernel_wrap`` -- one call per
    binary in a vmap-friendly form. With ``jax.grad`` this gives the
    analytic gradient of logL = d_h - 0.5 h_h w.r.t. params, in ~1-2x
    the cost of one forward call (vs 17x for central-difference).

    Returns:
        d_h: (num_bin,) float
        h_h: (num_bin,) float
    Caller computes logL = -0.5 d_d + d_h - 0.5 h_h.
    """
    # Per-binary FD generation.
    k_sky_b, u_sky_b, v_sky_b = jax.vmap(
        lambda p: get_sky_vectors(p, source)
    )(params_batch)                                      # each (num_bin, 3)

    def _fd_one(params, k_s, u_s, v_s):
        return gb_signal_het_fd_centered_jax(
            params, t_start, T_obs, source, orbits, tdi_config,
            k_s, u_s, v_s, N_sparse_fd, nchannels, dt, tukey_alpha,
        )
    X_het_all, k_f0_all = jax.vmap(_fd_one)(
        params_batch, k_sky_b, u_sky_b, v_sky_b,
    )                                                    # X_het: (num_bin, nch, N_sparse_fd)

    return gb_signal_het_get_ll_sparse_jax(
        X_het_all, k_f0_all,
        c0_sparse_all,
        A0_all, A1_all, B0_all, B1_all,
        wdm_window, n_sparse_local_arr,
        params_batch, data_index_all,
        nparams=nparams, f0_idx=f0_idx,
        Nf=Nf, Nt=Nt, Nf_active=Nf_active,
        Nt_layer=Nt_layer, N_sparse_t=N_sparse_t,
        stride=stride,
        ind_min_t=ind_min_t, ind_min_f=ind_min_f,
        m_active_half_width=m_active_half_width,
        layer_df=layer_df, dt=dt,
        nchannels=nchannels, tdi_type=tdi_type,
        N_sparse_fd=N_sparse_fd,
        max_r=max_r, floor_eps=floor_eps,
    )
