"""Test-support reference implementations. Not for use in runs.

Holds the numerical references the kernels are validated against, kept out of the computation
classes themselves so nothing in a run can reach them by accident.
"""

from typing import Optional

__all__ = ["stft_information_matrix_reference"]


def stft_information_matrix_reference(comp, params, holder=None, inds=None,
                                      param_eps=None, param_scales=None,
                                      param_eps_relative=1.0e-6,
                                      easy_central_difference: bool = False,
                                      noise_index=None,
                                      convert_to_ra_dec: Optional[bool] = None,
                                      batch_size: Optional[int] = None,
                                      linear_envelope: Optional[bool] = False,
                                      astro_track: bool = True):
    """Reference STFT Fisher: whole-grid templates from ``fill_global_stft``, contracted by einsum.

    The numerical reference for :meth:`STFTGBComputations.information_matrix`, which builds and
    contracts the derivative templates per pixel instead. Independent of the kernel at every step
    above the pixel evaluator, so it is what checks the four-point stencil, the cross-channel
    contraction and the ``inds`` mapping; the pixel ground-truth tests cover none of those.

    Its workspace is 114 ``(NT, NF_active)`` complex planes per source, hence ``batch_size``.

    Derivative templates ``dh/dtheta_i`` are central differences of the ``fill_global_stft`` rows,
    second order by default and first order with ``easy_central_difference``, folded into

    ``Gamma_ij = 4 df Re sum_{c,d,t,f} dh_i,c^*(t,f) invC_{c,d}(t,f) dh_j,d(t,f)``

    which is the normalisation the ``gb_stft_get_ll`` kernel applies at the fill-path template
    scale. XYZ uses the full cross-channel invC; channel-diagonal groups use the per-channel
    product.

    ``linear_envelope`` is INERT either way: the templates are built on the astro track, which
    exports no stencil amplitudes, so the first-moment slope is exactly zero.

    ! The kernel differs by design on a source whose astro carrier sits on a bin boundary. A
    ! derivative's four stencil templates share the union of their supports there, where this
    ! reference keeps each template on its own, and the two disagree by 1e3 to 2e4. Exclude
    ! boundary sources from any parity comparison.

    Args:
        comp: the :class:`STFTGBComputations` whose group, steps and fill path are used.
        params: ``(num_bin, nparams)`` GB parameters (1D auto-promoted), in the run frame the
            STFT kernels consume; ``convert_to_ra_dec`` must be falsy.
        holder: ``None`` uses the bound ``stft_comps``; an ``AnalysisContainerArray`` resolves to
            its single split's group.
        inds: parameter indices to include (default all ``nparams``).
        param_eps / param_scales / param_eps_relative: finite-difference steps. Note that
            ``param_eps_relative`` is read only when ``param_scales`` is given; otherwise the
            absolute default table is used and the argument does nothing.
        easy_central_difference: first-order derivative when ``True``.
        noise_index: per-binary index into the group's invC rows.
        batch_size: source-batch size bounding the ``dh`` workspace (default: all at once).

    Returns:
        ``(num_bin, num_derivs, num_derivs)`` real Fisher matrices.
    """
    if convert_to_ra_dec:
        raise ValueError(
            "stft_information_matrix_reference takes run-frame params directly; "
            "convert_to_ra_dec is not supported."
        )
    xp = comp.xp
    group = comp._resolve_info_group(holder)

    p = comp._prep_params(params)
    num_bin, nparams = p.shape
    if inds is None:
        inds = list(range(nparams))
    else:
        inds = [int(i) for i in xp.asarray(inds).tolist()]
    num_derivs = len(inds)

    eps_theta, scales = comp._information_matrix_steps(
        nparams, param_eps, param_scales, param_eps_relative,
    )

    info = xp.zeros((num_bin, num_derivs, num_derivs), dtype=xp.float64)
    if num_bin == 0:
        return info

    _, noise_index = comp._resolve_indices(num_bin, None, noise_index)
    assert int(noise_index.max()) < group.num_noise

    settings = group.settings
    n_t, n_f = int(settings.NT), int(settings.NF_active)
    nch = int(group.num_channels)
    df = float(settings.df)

    # The SAME owned complex buffer the C++ domain reads (pinned by the group; see
    # _owned_cpp_array's dangling-pointer contract).
    inv_flat = group._owned_cpp_array("invC", group.invC_arr, xp.complex128)
    if inv_flat.size == group.num_noise * nch * nch * n_t * n_f:
        inv_rows = inv_flat.reshape(group.num_noise, nch, nch, n_t, n_f)
        cross_channel = True
    elif inv_flat.size == group.num_noise * nch * n_t * n_f:
        inv_rows = inv_flat.reshape(group.num_noise, nch, n_t, n_f)
        cross_channel = False
    else:
        raise ValueError(
            f"invC buffer size {inv_flat.size} matches neither the cross-channel nor the "
            f"diagonal layout for num_noise={group.num_noise}, nch={nch}, NT={n_t}, "
            f"NF_active={n_f}."
        )

    if batch_size is None or batch_size <= 0:
        batch_size = num_bin

    # fill_global_stft routes through comp.stft_comps' cpp objects; bind the resolved group for
    # the duration so the fill grid and the invC rows come from one source of truth.
    #
    # ! Templates are ALWAYS evaluated on the astro track (freq_from_tdi_phase=False) whatever the
    # ! sampling configuration. The TDI-phase-derived per-column (f0, fdot) estimates carry an
    # ! evaluation-noise floor, so central differences across that machinery diverge as 1/eps. The
    # ! astro-track family differences exactly (dh_phi0 == -i*h to machine precision) and agrees
    # ! with the TDI-phase family to the ~1e-4-relative Doppler in <h|h>, which is far below the
    # ! finite-difference budget.
    prev_group = comp.stft_comps
    prev_fftp = comp.freq_from_tdi_phase
    try:
        with comp._envelope_context(linear_envelope, group=group):
            if group is not prev_group:
                comp.stft_comps = group
            if astro_track:
                comp.freq_from_tdi_phase = False

            for start in range(0, num_bin, batch_size):
                stop = min(start + batch_size, num_bin)
                params_batch = p[start:stop]
                n_batch = stop - start
                row_idx = xp.arange(n_batch, dtype=xp.int32)
                buf = xp.zeros((n_batch, nch, n_t, n_f), dtype=xp.complex128)

                def _templates(shifted):
                    buf[:] = 0.0
                    comp.fill_global_stft(shifted, buf, data_index=row_idx)
                    return buf.copy()

                dh = xp.zeros((n_batch, num_derivs, nch, n_t, n_f), dtype=xp.complex128)
                for di, ind in enumerate(inds):
                    step = float(eps_theta[ind])
                    up1 = params_batch.copy(); up1[:, ind] += step
                    dn1 = params_batch.copy(); dn1[:, ind] -= step
                    h_up1 = _templates(up1)
                    h_dn1 = _templates(dn1)
                    if easy_central_difference:
                        dh[:, di] = (h_up1 - h_dn1) / (2.0 * step)
                    else:
                        up2 = params_batch.copy(); up2[:, ind] += 2.0 * step
                        dn2 = params_batch.copy(); dn2[:, ind] -= 2.0 * step
                        h_up2 = _templates(up2)
                        h_dn2 = _templates(dn2)
                        dh[:, di] = (-h_up2 + h_dn2 + 8.0 * (h_up1 - h_dn1)) / (12.0 * step)

                invc_b = inv_rows[noise_index[start:stop]]
                if cross_channel:
                    info[start:stop] = (4.0 * df * xp.einsum(
                        "bicts,bcdts,bjdts->bij", dh.conj(), invc_b, dh)).real
                else:
                    info[start:stop] = (4.0 * df * xp.einsum(
                        "bicts,bjcts,bcts->bij", dh.conj(), dh, invc_b)).real
    finally:
        comp.freq_from_tdi_phase = prev_fftp
        if group is not prev_group:
            comp.stft_comps = prev_group

    if scales is not None:
        scales_sel = xp.asarray([scales[i] for i in inds], dtype=xp.float64)
        info = info * scales_sel[None, :, None] * scales_sel[None, None, :]
    return info
