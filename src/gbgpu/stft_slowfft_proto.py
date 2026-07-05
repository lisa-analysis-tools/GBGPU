"""FastGB-STFT prototype: heterodyned slow part on the STFT sub-grid -> narrow
per-segment DFT -> time-resolved inner product. Phase-1 prototype (design
docs/specs/2026-07-04-stft-gb-slowpart-fft-design.md). CPU/GPU via the `xp` of the
computation group; validated vs the injected brute STFT, benchmarked vs Fresnel."""

from copy import deepcopy

import numpy as np

from lisatools.utils.constants import C_SI


def make_slowpart_gbgpu(gb):
    """Build the private FastGB ``GBGPU`` used by :func:`slow_part_on_stft_grid`.

    Constructs a ``GBGPU`` on a **deepcopy** of ``gb.orbits`` (so ``gb``'s privately
    owned orbit object is never mutated), on the same backend and reference epoch as
    ``gb`` (``t0=gb.t_ref``), with ``flip_ref_phase=True`` (the ``GBTDIonTheFly``
    reference-phase convention -- see :func:`slow_part_on_stft_grid`).

    Exposed as a helper so a caller that evaluates the slow part MANY times (a
    benchmark, a sampler) can build this object ONCE and pass it back via
    ``slow_part_on_stft_grid(..., gbtmp=...)`` / ``get_ll_stft_slowfft_proto(...,
    gbtmp=...)``. Construction re-runs the orbit configuration and is the dominant
    per-call cost (seconds), yet it is orbit/param/``n_sub``-independent, so hoisting
    it out of the per-call path is both correct and a large speedup. Building it
    invalidates a live :class:`STFTComputationGroup`'s C++ device state (the documented
    "two live instances" hazard) -- identical to the inline path it replaces.
    """
    from gbgpu.gbgpu import GBGPU

    xp = gb.xp
    on_gpu = hasattr(xp, "cuda")
    return GBGPU(
        orbits=deepcopy(gb.orbits),
        force_backend=("gpu" if on_gpu else "cpu"),
        t0=float(gb.t_ref),
        flip_ref_phase=True,
    )


def slow_part_on_stft_grid(gb, params, t_seg, dt, n_sub, gbtmp=None):
    """Stage 1: FastGB heterodyned time-domain "slow part" on the STFT sub-grid.

    Reuses the classic ``GBGPU`` FastGB slow part (the heterodyned,
    response-modulated envelope built by :meth:`GBGPU._construct_slow_part` +
    the time-domain assembly of :meth:`GBGPU._computeXYZ`) but samples it on the
    STFT segment sub-grid instead of the FastGB FFT grid. No FFT is taken -- we
    stop at the time-domain ``XYZsl`` (``gbgpu.py`` line 510), the smooth
    envelope heterodyned against the carrier ``q/T``.

    Args:
        gb (STFTGBComputations): supplies ``orbits``, ``tdi_config``, ``T``,
            ``t_ref`` and the array module ``xp``.
        params (array): ``[num_bin, 9]`` in ``GBTDIonTheFly`` order
            ``(amp, f0, fdot, fddot, phi0, iota, psi, lam, beta)``.
        t_seg (array): ``[NT]`` STFT segment start times (relative, seconds).
        dt (float): STFT segment length ``stft_dt`` (seconds).
        n_sub (int): sub-samples per segment.
        gbtmp (GBGPU, optional): a prebuilt private FastGB ``GBGPU`` (from
            :func:`make_slowpart_gbgpu`) to REUSE instead of constructing one on this
            call. ``None`` (default) builds it inline (behavior unchanged). Reuse is
            safe -- ``GBGPU`` is designed for many waveform calls -- and hoists the
            dominant (~seconds) per-call construction out of a repeated-eval loop.

    Returns:
        (E, q): ``E`` xp complex ``[num_bin, 3, NT, n_sub]`` heterodyned
        time-domain slow part (X/Y/Z); ``q`` int ``[num_bin]`` FastGB carrier
        bin ``rint(f0 * T)``.

    Notes:
        ``E`` is the FastGB *slow part* with the amplitude divided out (the
        ``/amp`` in ``fctr2``; the ``ampl*`` restore on ``gbgpu.py`` line 513 is
        deliberately skipped, per the design). It also excludes the TDI-2 factor
        (``gbgpu.py`` lines 434-436) and the ``0.5*T/N`` FFT normalization.
        Restoring ``amp`` and applying the per-segment DFT + (for a 2nd-gen data
        STFT) the TDI-2 factor is Task 3's job (``stft_template_from_slow_part``).

        ``E`` IS generated in the ``GBTDIonTheFly`` reference-phase convention
        (``flip_ref_phase=True`` -> ``-phi0``), so no phi0-dependent phase remains
        for Task 3 to apply: the only FastGB->GBTDIonTheFly reconciliation left is
        the param-independent ``tdi2_factor(f0)`` (derived from ``q``).
    """
    xp = gb.xp

    # A private GBGPU on the SAME orbits / backend / reference epoch as `gb`.
    # deepcopy the orbits: GBGPUBase's setter re-runs ``configure`` in place, and
    # we must not mutate gb's (privately owned) orbit object.
    #
    # ``flip_ref_phase=True`` (-> the slow part is built with -phi0) matches the
    # ``GBTDIonTheFly`` (LAT) reference-phase convention, which is the data source
    # for this prototype. gbgpu's FastGB default (+phi0) differs from GBTDIonTheFly
    # by exactly exp(-2i*phi0); this is the "constant phase offset" the Task-2 report
    # measured as ~-102 deg (== -2*phi0 for that source's phi0=0.892). Homing the
    # phi0 convention here (at slow-part GENERATION, where phi0 is applied) leaves
    # the Stage-2 template with only the param-INDEPENDENT, derivable TDI-generation
    # factor to reconcile -- verified: after the flip the FastGB->GBTDIonTheFly gap
    # is a pure ``tdi2_factor(f0)`` (arg +64.6 deg), no phi0-dependent residual.
    #
    # Building this GBGPU re-runs the orbit configuration and is the dominant per-call
    # cost (~seconds); it is orbit/param/``n_sub``-independent, so a caller evaluating
    # the slow part repeatedly may build it ONCE with :func:`make_slowpart_gbgpu` and
    # pass it in as ``gbtmp`` (default ``None`` -> build inline; behavior unchanged).
    if gbtmp is None:
        gbtmp = make_slowpart_gbgpu(gb)
    T = float(gb.T)
    t0_abs = gbtmp.t0_abs                 # == gb.t_ref (0.0 for the fixture)
    arm_length = gbtmp.orbits.armlength

    # ---- parameters, GBTDIonTheFly / run_wave order ----
    params = np.atleast_2d(np.asarray(params))
    num_bin = int(params.shape[0])
    amp   = np.atleast_1d(params[:, 0]).copy()
    f0    = np.atleast_1d(params[:, 1]).copy()
    fdot  = np.atleast_1d(params[:, 2]).copy()
    fddot = np.atleast_1d(params[:, 3]).copy()
    phi0  = np.atleast_1d(params[:, 4]).copy()
    iota  = np.atleast_1d(params[:, 5]).copy()
    psi   = np.atleast_1d(params[:, 6]).copy()
    lam   = np.atleast_1d(params[:, 7]).copy()
    beta  = np.atleast_1d(params[:, 8]).copy()

    # === run_wave argument construction (gbgpu.py:254-418), verbatim ===
    # polar angle from ecliptic latitude
    theta = np.pi / 2 - beta

    if gbtmp.flip_ref_phase:
        # Active here (flip_ref_phase=True above): negate phi0 so the slow part is
        # generated in the GBTDIonTheFly reference-phase convention (removes the
        # exp(-2i*phi0) FastGB<->ToF gap). Same mechanism run_wave uses to match jaxgb.
        phi0 = -phi0

    # copy to GPU if needed
    amp   = xp.asarray(amp.copy())
    f0    = xp.asarray(f0.copy())
    fdot  = xp.asarray(fdot.copy())
    fddot = xp.asarray(fddot.copy())
    phi0  = xp.asarray(phi0.copy())
    iota  = xp.asarray(iota.copy())
    psi   = xp.asarray(psi.copy())
    lam   = xp.asarray(lam.copy())
    theta = xp.asarray(theta.copy())

    cosiota = xp.cos(iota)

    # transfer frequency
    fstar = C_SI / (arm_length * 2 * np.pi)

    cosps, sinps = xp.cos(2.0 * psi), xp.sin(2.0 * psi)

    Aplus = amp * (1.0 + cosiota * cosiota)
    Across = -2.0 * amp * cosiota

    DP = Aplus * cosps - 1.0j * Across * sinps
    DC = -Aplus * sinps - 1.0j * Across * cosps

    # sky location basis vectors
    sinth, costh = xp.sin(theta), xp.cos(theta)
    sinph, cosph = xp.sin(lam), xp.cos(lam)
    u = xp.array([costh * cosph, costh * sinph, -sinth]).T[:, None, :]
    v = xp.array([sinph, -cosph, xp.zeros_like(cosph)]).T[:, None, :]
    k = xp.array([-sinth * cosph, -sinth * sinph, -costh]).T[:, None, :]

    # polarization tensors
    eplus = xp.matmul(v.transpose(0, 2, 1), v) - xp.matmul(u.transpose(0, 2, 1), u)
    ecross = xp.matmul(u.transpose(0, 2, 1), v) + xp.matmul(v.transpose(0, 2, 1), u)

    # === STFT sub-grid (midpoint quadrature, matches FFTColumn) ===
    t_seg = xp.asarray(t_seg)
    NT = int(t_seg.shape[0])
    sub = (xp.arange(n_sub) + 0.5) * (dt / n_sub)         # [n_sub]
    tm = (t_seg[:, None] + sub[None, :]).reshape(-1)      # [M], relative time

    # spacecraft on the ABSOLUTE sub-grid (run_wave feeds _spacecraft tm_abs),
    # evaluated once and reused across binaries.
    Ps = gbtmp._spacecraft(tm + t0_abs)

    # slow part on the RELATIVE sub-grid (run_wave feeds _construct_slow_part tm_rel)
    Gs, q = gbtmp._construct_slow_part(
        T, arm_length, Ps, tm,
        f0, fdot, fddot, fstar, phi0, k, DP, DC, eplus, ecross,
    )

    # === _computeXYZ time-domain assembly (gbgpu.py:486-510), stop before fft ===
    f = (
        f0[:, None]
        + fdot[:, None] * tm[None, :]
        + 1 / 2 * fddot[:, None] * tm[None, :] ** 2
    )
    omL = f / fstar
    SomL = xp.sin(omL)
    fctr = xp.exp(-1.0j * omL)
    fctr2 = 4.0 * omL * SomL * fctr / amp[:, None]

    Xsl = Gs["21"] - Gs["31"] + (Gs["12"] - Gs["13"]) * fctr
    Ysl = Gs["32"] - Gs["12"] + (Gs["23"] - Gs["21"]) * fctr
    Zsl = Gs["13"] - Gs["23"] + (Gs["31"] - Gs["32"]) * fctr

    # time domain slow part [num_bin, 3, M]
    XYZsl = fctr2[:, None, :] * xp.array([Xsl, Ysl, Zsl]).transpose(1, 0, 2)

    E = XYZsl.reshape(num_bin, 3, NT, n_sub)
    q = xp.asarray(q).reshape(num_bin).astype(xp.int64)
    return E, q


def stft_template_from_slow_part(gb, E, q, t_seg, dt, n_sub, n_side, settings):
    """Stage 2: narrow per-segment DFT of the slow part -> STFT template columns.

    Ports :cpp:class:`FFTColumn` (``lat_stft_kernels.hh``) to Python, operating on
    the precomputed slow part ``E`` instead of ``get_tdi_Xf_single``. For each STFT
    segment it windows the ``n_sub`` sub-samples, re-heterodynes from the FastGB
    carrier ``q/T`` to the segment's carrier bin, and does a targeted DFT to the
    ``2*n_side+1`` frequency bins around that carrier -- the same windowed-Fourier
    value the analytic Fresnel/`FFTColumn` per-pixel path builds, but from ``E``.

    Reconciliation FastGB(``E``) -> the ``GBTDIonTheFly`` brute STFT: the only
    factor applied here is the **param-independent** TDI-generation factor
    ``tdi2_factor(f0) = 2j*sin(2*wL)*exp(-2j*wL)``, ``wL = 2*pi*f0*L/c`` with
    ``f0 = q/T`` (``gbgpu.py`` :func:`GBGPU.run_wave` lines 434-436, evaluated at the
    carrier). The phi0 reference-phase convention (the Task-2 report's ~-102 deg
    "constant phase", == ``-2*phi0``) is handled upstream in
    :func:`slow_part_on_stft_grid` (``flip_ref_phase=True``), so no phi0-dependent
    phase is needed here -- the reconciliation is param-independent.

    The DFT is written as a direct twiddle sum; it is algebraically identical to
    ``FFTColumn``'s recurrence ``0.5*dts_sub * base^diff * sum_m demod[m]*(W^diff)^m``
    (``base = exp(-2j*pi*(df*t_seg + 0.5/n_sub))``, ``W = exp(-2j*pi/n_sub)``) since
    ``df*dt == 1`` makes ``base^diff*(W^diff)^m = exp(-2j*pi*diff*df*tau_m)``.

    Args:
        gb: supplies ``xp``, ``T``, ``orbits.armlength`` and the analysis window
            (``window_alpha``).
        E (array): ``[num_bin, 3, NT, n_sub]`` raw heterodyned slow part from
            :func:`slow_part_on_stft_grid` (already in the ToF phi0 convention).
        q (array): ``[num_bin]`` FastGB carrier bins.
        t_seg (array): ``[NT]`` STFT segment start times (relative, seconds).
        dt (float): STFT segment length ``stft_dt`` (seconds).
        n_sub (int): sub-samples per segment (must match ``E``).
        n_side (int): half-width (in freq bins) of the template band per segment.
        settings: the group's :class:`STFTSettings` (``min_freq`` == C++ ``f_min``,
            ``df``, ``NF_active`` == C++ ``num_freqs``).

    Returns:
        ``H`` xp complex ``[num_bin, 3, NT, NF_active]`` STFT template, zero outside
        the ``2*n_side+1`` band around each segment's carrier bin.

    Notes:
        ``H`` is **amplitude-free** (``E`` has ``amp`` divided out and the signature
        carries no ``amp``). The band mismatch vs the brute STFT is amplitude-
        invariant, so this is exact for the Task-3 test; the Task-4 likelihood must
        scale by ``amp`` (available there from ``params``).
    """
    xp = gb.xp
    E = xp.asarray(E)
    q = xp.asarray(q)
    num_bin = int(E.shape[0])
    NT = int(E.shape[2])

    # settings: brief names f_min/num_freqs map to the STFTSettings attrs the C++
    # STFTDomain is actually built from (domaincomputation.py: min_freq, NF_active).
    df = float(settings.df)
    f_min = float(settings.min_freq)
    num_freqs = int(settings.NF_active)

    T = float(gb.T)
    arm = float(gb.orbits.armlength)
    dts_sub = dt / n_sub

    # Analysis window per sub-sample (midpoint tau_local = (m+0.5)*dts_sub), matching
    # FFTColumn: Tukey taper of duration alpha*dt/2 when window_alpha>0, else flat.
    # The authoritative alpha is the computation group's ``window_alpha`` -- the very
    # value the windowed data STFT + the C++ Fresnel/FFT paths use. ``STFTGBComputations``
    # carries no ``window_alpha`` of its own, so a bare ``getattr(gb, "window_alpha")``
    # would silently default to 0.0 and skip the taper on a windowed fixture (leaving
    # the template rectangular vs windowed data -> lost overlap). An explicit
    # ``gb.window_alpha`` still overrides the group when present.
    alpha = getattr(gb, "window_alpha", None)
    if alpha is None:
        alpha = getattr(getattr(gb, "stft_comps", None), "window_alpha", 0.0)
    alpha = float(alpha or 0.0)
    m_idx = xp.arange(n_sub)
    tau_local = (m_idx + 0.5) * dts_sub                    # [n_sub], relative to seg start
    w = xp.ones(n_sub)
    if alpha > 0.0:
        taper = alpha * dt / 2.0
        left = tau_local < taper
        w = xp.where(left, 0.5 * (1.0 - xp.cos(np.pi * tau_local / taper)), w)
        right = tau_local > dt - taper
        w = xp.where(right, 0.5 * (1.0 - xp.cos(np.pi * (dt - tau_local) / taper)), w)

    t_seg = xp.asarray(t_seg)
    tau = t_seg[:, None] + tau_local[None, :]              # [NT, n_sub] absolute times
    H = xp.zeros((num_bin, 3, NT, num_freqs), dtype=xp.complex128)

    for b in range(num_bin):
        carrier_f = float(q[b]) / T
        freq_j = int((carrier_f - f_min) / df)             # STFTDomain.get_freq_index(f0)
        lo = max(0, freq_j - n_side)
        hi = min(num_freqs - 1, freq_j + n_side)
        if hi < lo:
            continue

        # TDI-generation factor at the carrier (param-independent given q). Host scalar.
        omegaL = 2.0 * np.pi * carrier_f * (arm / C_SI)
        tdi2 = 2.0j * np.sin(2.0 * omegaL) * np.exp(-2.0j * omegaL)

        freq_here = f_min + xp.arange(lo, hi + 1) * df      # [nb] absolute band freqs
        # de-heterodyne E (carrier q/T) + window, then targeted DFT to the band:
        #   H[c,seg,bin] = 0.5*dts_sub*tdi2 * sum_m w_m E[c,seg,m]
        #                    * exp(2j*pi*(q/T)*tau_m) * exp(-2j*pi*freq_here*tau_m)
        g = w[None, None, :] * E[b] * xp.exp(2.0j * np.pi * carrier_f * tau)[None, :, :]  # [3,NT,ns]
        ph = xp.exp(-2.0j * np.pi * freq_here[:, None, None] * tau[None, :, :])            # [nb,NT,ns]
        band = xp.einsum("csm,bsm->csb", g, ph)             # [3, NT, nb]
        H[b, :, :, lo:hi + 1] = (0.5 * dts_sub * tdi2) * band

    return H


def get_ll_stft_slowfft_proto(grp, gb, params, n_sub=32, gbtmp=None):
    """Fill the FastGB-STFT prototype end-to-end and score the injection.

    Stage 1 (:func:`slow_part_on_stft_grid`, heterodyned slow part on the STFT
    sub-grid) -> Stage 2 (:func:`stft_template_from_slow_part`, narrow per-segment
    DFT template ``H``) -> restore the amplitude -> score ``H`` against the data
    STFT + ``invC``.

    Scoring contracts ``H`` against the data STFT + ``invC`` with the one-sided
    factor, exactly as :meth:`STFTComputationGroup.compute_signal_likelihood_terms`'s
    C++ kernel does (``get_inner_product_cross`` + ``4 * domain.diff_comp``):

        ``d_h = 4*df * sum_{t,f,ci,cj} conj(D[ci]) invC[ci,cj] H[cj]``
        ``h_h = 4*df * sum_{t,f,ci,cj} conj(H[ci]) invC[ci,cj] H[cj]``

    (full XYZ 3x3 cross-channel contraction; diagonal AET path handled too). We
    contract manually -- reading ``grp.data_arr`` / ``grp.invC_arr`` / ``settings.df``
    -- rather than calling the group kernel, because Stage 1 constructs a private
    ``GBGPU`` and **the LAT STFT computation group's C++ ``STFTDomain`` device state
    is invalidated by building a second waveform/orbits instance** (the documented
    "STFT groups do not support two live instances" hazard; verified:
    ``compute_signal_likelihood_terms`` returns ~1e-44 garbage after Stage 1, while
    the numpy ``data_arr`` / ``invC_arr`` snapshots stay intact and reproduce the
    kernel's ``<d|d>`` to machine precision). The brief sanctions this manual
    fallback. The two paths agree bit-for-bit pre-clobber (validated).

    Args:
        grp (STFTComputationGroup): owns the data STFT (``grp.data_arr``), the
            inverse covariance (``grp.invC_arr``) and the grid (``grp.settings``).
        gb (STFTGBComputations): supplies ``xp``, ``T``, ``orbits``, ``n_side_bins``
            and -- via ``gb.stft_comps.window_alpha`` -- the analysis window.
        params (array): ``[num_bin, 9]`` in GBTDIonTheFly order.
        n_sub (int): sub-samples per STFT segment (Stage-1/2 quadrature density).
        gbtmp (GBGPU, optional): prebuilt private FastGB ``GBGPU`` (see
            :func:`make_slowpart_gbgpu`) reused across calls to hoist the ~seconds
            construction out of the per-call path; ``None`` builds it inline.

    Returns:
        (d_h, h_h): xp complex ``[num_bin]`` raw inner products, also stored on
        ``gb`` as ``d_h_out_slowfft`` / ``h_h_out_slowfft`` (mirrors how
        :meth:`get_ll_stft` stores ``d_h_out`` / ``h_h_out``). The caller forms the
        recovery ``mm = |1 - d_h.real / sqrt(d_d * h_h.real)|`` with
        ``d_d = grp.d_d``.
    """
    xp = gb.xp
    settings = grp.settings

    # Prototype validated only at t_ref==0: the absolute/relative STFT time split below
    # (Stage-1 relative t_seg vs Stage-2 absolute, tied to gb.t_ref == settings.t0 == 0)
    # would need re-deriving for a nonzero epoch. Fail loud rather than score silently.
    assert float(gb.t_ref) == 0.0, (
        "prototype validated only at t_ref==0; nonzero epoch needs the "
        "abs/rel time split re-derived (Phase 2)")

    p = np.atleast_2d(np.asarray(params))
    amp = xp.asarray(p[:, 0].copy())                    # [num_bin]; H is amp-free

    NT = int(settings.NT)
    NF = int(settings.NF_active)
    nch = int(grp.num_channels)
    df = float(settings.df)
    stft_dt = float(settings.dt)
    t0 = float(settings.t0)

    # Snapshot the (numpy/xp) data + inverse-covariance BEFORE Stage 1 builds a
    # GBGPU (which clobbers the group's C++ domain but leaves these host arrays
    # intact). data_arr: [num_data, nch, NT, NF]; take data_index 0 (the injection).
    D = xp.asarray(grp.data_arr).reshape(-1, nch, NT, NF)[0]           # [nch, NT, NF]
    invC = xp.asarray(grp.invC_arr)

    # STFT segment start times on the domain's ABSOLUTE t0 (the phase reference the
    # data STFT uses), NOT a 0-based grid: a 0-based grid would incur a per-segment
    # phase error whenever settings.t0 != 0 (Task-3 review). The same t_seg feeds
    # Stage 1 (relative, since gb.t_ref == settings.t0 here) and Stage 2 (absolute).
    t_seg = t0 + xp.arange(NT) * stft_dt

    # Stage 1 -> Stage 2 (n_side from the group's per-binary side-band config).
    # ``gbtmp`` (optional, from :func:`make_slowpart_gbgpu`) is passed through so a
    # repeated-eval caller can hoist the ~seconds private-GBGPU construction out of the
    # per-call path (``None`` -> build inline; behavior unchanged).
    E, q = slow_part_on_stft_grid(gb, p, t_seg, stft_dt, n_sub, gbtmp=gbtmp)
    H = stft_template_from_slow_part(
        gb, E, q, t_seg, stft_dt, n_sub, gb.n_side_bins, settings)      # [num_bin,nch,NT,NF]

    # Restore the amplitude divided out in Stage 1 (H is amplitude-free) before scoring.
    H = H * amp[:, None, None, None]

    # One-sided noise-weighted inner products (4*df factor == C++ 4*diff_comp).
    four_df = 4.0 * df
    # The XYZ(3x3)/AET(diagonal) disambiguation below keys off invC.size, which is
    # unambiguous ONLY at num_noise==1: a multi-noise AET invC [num_noise,nch,NT,NF]
    # aliases a single-noise XYZ invC [1,nch,nch,NT,NF]. Guard the assumption loudly.
    num_noise = int(grp.num_noise)
    assert num_noise == 1, (
        "prototype supports single-noise (num_noise==1); "
        "multi-noise invC disambiguation is a Phase-2 item")
    if invC.size == nch * nch * NT * NF:                # XYZ full 3x3 cross-channel
        invC = invC.reshape(-1, nch, nch, NT, NF)[0]                   # [nch,nch,NT,NF]
        d_h = four_df * xp.einsum("itf,ijtf,bjtf->b", xp.conj(D), invC, H)
        h_h = four_df * xp.einsum("bitf,ijtf,bjtf->b", xp.conj(H), invC, H)
    else:                                               # AET diagonal
        invC = invC.reshape(-1, nch, NT, NF)[0]                        # [nch,NT,NF]
        d_h = four_df * xp.einsum("itf,itf,bitf->b", xp.conj(D), invC, H)
        h_h = four_df * xp.einsum("bitf,itf,bitf->b", xp.conj(H), invC, H)

    d_h = xp.ascontiguousarray(d_h.astype(xp.complex128))
    h_h = xp.ascontiguousarray(h_h.astype(xp.complex128))
    gb.d_h_out_slowfft = d_h
    gb.h_h_out_slowfft = h_h
    return d_h, h_h
