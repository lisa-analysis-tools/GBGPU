"""FastGB-STFT prototype: heterodyned slow part on the STFT sub-grid -> narrow
per-segment DFT -> time-resolved inner product. Phase-1 prototype (design
docs/specs/2026-07-04-stft-gb-slowpart-fft-design.md). CPU/GPU via the `xp` of the
computation group; validated vs the injected brute STFT, benchmarked vs Fresnel."""

from copy import deepcopy

import numpy as np

from lisatools.utils.constants import C_SI


def slow_part_on_stft_grid(gb, params, t_seg, dt, n_sub):
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
    # local import to avoid a heavy import at module load (mirrors gbgpu.py deps)
    from gbgpu.gbgpu import GBGPU

    xp = gb.xp
    on_gpu = hasattr(xp, "cuda")

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
    gbtmp = GBGPU(
        orbits=deepcopy(gb.orbits),
        force_backend=("gpu" if on_gpu else "cpu"),
        t0=float(gb.t_ref),
        flip_ref_phase=True,
    )
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
    M = NT * n_sub

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
    alpha = float(getattr(gb, "window_alpha", 0.0) or 0.0)
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


def get_ll_stft_slowfft_proto(grp, gb, params, n_sub=32):
    xp = gb.xp
    num_bin = int(params.shape[0])
    d_h = xp.zeros(num_bin, dtype=xp.complex128)
    h_h = xp.zeros(num_bin, dtype=xp.complex128)
    return d_h, h_h  # stub; filled in Tasks 2-4
