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
        STFT) the TDI-2 factor is Task 3's job when forming the actual template.
    """
    # local import to avoid a heavy import at module load (mirrors gbgpu.py deps)
    from gbgpu.gbgpu import GBGPU

    xp = gb.xp
    on_gpu = hasattr(xp, "cuda")

    # A private GBGPU on the SAME orbits / backend / reference epoch as `gb`.
    # deepcopy the orbits: GBGPUBase's setter re-runs ``configure`` in place, and
    # we must not mutate gb's (privately owned) orbit object.
    gbtmp = GBGPU(
        orbits=deepcopy(gb.orbits),
        force_backend=("gpu" if on_gpu else "cpu"),
        t0=float(gb.t_ref),
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
        # if matching jaxgb, then we need to input - phi0
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


def get_ll_stft_slowfft_proto(grp, gb, params, n_sub=32):
    xp = gb.xp
    num_bin = int(params.shape[0])
    d_h = xp.zeros(num_bin, dtype=xp.complex128)
    h_h = xp.zeros(num_bin, dtype=xp.complex128)
    return d_h, h_h  # stub; filled in Tasks 2-4
