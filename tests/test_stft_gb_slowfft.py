import unittest
import numpy as np
try:
    from tests.test_stft_gb_fft import (          # reuse the reference fixture + source
        _build_fixture, _make_gb, _tukey,
        AMP, F0, FDOT, FDDOT, PHI0, INC, PSI, LAM, BETA)
    from gbgpu.stft_slowfft_proto import get_ll_stft_slowfft_proto
    HAVE = True
except (ImportError, ModuleNotFoundError) as _e:
    HAVE = False; _ERR = repr(_e)


def _build_multi_fixture(specs, alpha=0.0, N_sparse=8192):
    """Like ``_build_fixture`` but injects the SUM of several GBs as the data STFT.

    ``specs`` is a list of 9-element param lists (GBTDIonTheFly order
    ``(amp, f0, fdot, fddot, phi0, iota, psi, lam, beta)``). Sources should be
    well separated in ``f0`` so each recovers against a disjoint STFT band. The
    Tukey ``alpha`` suppresses inter-source spectral leakage (and exercises the
    Stage-2 taper branch). Returns the same dict as ``_build_fixture`` with
    ``params = np.array(specs)`` (shape ``(len(specs), 9)``).

    Same single-live-instance caveat as ``_build_fixture``: build + fully use one
    fixture before building another.
    """
    from lisatools.detector import DefaultOrbits
    from lisatools.response.tdiconfig import TDIConfig
    from lisatools.response.tdionfly import GBTDIonTheFly
    from lisatools.domains import TDSignal, TDSettings, get_stft_settings
    from lisatools.domaincomputation import STFTComputationGroup
    from lisatools.datacontainer import DataResidualArray
    from lisatools.sensitivity import XYZSensitivityBackend
    from lisatools.analysiscontainer import AnalysisContainer, AnalysisContainerArray

    fb = "cpu"; xp = np
    orbits = DefaultOrbits(force_backend=fb); orbits.configure(linear_interp_setup=True)
    tdi_config = TDIConfig("2nd generation", force_backend=fb)
    dt = 10.0; stft_dt = 6 * 3600.0; n_stft = 256
    nperseg = int(stft_dt / dt); nobs = n_stft * nperseg; Tobs = nobs * dt
    t_tdi = xp.linspace(0.0, Tobs, N_sparse + 1)[1:-1]
    data_t = xp.arange(nobs) * dt
    df_grid = 1.0 / stft_dt
    settings = get_stft_settings(data_t, stft_dt, min_freq=70 * df_grid,
                                 max_freq=115 * df_grid, force_backend=fb)
    gb_gen = GBTDIonTheFly(t_tdi, Tobs, 0.0, 1.0 / dt, 1, tdi_config=tdi_config,
                           orbits=orbits, tdi_chan="XYZ", force_backend=fb)
    keep = (data_t > t_tdi[0]) & (data_t < t_tdi[-1])
    tdi_output = xp.zeros((1, 3, nobs))
    for s in specs:                                    # data = sum of the sources
        out = gb_gen(*s, convert_to_ra_dec=False, return_spline=True)
        tdi_output[:, :, keep] += out.eval_tdi(data_t[keep])
    td_signal = TDSignal(tdi_output[0], settings=TDSettings(nobs, dt, 0.0, force_backend=fb))
    stft_signal = td_signal.stft(window=_tukey(nperseg, alpha), settings=settings)
    data_res = DataResidualArray(stft_signal)
    sens = XYZSensitivityBackend(orbits=orbits, settings=settings, force_backend=fb)
    sens.sens_mat = sens.compute_sensitivity_matrix(sens.basis_settings.f_arr, 15e-12, 3e-15)
    ac = AnalysisContainer(data_res, sens)
    acs = AnalysisContainerArray([ac], gpus=None)
    grp = STFTComputationGroup(acs, split_index=0, window_alpha=alpha, force_backend=fb)
    grp.compute_d_d_term()
    d_d = float(np.asarray(grp.d_d).reshape(-1)[0].real)
    params = np.array(specs)
    return dict(orbits=orbits, tdi_config=tdi_config, grp=grp, d_d=d_d,
                Tobs=Tobs, params=params)


def _build_fixture_dt(stft_hours, taper_s, N_sparse=8192):
    """``_build_fixture`` variant parametrized by the STFT segment length ``Delta``.

    Injects the SAME reference GB (``AMP, F0, ...`` from ``test_stft_gb_fft``) as the
    data STFT, but with:

    * ``stft_dt = stft_hours * 3600`` (the segment length under test);
    * ``n_stft = round(YRSID_SI / stft_dt)`` so ``Tobs ~ 1 yr`` at every ``Delta``
      (fixed physical baseline; only the segmentation changes);
    * a Tukey analysis window with ``alpha = min(1.0, 2*taper_s/stft_dt)``, i.e. a
      fixed ``~taper_s``-second taper per side (``taper_s=1e4`` -> a ~1e-4 Hz taper),
      independent of ``Delta`` (exercises the Stage-2 taper branch);
    * ``N_sparse=8192`` -- the SHARP ToF-spline oracle. The default 512 brute STFT is
      only ~4e-3 accurate and cannot validate a large-``Delta`` win (see
      ``_build_fixture``); 8192 gives a <1e-3-accurate ground truth.

    Band placement: ``_build_fixture`` hard-codes ``min_freq=70*df_grid,
    max_freq=115*df_grid`` with ``df_grid = 1/stft_dt``. Because ``df_grid`` scales
    with ``stft_dt``, that FIXED-bin band slides off the source carrier as ``Delta``
    grows (at 24 h ``F0`` sits at bin ~365, outside [70,115] -> the signal would fall
    entirely OUTSIDE the analysis band). So the band is re-centered on the carrier
    bin ``qc = round(F0*stft_dt)`` as ``[qc-25, qc+25]`` (51 active bins, carrier at
    ~bin 25), giving >=4 bins of margin beyond ``n_side=20`` at 6/24/96 h. Same
    single-live-instance caveat as ``_build_fixture``.
    """
    from lisatools.utils.constants import YRSID_SI
    from lisatools.detector import DefaultOrbits
    from lisatools.response.tdiconfig import TDIConfig
    from lisatools.response.tdionfly import GBTDIonTheFly
    from lisatools.domains import TDSignal, TDSettings, get_stft_settings
    from lisatools.domaincomputation import STFTComputationGroup
    from lisatools.datacontainer import DataResidualArray
    from lisatools.sensitivity import XYZSensitivityBackend
    from lisatools.analysiscontainer import AnalysisContainer, AnalysisContainerArray

    fb = "cpu"; xp = np
    orbits = DefaultOrbits(force_backend=fb); orbits.configure(linear_interp_setup=True)
    tdi_config = TDIConfig("2nd generation", force_backend=fb)
    dt = 10.0
    stft_dt = stft_hours * 3600.0
    n_stft = int(round(YRSID_SI / stft_dt))                 # Tobs ~ 1 yr, any Delta
    alpha = min(1.0, 2.0 * taper_s / stft_dt)               # fixed ~taper_s-per-side Tukey
    nperseg = int(stft_dt / dt); nobs = n_stft * nperseg; Tobs = nobs * dt
    t_tdi = xp.linspace(0.0, Tobs, N_sparse + 1)[1:-1]
    data_t = xp.arange(nobs) * dt
    df_grid = 1.0 / stft_dt
    qc = int(round(F0 * stft_dt))                           # source carrier bin at this df
    settings = get_stft_settings(data_t, stft_dt, min_freq=(qc - 25) * df_grid,
                                 max_freq=(qc + 25) * df_grid, force_backend=fb)
    gb_gen = GBTDIonTheFly(t_tdi, Tobs, 0.0, 1.0 / dt, 1, tdi_config=tdi_config,
                           orbits=orbits, tdi_chan="XYZ", force_backend=fb)
    out = gb_gen(AMP, F0, FDOT, FDDOT, PHI0, INC, PSI, LAM, BETA,
                 convert_to_ra_dec=False, return_spline=True)
    keep = (data_t > t_tdi[0]) & (data_t < t_tdi[-1])
    tdi_output = xp.zeros((1, 3, nobs)); tdi_output[:, :, keep] = out.eval_tdi(data_t[keep])
    td_signal = TDSignal(tdi_output[0], settings=TDSettings(nobs, dt, 0.0, force_backend=fb))
    stft_signal = td_signal.stft(window=_tukey(nperseg, alpha), settings=settings)
    data_res = DataResidualArray(stft_signal)
    sens = XYZSensitivityBackend(orbits=orbits, settings=settings, force_backend=fb)
    sens.sens_mat = sens.compute_sensitivity_matrix(sens.basis_settings.f_arr, 15e-12, 3e-15)
    ac = AnalysisContainer(data_res, sens)
    acs = AnalysisContainerArray([ac], gpus=None)
    grp = STFTComputationGroup(acs, split_index=0, window_alpha=alpha, force_backend=fb)
    grp.compute_d_d_term()
    d_d = float(np.asarray(grp.d_d).reshape(-1)[0].real)
    params = np.array([[AMP, F0, FDOT, FDDOT, PHI0, INC, PSI, LAM, BETA]])
    return dict(orbits=orbits, tdi_config=tdi_config, grp=grp, d_d=d_d,
                Tobs=Tobs, params=params)


@unittest.skipUnless(HAVE, "requires GBGPU STFT-GB build + LAT stack")
class SlowFFTSmokeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fx = _build_fixture()          # 6 h segments, alpha=0.0, injected brute STFT = data
    def test_returns_finite_shaped(self):
        gb = _make_gb(self.fx, n_side_bins=5)
        d_h, h_h = get_ll_stft_slowfft_proto(self.fx["grp"], gb, self.fx["params"], n_sub=32)
        self.assertEqual(np.asarray(d_h).shape, (1,))
        self.assertEqual(np.asarray(h_h).shape, (1,))
        self.assertTrue(np.all(np.isfinite(np.asarray(d_h))))


@unittest.skipUnless(HAVE, "requires GBGPU STFT-GB build + LAT stack")
class SlowPartStage1Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fx = _build_fixture()
    def test_slow_part_reconstructs_td_signal(self):
        from gbgpu.stft_slowfft_proto import slow_part_on_stft_grid
        gb = _make_gb(self.fx, n_side_bins=20)
        # Rebuild the SAME brute TD signal the fixture STFT'd, to compare against.
        # (fixture uses dt=10, stft_dt=6*3600, n_stft=256; see _build_fixture)
        dt = 10.0; stft_dt = 6*3600.0; n_stft = 256; nps = int(stft_dt/dt)
        t_seg = np.arange(n_stft) * stft_dt
        n_sub = 64
        E, q = slow_part_on_stft_grid(gb, self.fx["params"], t_seg, stft_dt, n_sub)
        self.assertEqual(np.asarray(E).shape, (1, 3, n_stft, n_sub))
        # Reconstruct s(tau) = Re[ E * exp(2pi i (q/T) tau) ] at the sub-grid and
        # compare to the brute GBTDIonTheFly TD signal at the same tau (mid-segment
        # sample). Full parity check lives in Task 3 (template vs brute STFT); here
        # assert E is finite, non-trivial, and smooth across sub-samples.
        E = np.asarray(E)
        self.assertTrue(np.all(np.isfinite(E)))
        self.assertGreater(np.abs(E).max(), 0.0)
        # smoothness: successive sub-sample differences are small vs the values
        d = np.abs(np.diff(E, axis=-1)); v = np.abs(E[..., :-1]) + 1e-300
        self.assertLess(np.median(d / v), 0.5)


@unittest.skipUnless(HAVE, "requires GBGPU STFT-GB build + LAT stack")
class SlowPartStage2Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # DENSE-spline ground truth. The brute STFT is the oracle; the default
        # N_sparse=512 spline (10800 s knots) is itself only ~4e-3 accurate -- it
        # cannot validate a <1e-3 template (measured: the E-template converges to the
        # brute as the spline sharpens, 3.95e-3 -> 1.38e-3 -> 7.4e-4 at 512/2048/8192).
        # A denser spline sharpens the GROUND TRUTH (strengthens, not weakens, the test).
        cls.fx = _build_fixture(N_sparse=8192)

    def test_template_matches_brute_stft(self):
        from gbgpu.stft_slowfft_proto import (
            slow_part_on_stft_grid, stft_template_from_slow_part)
        gb = _make_gb(self.fx, n_side_bins=20)
        stft_dt = 6 * 3600.0; n_stft = 256
        t_seg = np.arange(n_stft) * stft_dt
        settings = self.fx["grp"].settings
        # Brute-STFT ground truth [3, NT, NF_active] (C++ layout
        # [num_data, nchannels, num_times, num_freqs]; num_data==1 here).
        D = np.asarray(self.fx["grp"].data_arr).reshape(3, settings.NT, settings.NF_active)

        def band_mm(n_sub):
            E, q = slow_part_on_stft_grid(gb, self.fx["params"], t_seg, stft_dt, n_sub)
            H = np.asarray(stft_template_from_slow_part(
                gb, E, q, t_seg, stft_dt, n_sub, 20, settings))[0]
            m = np.abs(D) > 0
            num = np.vdot(D[m], H[m])
            den = np.sqrt(np.vdot(D[m], D[m]) * np.vdot(H[m], H[m]))
            return abs(1.0 - (num / den).real)

        mm32, mm64 = band_mm(32), band_mm(64)
        print(f"\n[stage2] template-vs-brute mm: n_sub=32 {mm32:.3e}  n_sub=64 {mm64:.3e}")
        self.assertLess(mm64, mm32)      # converges with n_sub
        self.assertLess(mm64, 1e-3)      # matches the brute STFT template


@unittest.skipUnless(HAVE, "requires GBGPU STFT-GB build + LAT stack")
class SlowPartRecoveryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Sharp oracle: N_sparse=8192 spline (~7e-4 accurate) -- the default 512
        # brute STFT is only ~4e-3, too coarse to trust a <1e-2 recovery claim.
        cls.fx = _build_fixture(N_sparse=8192)

    def test_recovers_injection_and_converges(self):
        gb = _make_gb(self.fx, n_side_bins=20)
        d_d = self.fx["d_d"]; mms = []
        for n_sub in (32, 64, 96):
            d_h, h_h = get_ll_stft_slowfft_proto(self.fx["grp"], gb, self.fx["params"], n_sub=n_sub)
            d_h = complex(np.asarray(d_h).reshape(-1)[0]); h_h = float(np.asarray(h_h).reshape(-1)[0].real)
            mms.append(abs(1.0 - d_h.real / np.sqrt(d_d * h_h)))
            print(f"[slowfft-recovery] n_sub={n_sub:3d} mm={mms[-1]:.3e}")
        self.assertLess(mms[-1], 1e-2)       # recovers the injection
        self.assertLessEqual(mms[-1], mms[0]) # converges with n_sub

    def test_matches_fresnel_short_segments(self):
        # Order-independent by construction: build a FRESH fixture *inside* the method
        # instead of reusing the shared setUpClass group. The Fresnel oracle below needs
        # a LIVE C++ STFTDomain, but the sibling test's slow-FFT Stage-1 GBGPU clobbers
        # the shared group -- so a rename/reorder (or any earlier slowfft call) that ran
        # the sibling first would score a clobbered domain -> garbage oracle -> silent
        # false-pass. Own fixture + Fresnel FIRST, THEN the slow-FFT (which clobbers)
        # mirrors test_stft_gb_fft.py::test_fft_windowed_recovers_and_matches_fresnel.
        fx = _build_fixture(N_sparse=8192)
        gb = _make_gb(fx, n_side_bins=20)
        gb.get_ll_stft(fx["params"])                     # Fresnel oracle: needs a live group
        d_h_f = complex(np.asarray(gb.d_h_out).reshape(-1)[0]); h_h_f = float(np.asarray(gb.h_h_out).reshape(-1)[0].real)
        mm_f = abs(1.0 - d_h_f.real / np.sqrt(fx["d_d"] * h_h_f))
        d_h, h_h = get_ll_stft_slowfft_proto(fx["grp"], gb, fx["params"], n_sub=64)  # clobbers the group
        d_h = complex(np.asarray(d_h).reshape(-1)[0]); h_h = float(np.asarray(h_h).reshape(-1)[0].real)
        mm_x = abs(1.0 - d_h.real / np.sqrt(fx["d_d"] * h_h))
        print(f"[slowfft-vs-fresnel 6h] fresnel {mm_f:.3e}  slowfft {mm_x:.3e}")
        self.assertLessEqual(mm_x, mm_f * 1.20 + 1e-4)   # at least Fresnel accuracy at 6 h


@unittest.skipUnless(HAVE, "requires GBGPU STFT-GB build + LAT stack")
class SlowFFTWindowedRecoveryTest(unittest.TestCase):
    """Recovery with a Tukey analysis window (alpha>0): exercises the Stage-2
    taper branch, which the rectangular fixtures never touch. The template's
    per-sub-sample window must match the windowed data STFT, else <d|h> falls."""
    @classmethod
    def setUpClass(cls):
        cls.fx = _build_fixture(alpha=0.3, N_sparse=8192)

    def test_windowed_recovers(self):
        gb = _make_gb(self.fx, n_side_bins=20)
        d_d = self.fx["d_d"]
        d_h, h_h = get_ll_stft_slowfft_proto(self.fx["grp"], gb, self.fx["params"], n_sub=64)
        d_h = complex(np.asarray(d_h).reshape(-1)[0]); h_h = float(np.asarray(h_h).reshape(-1)[0].real)
        mm = abs(1.0 - d_h.real / np.sqrt(d_d * h_h))
        print(f"[slowfft-windowed alpha=0.3] mm={mm:.3e}")
        self.assertLess(mm, 1e-2)


@unittest.skipUnless(HAVE, "requires GBGPU STFT-GB build + LAT stack")
class SlowFFTMultiSourceTest(unittest.TestCase):
    """>=2 sources with different phi0/f0/sky in one params batch, each recovering
    against a shared data STFT that holds both -- validates the reconciliation is
    param-independent (the vectorized Stage-1/2 handles a heterogeneous batch, no
    cross-talk, correct per-source phi0-flip + tdi2 factor).

    Sources are placed in disjoint STFT bands (f0 ~ bin 78 and ~ bin 108, n_side=5
    => 11-bin bands, ~19-bin gap) and the data is Tukey-windowed to suppress
    inter-source spectral leakage. Recovery per source uses the projection form
    ``|1 - Re<d|h_i>/<h_i|h_i>|`` (the shared <d|d> holds BOTH sources, so the
    cosine-overlap normalization would under-count each source's power)."""
    # (amp, f0, fdot, fddot, phi0, iota, psi, lam, beta); f0 in [70,115]/stft_dt.
    SRC_A = [1.0e-23, 3.6300e-3, 1.0e-18, 0.0, 0.8923, 1.2310, 3.0091, 4.8273, -0.5092]
    SRC_B = [1.3e-23, 4.9800e-3, 2.0e-18, 0.0, 2.1044, 0.9007, 1.5003, 2.3010,  0.4001]

    @classmethod
    def setUpClass(cls):
        cls.fx = _build_multi_fixture([cls.SRC_A, cls.SRC_B], alpha=0.3, N_sparse=8192)

    @staticmethod
    def _proj_mm(d_h, h_h):
        d_h = np.asarray(d_h).reshape(-1); h_h = np.asarray(h_h).reshape(-1).real
        return np.abs(1.0 - d_h.real / h_h)          # per-source projection recovery

    def test_each_source_recovers(self):
        gb = _make_gb(self.fx, n_side_bins=5)
        params = self.fx["params"]                    # [[A],[B]]
        d_h, h_h = get_ll_stft_slowfft_proto(self.fx["grp"], gb, params, n_sub=64)
        mm = self._proj_mm(d_h, h_h)
        print(f"[slowfft-multi] mm_A={mm[0]:.3e}  mm_B={mm[1]:.3e}")
        self.assertLess(mm[0], 1e-2)                  # source A recovers
        self.assertLess(mm[1], 1e-2)                  # source B recovers

    def test_batch_order_independent(self):
        # Swapping the batch order swaps the outputs (no cross-talk / no source-0
        # params leaking into source 1): confirms param-independence of the batch.
        gb = _make_gb(self.fx, n_side_bins=5)
        d_h, h_h = get_ll_stft_slowfft_proto(self.fx["grp"], gb, self.fx["params"], n_sub=64)
        d_h_s, h_h_s = get_ll_stft_slowfft_proto(
            self.fx["grp"], gb, self.fx["params"][::-1], n_sub=64)
        d_h = np.asarray(d_h).reshape(-1); d_h_s = np.asarray(d_h_s).reshape(-1)
        np.testing.assert_allclose(d_h, d_h_s[::-1], rtol=1e-10, atol=0.0)


@unittest.skipUnless(HAVE, "requires GBGPU STFT-GB build + LAT stack")
class SlowPartLargeDeltaTest(unittest.TestCase):
    """The value proposition: slow-FFT stays EXACT at large STFT segment Delta, where
    the analytic Fresnel kernel is not even usable.

    Ground truth is the INJECTED brute STFT, never Fresnel (design 2026-07-04 §6:
    "Ground truth = injected brute STFT, never Fresnel (Fresnel has its own model
    error)"). slow-FFT samples the response envelope at ``n_sub`` sub-points (exact
    midpoint quadrature) and heterodynes against the FastGB carrier ``q/T``, so it
    recovers the injection at every Delta -- mm <= 1e-4 @ 24 h, <= 1e-3 @ 96 h
    (measured 7.05e-6 / 3.21e-4).

    Fresnel (``get_ll_stft``) is scored + PRINTED as documented evidence ONLY -- it is
    numerically DEGENERATE at Delta >= 24 h for this near-monochromatic source, so its mm
    is unreliable (nan @ 24 h; a garbage finite value such as ~4e-3 @ 96 h, itself
    process-history-dependent) and is NEVER asserted on. Two root causes (Task-5 report):
      (1) ``freq_from_tdi_phase`` derives the per-pixel frequency from a central finite
          difference of half-width ``stft_dt``; the first segment samples the orbit at
          ``t < 0`` (orbit starts at t=0) and extrapolates to garbage -- worsening with
          Delta: OK @ 6 h -> zeroed template @ 24 h (h_h ~ -3e-38) -> overflow (~1e260)
          @ 96 h; and
      (2) the analytic Fresnel value's ``amp/sqrt(2|fdot0|)`` / large-argument Fresnel
          integrals degenerate for the fdot=1e-18 source (chirp bandwidth ~1e-13 Hz per
          segment; fails even with ``freq_from_tdi_phase=False``).
    slow-FFT is immune to both (in-range sub-grid + FastGB carrier, no get_freq_index).

    Order within each ``hours`` block: a FRESH fixture per ``hours``, Fresnel scored
    FIRST (its C++ path needs a LIVE ``STFTDomain``) THEN slow-FFT (whose Stage-1
    private ``GBGPU`` clobbers the live group's device buffers) -- mirrors
    ``test_matches_fresnel_short_segments``."""

    def test_recovers_injection_at_large_delta(self):
        target = {24.0: 1e-4, 96.0: 1e-3}                    # injection-oracle recovery gate
        for hours in (24.0, 96.0):                           # 1 day, 4 days
            fx = _build_fixture_dt(hours, taper_s=1e4)       # ~1 yr, fixed ~1e-4 Hz taper
            gb = _make_gb(fx, n_side_bins=20)
            # Fresnel is EVIDENCE ONLY (printed, never asserted): degenerate at large
            # Delta (finite-diff samples the orbit at t<0; analytic 1/sqrt(fdot)) -> mm_f
            # is nan/garbage. Score it first: its C++ path needs a still-live group.
            gb.get_ll_stft(fx["params"])
            d_h_f = complex(np.asarray(gb.d_h_out).reshape(-1)[0])
            h_h_f = float(np.asarray(gb.h_h_out).reshape(-1)[0].real)
            mm_f = (abs(1.0 - d_h_f.real / np.sqrt(fx["d_d"] * h_h_f))
                    if np.isfinite(h_h_f) and h_h_f > 0.0 else float("nan"))
            # slow-FFT scored against the INJECTION (grp.d_d) -- the real oracle. Its
            # Stage-1 private GBGPU clobbers the group, so it MUST run after Fresnel.
            d_h, h_h = get_ll_stft_slowfft_proto(fx["grp"], gb, fx["params"], n_sub=96)
            d_h = complex(np.asarray(d_h).reshape(-1)[0])
            h_h = float(np.asarray(h_h).reshape(-1)[0].real)
            mm_x = abs(1.0 - d_h.real / np.sqrt(fx["d_d"] * h_h))
            print(f"[large-dt {hours:.0f}h] slowfft mm={mm_x:.3e}   "
                  f"(fresnel mm={mm_f:.3e} -- degenerate, evidence only)")
            self.assertLess(mm_x, target[hours])   # slow-FFT recovers the injected GB
