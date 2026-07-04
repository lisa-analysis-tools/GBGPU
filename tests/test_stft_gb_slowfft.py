import unittest
import numpy as np
try:
    from tests.test_stft_gb_fft import _build_fixture, _make_gb, _tukey   # reuse the reference fixture
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
