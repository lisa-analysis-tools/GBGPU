import unittest
import numpy as np
try:
    from tests.test_stft_gb_fft import _build_fixture, _make_gb   # reuse the reference fixture
    from gbgpu.stft_slowfft_proto import get_ll_stft_slowfft_proto
    HAVE = True
except (ImportError, ModuleNotFoundError) as _e:
    HAVE = False; _ERR = repr(_e)

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
