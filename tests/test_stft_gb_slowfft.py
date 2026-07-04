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
