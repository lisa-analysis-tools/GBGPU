"""FFT-per-column STFT/Fresnel GB get_ll: smoke + accuracy + N_sub convergence + cost.

Head-to-head against the analytic Fresnel get_ll (the correctness oracle) and the
injected-GB brute STFT, mirroring test_stft_gb_accuracy.py's fixture. The FFT path
(FFTColumn policy) computes each STFT column's template as a targeted DFT of the
response sub-sampled at n_sub points -- a Riemann sum of the same per-segment
integral the Fresnel kernel evaluates analytically -- so Fresnel get_ll is the
oracle and error must shrink as n_sub grows.
"""
import time
import unittest
import numpy as np

try:
    from lisatools.detector import DefaultOrbits
    from lisatools.response.tdiconfig import TDIConfig
    from lisatools.response.tdionfly import GBTDIonTheFly
    from lisatools.domains import TDSignal, TDSettings, get_stft_settings
    from lisatools.domaincomputation import STFTComputationGroup
    from lisatools.datacontainer import DataResidualArray
    from lisatools.sensitivity import XYZSensitivityBackend
    from lisatools.analysiscontainer import AnalysisContainer, AnalysisContainerArray
    from gbgpu.gbcomps import STFTGBComputations
    HAVE = True
except (ImportError, ModuleNotFoundError) as _e:
    HAVE = False
    _ERR = repr(_e)

AMP, F0, FDOT, FDDOT = 1e-23, 4.2300812341e-3, 1e-18, 0.0
PHI0, INC, PSI, LAM, BETA = 0.892342342342, 1.2309804223, 3.00908098, 4.827342308, -0.50923423

_FIXTURE = None


def _build_fixture():
    """Inject a real GB, build its true STFT as the data + XYZ noise (rectangular
    window). Cached module-wide so the heavy 64-day STFT setup builds once."""
    global _FIXTURE
    if _FIXTURE is not None:
        return _FIXTURE
    fb = "cpu"; xp = np
    orbits = DefaultOrbits(force_backend=fb); orbits.configure(linear_interp_setup=True)
    tdi_config = TDIConfig("2nd generation", force_backend=fb)
    dt = 10.0; stft_dt = 6 * 3600.0; n_stft = 256
    nperseg = int(stft_dt / dt); nobs = n_stft * nperseg; Tobs = nobs * dt
    N_sparse = 512
    t_tdi = xp.linspace(0.0, Tobs, N_sparse + 1)[1:-1]
    data_t = xp.arange(nobs) * dt
    df_grid = 1.0 / stft_dt
    settings = get_stft_settings(data_t, stft_dt, min_freq=70 * df_grid,
                                 max_freq=115 * df_grid, force_backend=fb)
    gb_gen = GBTDIonTheFly(t_tdi, Tobs, 0.0, 1.0 / dt, 1, tdi_config=tdi_config,
                           orbits=orbits, tdi_chan="XYZ", force_backend=fb)
    out = gb_gen(AMP, F0, FDOT, FDDOT, PHI0, INC, PSI, LAM, BETA,
                 convert_to_ra_dec=False, return_spline=True)
    keep = (data_t > t_tdi[0]) & (data_t < t_tdi[-1])
    tdi_output = xp.zeros((1, 3, nobs)); tdi_output[:, :, keep] = out.eval_tdi(data_t[keep])
    td_signal = TDSignal(tdi_output[0], settings=TDSettings(nobs, dt, 0.0, force_backend=fb))
    stft_signal = td_signal.stft(window=xp.ones(nperseg), settings=settings)
    data_res = DataResidualArray(stft_signal)
    sens = XYZSensitivityBackend(orbits=orbits, settings=settings, force_backend=fb)
    sens.sens_mat = sens.compute_sensitivity_matrix(sens.basis_settings.f_arr, 15e-12, 3e-15)
    ac = AnalysisContainer(data_res, sens)
    acs = AnalysisContainerArray([ac], gpus=None)
    grp = STFTComputationGroup(acs, split_index=0, window_alpha=0.0, force_backend=fb)
    grp.compute_d_d_term()
    d_d = float(np.asarray(grp.d_d).reshape(-1)[0].real)
    params = np.array([[AMP, F0, FDOT, FDDOT, PHI0, INC, PSI, LAM, BETA]])
    _FIXTURE = dict(orbits=orbits, tdi_config=tdi_config, grp=grp, d_d=d_d,
                    Tobs=Tobs, params=params)
    return _FIXTURE


def _make_gb(fx, n_side_bins):
    return STFTGBComputations(
        stft_comps=fx["grp"], T=fx["Tobs"], t_ref=0.0,
        orbits=fx["orbits"], tdi_config=fx["tdi_config"],
        force_backend="cpu", n_side_bins=n_side_bins, window_factor=1.0,
        freq_from_tdi_phase=True)


@unittest.skipUnless(HAVE, "requires GBGPU STFT-GB build + LAT stack")
class STFTGBFFTSmokeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fx = _build_fixture()

    def test_fft_get_ll_is_finite_and_shaped(self):
        gb = _make_gb(self.fx, n_side_bins=5)
        ll = gb.get_ll_stft_fft(self.fx["params"], n_sub=32)
        self.assertEqual(np.asarray(ll).shape, (1,))
        d_h = complex(np.asarray(gb.d_h_out_fft).reshape(-1)[0])
        h_h = complex(np.asarray(gb.h_h_out_fft).reshape(-1)[0])
        self.assertTrue(np.isfinite(d_h.real) and np.isfinite(d_h.imag))
        self.assertTrue(np.isfinite(h_h.real) and h_h.real > 0.0)


@unittest.skipUnless(HAVE, "requires GBGPU STFT-GB build + LAT stack")
class STFTGBFFTAccuracyTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fx = _build_fixture()

    def test_fft_at_least_as_accurate_as_fresnel(self):
        """FFT and analytic Fresnel recover the SAME injection to comparable
        accuracy; the FFT (exact midpoint quadrature) is at least as good vs the
        ground-truth brute STFT. Comparing FFT to Fresnel *directly* floors at
        Fresnel's own ~4e-3 error (rectangular-window leakage + Fresnel-integral
        approximation), so the injection is the oracle -- design 2026-07-01."""
        gb = _make_gb(self.fx, n_side_bins=20)
        gb.get_ll_stft(self.fx["params"])                        # analytic Fresnel
        d_h_f = complex(np.asarray(gb.d_h_out).reshape(-1)[0])
        h_h_f = float(np.asarray(gb.h_h_out).reshape(-1)[0].real)
        mm_f = abs(1.0 - d_h_f.real / np.sqrt(self.fx["d_d"] * h_h_f))
        gb.get_ll_stft_fft(self.fx["params"], n_sub=64)          # FFT candidate
        d_h_x = complex(np.asarray(gb.d_h_out_fft).reshape(-1)[0])
        h_h_x = float(np.asarray(gb.h_h_out_fft).reshape(-1)[0].real)
        mm_x = abs(1.0 - d_h_x.real / np.sqrt(self.fx["d_d"] * h_h_x))
        rel_dh = abs(d_h_x - d_h_f) / abs(d_h_f)
        print(f"\n[fft-vs-fresnel] fresnel mm={mm_f:.3e}  fft mm={mm_x:.3e}  "
              f"dh rel={rel_dh:.3e}")
        self.assertLess(mm_x, 1e-2)                       # FFT recovers injection
        self.assertLessEqual(mm_x, mm_f * 1.15 + 1e-4)    # >= Fresnel accuracy
        self.assertLess(rel_dh, 1.5e-2)                   # agree within combined error

    def test_fft_recovers_injection(self):
        """FFT get_ll recovers the injected GB (normalized mismatch < 1%),
        mirroring test_stft_gb_accuracy.py::test_recovers_injection."""
        gb = _make_gb(self.fx, n_side_bins=20)
        gb.get_ll_stft_fft(self.fx["params"], n_sub=64)
        d_h = complex(np.asarray(gb.d_h_out_fft).reshape(-1)[0])
        h_h = float(np.asarray(gb.h_h_out_fft).reshape(-1)[0].real)
        mm = 1.0 - d_h.real / np.sqrt(self.fx["d_d"] * h_h)
        print(f"[fft-recovery] d_h.re={d_h.real:.6e} h_h={h_h:.6e} mismatch={mm:.3e}")
        self.assertLess(abs(mm), 1e-2)
        self.assertAlmostEqual(h_h / self.fx["d_d"], 1.0, delta=0.05)


@unittest.skipUnless(HAVE, "requires GBGPU STFT-GB build + LAT stack")
class STFTGBFFTConvergenceCostTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fx = _build_fixture()

    def test_n_sub_convergence(self):
        """FFT recovery mismatch vs the ground-truth brute STFT shrinks as n_sub
        grows (the accuracy knob). Measured vs the injection, NOT vs Fresnel (which
        has its own ~4e-3 floor the FFT converges past) -- design 2026-07-01."""
        gb = _make_gb(self.fx, n_side_bins=20)
        mms = []
        for n_sub in (8, 16, 32, 64):
            gb.get_ll_stft_fft(self.fx["params"], n_sub=n_sub)
            d_h = complex(np.asarray(gb.d_h_out_fft).reshape(-1)[0])
            h_h = float(np.asarray(gb.h_h_out_fft).reshape(-1)[0].real)
            mms.append(abs(1.0 - d_h.real / np.sqrt(self.fx["d_d"] * h_h)))
            print(f"[n_sub] n_sub={n_sub:3d} recovery_mismatch={mms[-1]:.3e}")
        # The targeted DFT needs n_sub >~ 2*n_side+1 to resolve the requested band
        # (here 41 bins): below that the far bins alias (n_sub=8,16 -> O(0.5)), above
        # it midpoint quadrature converges 2nd-order. Assert improvement in that
        # resolved regime (finest beats second-finest) + a well-converged floor.
        self.assertLess(mms[-1], mms[-2])      # 64 improves on 32 (resolved regime)
        self.assertLess(mms[-1], 6e-3)         # converged near Fresnel accuracy

    def test_cpu_cost_proxy(self):
        """Relative CPU wall-clock, Fresnel vs FFT at fixed workload. Informational
        (the GPU box is the real verdict) but must complete and report a ratio."""
        params = np.repeat(self.fx["params"], 64, axis=0)   # 64 binaries
        gb = _make_gb(self.fx, n_side_bins=10)
        gb.get_ll_stft(params)                               # warm up
        t0 = time.perf_counter(); gb.get_ll_stft(params); t_fres = time.perf_counter() - t0
        gb.get_ll_stft_fft(params, n_sub=32)                 # warm up
        t0 = time.perf_counter(); gb.get_ll_stft_fft(params, n_sub=32); t_fft = time.perf_counter() - t0
        print(f"\n[cost] num_bin=64  fresnel={t_fres*1e3:.1f} ms  fft(n_sub=32)={t_fft*1e3:.1f} ms  "
              f"ratio(fft/fresnel)={t_fft/max(t_fres,1e-9):.2f}")
        self.assertGreater(t_fft, 0.0)


if __name__ == "__main__":
    unittest.main()
