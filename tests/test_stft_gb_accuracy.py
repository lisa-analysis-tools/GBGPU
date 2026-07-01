"""End-to-end cross-domain accuracy test for the STFT/Fresnel GB likelihood.

`test_stft_gb.py` only checks *intra-STFT* self-consistency (on-the-fly get_ll ==
fill_global -> domain inner product) on synthetic data -- which cannot catch an
error common to both paths (a wrong overall convention, f0/fdot0, or window
factor), since both share the same Fresnel kernel.

This test is the independent check (the brute-force reference recipe from
lisa-on-gpu/dev_stft.py): inject a *real* GB TDI time series, build its actual
STFT as the data with the real XYZ LISA noise PSD, and verify that the on-the-fly
Fresnel `get_ll_stft` recovers the injection (normalized mismatch -> 0) and
converges as `n_side_bins` grows. It also checks the Doppler f0/fdot0 fix
(`freq_from_tdi_phase=True`) does not do worse than the astrophysical-only path.

Rectangular window (window_alpha=0) is used so the comparison is purely the
Fresnel approximation vs the true DFT-per-segment -- no windowing convention in
the way.

Note on the residual mismatch (~4e-3 at n_side_bins=20 here): it is NOT a Fresnel
modeling error. Decomposed, it is (1) rectangular-window spectral leakage from the
off-bin carrier (shrinks with n_side_bins / a real window) and (2) the two edge
time-segments, where `freq_from_tdi_phase` finite-differences the TDI phase at
t +- dt and at the first/last bins samples outside [0, T] (spline orbits then
extrapolate). The companion STFTGBInteriorAccuracyTest isolates the intrinsic
accuracy (on-bin carrier, interior segments) and gets ~2e-5 -- the model is
essentially exact for a GB. The edge contribution dilutes as 1/NT for longer
observations.
"""

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
    from lisatools.utils.parallelbase import LISAToolsParallelModule

    from gbgpu.gbcomps import STFTGBComputations

    class _BackendHolder(LISAToolsParallelModule):
        """Minimal concrete module used to resolve the CPU backend object."""

    HAVE_E2E = True
except (ImportError, ModuleNotFoundError) as _e:
    HAVE_E2E = False
    _IMPORT_ERR = repr(_e)


# Injection params (single GB, carrier ~4.23 mHz) -- the dev_stft.py source.
AMP, F0, FDOT, FDDOT = 1e-23, 4.2300812341e-3, 1e-18, 0.0
PHI0, INC, PSI, LAM, BETA = 0.892342342342, 1.2309804223, 3.00908098, 4.827342308, -0.50923423


@unittest.skipUnless(HAVE_E2E, "requires GBGPU STFT-GB build + LAT response/sensitivity/domaincomputation")
class STFTGBEndToEndTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        fb = "cpu"
        xp = np
        cls.orbits = DefaultOrbits(force_backend=fb)
        cls.orbits.configure(linear_interp_setup=True)
        cls.tdi_config = TDIConfig("2nd generation", force_backend=fb)
        cls.t_ref = 0.0
        dt = 10.0
        stft_dt = 6 * 3600.0
        n_stft = 256                              # 256 six-hour segments ~ 64 days
        nperseg = int(stft_dt / dt)
        nobs = n_stft * nperseg
        cls.Tobs = nobs * dt
        N_sparse = 512                            # spline knots for the injected TDI
        t_tdi = xp.linspace(0.0, cls.Tobs, N_sparse + 1)[1:-1]

        data_t = xp.arange(nobs) * dt
        # IMPORTANT: min_freq / max_freq must be exact integer multiples of
        # df = 1/stft_dt. The real STFT lives on the FFT grid (j*df from 0), but
        # STFTDomain labels its active bins as f_min + k*df and get_freq_index
        # rounds (f0 - f_min)/df. If f_min is off-grid (e.g. 3.5e-3, = 75.6*df),
        # the domain's frequency origin is offset by a fraction of a bin, the
        # carrier mis-indexes by one bin, and the overlap collapses (mismatch
        # ~0.76 with norms still matching). get_stft_settings does NOT snap
        # min_freq itself, so the caller must. Band ~[3.24e-3, 5.32e-3] around
        # the f0~4.23 mHz carrier (full bin 91), with headroom for the sweep.
        df_grid = 1.0 / stft_dt
        cls.settings = get_stft_settings(
            data_t, stft_dt, min_freq=70 * df_grid, max_freq=115 * df_grid,
            force_backend=fb,
        )

        # --- inject the real GB TDI time series (spline-evaluated) ---
        # GBTDIonTheFly(t, T, t_ref, *args, **kwargs) forwards *args to the base
        # TDIonTheFly(sampling_frequency, num_sub, ...) and injects n_params=9
        # itself -- so pass sampling_frequency (1/dt) and num_sub positionally,
        # and do NOT pass n_params (the LAT ctor sets it; passing collides).
        gb_gen = GBTDIonTheFly(
            t_tdi, cls.Tobs, cls.t_ref, 1.0 / dt, 1,
            tdi_config=cls.tdi_config, orbits=cls.orbits, tdi_chan="XYZ",
            force_backend=fb,
        )
        out = gb_gen(AMP, F0, FDOT, FDDOT, PHI0, INC, PSI, LAM, BETA,
                     convert_to_ra_dec=False, return_spline=True)
        keep = (data_t > t_tdi[0]) & (data_t < t_tdi[-1])
        tdi_output = xp.zeros((1, 3, nobs))
        tdi_output[:, :, keep] = out.eval_tdi(data_t[keep])

        # --- real STFT of the signal = the data (rectangular window) ---
        # TDSettings signature is (N, dt, t0=0.0).
        td_signal = TDSignal(
            tdi_output[0], settings=TDSettings(nobs, dt, cls.t_ref, force_backend=fb)
        )
        window = xp.ones(nperseg, dtype=xp.float64)         # rectangular
        cls.window_factor = 1.0
        stft_signal = td_signal.stft(window=window, settings=cls.settings)

        data_res = DataResidualArray(stft_signal)
        sens = XYZSensitivityBackend(orbits=cls.orbits, settings=cls.settings, force_backend=fb)
        sens.sens_mat = sens.compute_sensitivity_matrix(
            sens.basis_settings.f_arr, 15e-12, 3e-15
        )

        ac = AnalysisContainer(data_res, sens)
        acs = AnalysisContainerArray([ac], gpus=None)
        cls.stft_group = STFTComputationGroup(
            acs, split_index=0, window_alpha=0.0, force_backend=fb
        )
        cls.stft_group.compute_d_d_term()
        cls.d_d = float(np.asarray(cls.stft_group.d_d).reshape(-1)[0].real)
        cls.snr_opt = float(np.sqrt(cls.d_d))
        print(f"\n[e2e] grid NT={cls.settings.NT} NF_active={cls.settings.NF_active} "
              f"d_d={cls.d_d:.6e} optimal SNR={cls.snr_opt:.3f}")

        cls.params = np.array([[AMP, F0, FDOT, FDDOT, PHI0, INC, PSI, LAM, BETA]])

    def _gb(self, n_side_bins, freq_from_tdi_phase=True):
        return STFTGBComputations(
            stft_comps=self.stft_group, T=self.Tobs, t_ref=self.t_ref,
            orbits=self.orbits, tdi_config=self.tdi_config, force_backend="cpu",
            n_side_bins=n_side_bins, window_factor=self.window_factor,
            freq_from_tdi_phase=freq_from_tdi_phase,
        )

    def _mismatch(self, n_side_bins, freq_from_tdi_phase=True):
        gb = self._gb(n_side_bins, freq_from_tdi_phase)
        gb.get_ll_stft(self.params)
        d_h = complex(np.asarray(gb.d_h_out).reshape(-1)[0])
        h_h = float(np.asarray(gb.h_h_out).reshape(-1)[0].real)
        overlap = d_h.real / np.sqrt(self.d_d * h_h)
        return 1.0 - overlap, d_h, h_h

    def test_recovers_injection(self):
        mm, d_h, h_h = self._mismatch(n_side_bins=20)
        print(f"[e2e] recovery: d_h.re={d_h.real:.6e} h_h={h_h:.6e} mismatch={mm:.3e}")
        # On a df-aligned grid the on-the-fly Fresnel template matches the true
        # STFT of the injected signal to well within 1% once the side-band is
        # wide enough (n_side_bins=20 -> ~4e-3); also d_h, h_h, d_d all agree.
        self.assertLess(abs(mm), 1e-2)
        self.assertAlmostEqual(h_h / self.d_d, 1.0, delta=0.05)
        self.assertAlmostEqual(d_h.real / self.d_d, 1.0, delta=0.05)

    def test_n_side_bins_convergence(self):
        ns = (1, 2, 3, 5, 10, 20)
        mms = []
        for n in ns:
            mm, _, _ = self._mismatch(n_side_bins=n)
            mms.append(abs(mm))
            print(f"[e2e] n_side_bins={n:3d} mismatch={mm:.3e}")
        # Capturing more of the Fresnel side-band must monotonically improve the
        # match (each step <= the previous, up to FP noise) and converge well
        # below the n_side_bins=1 value -- the signature of a correct kernel.
        for prev, cur in zip(mms, mms[1:]):
            self.assertLessEqual(cur, prev + 1e-9)
        self.assertLess(mms[-1], 0.25 * mms[0])

    def test_doppler_fix_not_worse(self):
        mm_corr, _, _ = self._mismatch(10, freq_from_tdi_phase=True)
        mm_astro, _, _ = self._mismatch(10, freq_from_tdi_phase=False)
        print(f"[e2e] mismatch corrected={mm_corr:.3e}  astro-only={mm_astro:.3e}")
        self.assertLessEqual(abs(mm_corr), abs(mm_astro) * (1 + 1e-6) + 1e-12)


@unittest.skipUnless(HAVE_E2E, "requires GBGPU STFT-GB build + LAT response/sensitivity/domaincomputation")
class STFTGBInteriorAccuracyTest(unittest.TestCase):
    """Isolates the *intrinsic* per-segment Fresnel accuracy for a GB from the two
    confounders that inflate the naive end-to-end mismatch:

      * spectral leakage (off-bin carrier + boxcar) -> removed by injecting the
        carrier exactly on a bin center;
      * edge segments -> `freq_from_tdi_phase` finite-differences the TDI phase at
        t +- dt, which at the first/last STFT bins samples outside [0, T]; with
        spline orbits that extrapolates and corrupts those two bins (it is the
        edge caveat noted in the port's handoff).

    With both removed, the on-the-fly Fresnel template reproduces the true STFT of
    the GB to ~1e-4 (phase-maximized grid overlap) -- i.e. the model is essentially
    exact for a binary with negligible per-segment chirp, as physically expected.
    A GB's mismatch is therefore dominated by windowing/leakage and the two edge
    segments, NOT by the Fresnel approximation itself.
    """

    @classmethod
    def setUpClass(cls):
        fb = "cpu"
        cls.backend = _BackendHolder(force_backend=fb).backend
        cls.tdi_type = cls.backend.TDITypeDict["XYZ"]
        cls.orbits = DefaultOrbits(force_backend=fb)
        cls.orbits.configure(linear_interp_setup=True)
        cls.tdi_config = TDIConfig("2nd generation", force_backend=fb)
        dt = 10.0
        stft_dt = 6 * 3600.0
        cls.df = 1.0 / stft_dt
        nperseg = int(stft_dt / dt)
        n_stft = 96
        nobs = n_stft * nperseg
        cls.Tobs = nobs * dt
        # Start the observation 10 days INTO the (multi-year) orbit so the first
        # STFT segment is not at the orbit's t0 boundary -- the realistic case
        # (orbit files span more than the analysis segment). With the observation
        # pinned at t0=0 the first segment's TDI needs the orbit at t<0
        # (extrapolation), which alone inflates the full-grid mismatch ~60x
        # (6e-3 vs ~1e-4); the interior is unaffected either way.
        t_start = 10.0 * 86400.0
        data_t = t_start + np.arange(nobs) * dt
        cls.f0 = round(F0 / cls.df) * cls.df          # carrier ON a bin center
        cls.settings = get_stft_settings(
            data_t, stft_dt, min_freq=1 * cls.df, max_freq=185 * cls.df, force_backend=fb
        )
        # inject + real STFT (rectangular). Span the spline knots over the full
        # [0, Tobs] and evaluate every sample (no edge zero-padding): this test
        # targets the freq_from_tdi_phase edge stencil, so the *data* must be a
        # clean signal at the first/last bins too.
        t_tdi = np.linspace(t_start, t_start + cls.Tobs, 512)
        gen = GBTDIonTheFly(t_tdi, cls.Tobs, 0.0, 1.0 / dt, 1, tdi_config=cls.tdi_config,
                            orbits=cls.orbits, tdi_chan="XYZ", force_backend=fb)
        out = gen(AMP, cls.f0, FDOT, FDDOT, PHI0, INC, PSI, LAM, BETA,
                  convert_to_ra_dec=False, return_spline=True)
        tdi = out.eval_tdi(data_t)          # (1, 3, nobs), no edge zero-padding
        td = TDSignal(tdi[0], settings=TDSettings(nobs, dt, t_start, force_backend=fb))
        cls.D = np.asarray(td.stft(window=np.ones(nperseg), settings=cls.settings).arr)

        NT, NF = cls.settings.NT, cls.settings.NF_active
        zdata = np.zeros((1, 3, NT, NF), np.complex128)
        zinvC = np.zeros((1, 3, 3, NT, NF), np.complex128)
        s = cls.settings
        domain = cls.backend.STFTDomainWrap(NT, NF, 3, s.t0, s.min_freq, s.max_freq, s.dt, s.df,
                                            zdata.reshape(-1), zinvC.reshape(-1), 1, 1, cls.tdi_type)
        fres = cls.backend.STFTFresnelWrap(NT, NF, 3, s.t0, s.min_freq, s.max_freq, s.dt, s.df,
                                           window_alpha=0.0, use_midpoint=False)
        shim = type("S", (), {})()
        shim.cpp_fresnel, shim.cpp_domain, shim.d_d = fres, domain, None
        shim._keepalive = (zdata, zinvC)
        cls.shim = shim

    def _model_grid(self, n_side):
        gb = STFTGBComputations(stft_comps=self.shim, T=self.Tobs, t_ref=0.0, orbits=self.orbits,
                                tdi_config=self.tdi_config, force_backend="cpu", n_side_bins=n_side,
                                window_factor=1.0, freq_from_tdi_phase=True)
        M = np.zeros((1, 3, self.settings.NT, self.settings.NF_active), np.complex128)
        gb.fill_global_stft(np.array([[AMP, self.f0, FDOT, FDDOT, PHI0, INC, PSI, LAM, BETA]]),
                            M, data_index=np.array([0], dtype=np.int32))
        return M[0]

    @staticmethod
    def _mm(D, M):
        return 1.0 - np.abs(np.sum(np.conj(D) * M)) / (np.linalg.norm(D) * np.linalg.norm(M))

    def test_fresnel_accuracy_full_grid(self):
        """On-bin GB with the observation INTERIOR to the orbit (production case):
        the on-the-fly Fresnel template matches the true STFT to ~1e-4 over the
        WHOLE grid, first/last segments included.

        The edge segments are accurate only when the orbit extends beyond the
        observation. Pinning the observation at the orbit's t0=0 makes the first
        segment's TDI extrapolate the orbit (the helper samples t-dt < t0) and
        inflates the full-grid mismatch ~60x (6e-3 vs ~1e-4) -- a setup artifact,
        not a model error (verified by shifting the observation 10 days in).
        Interior accuracy is ~2-6e-5 regardless."""
        M = self._model_grid(n_side=10)
        full = self._mm(self.D, M)
        interior = self._mm(self.D[:, 1:-1, :], M[:, 1:-1, :])
        edge = self._mm(self.D[:, [0, -1], :], M[:, [0, -1], :])
        print(f"\n[accuracy] on-bin mismatch: full={full:.3e}  interior={interior:.3e}  edges-only={edge:.3e}")
        self.assertLess(full, 1e-3)
        self.assertLess(interior, 1e-3)


if __name__ == "__main__":
    unittest.main()
