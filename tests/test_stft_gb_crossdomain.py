"""Cross-domain optimal-SNR^2 consistency for the STFT/Fresnel GB likelihood.

``test_stft_gb_accuracy.py`` checks the Fresnel ``get_ll_stft`` against the
BRUTE-FORCE STFT (true DFT-per-segment). This test closes the loop against the
other two production time-frequency representations -- the FREQUENCY DOMAIN (FD)
and the WAVELET DOMAIN (WDM) -- by injecting ONE galactic binary and confirming
the optimal SNR^2 = (h|h) of that same physical source is recovered consistently
in all three domains.

Why (h|h): it is a domain-INDEPENDENT physical number (the band integral of
|h|^2 / S_n). Agreement across FD / STFT / WDM exercises every normalization
convention that differs between the domains -- the factor of 4, df, the STFT
``diff_comp`` / window factor, and the WDM wavelet normalization + PSD fold -- so
a convention bug in any single path surfaces as a cross-domain SNR^2 mismatch.

Sensitivity handling (the key subtlety): no single sensitivity construction
spans all three domains in the maintained code, so **FD is the common anchor**:

  * STFT vs FD uses the orbit-based ``XYZSensitivityBackend`` -- the only
    sensitivity that supports an STFT basis (``XYZ2SensitivityMatrix`` raises
    ``NotImplementedError`` for ``STFTSettings``). Identical PSD in both legs.
  * WDM vs FD uses analytic ``XYZ2SensitivityMatrix(model='scirdv1')`` -- the
    only path that folds the PSD into the wavelet basis. The orbit-based backend
    evaluates at layer centres WITHOUT the fold and is ~45% wrong for WDM, so it
    must not be used there.
  * The two FD anchors differ ONLY by the PSD model (~1.4% at the carrier), so
    STFT and WDM are each compared against the FD reference built with the SAME
    PSD as that leg -- the comparison is never contaminated by a PSD mismatch.

Measured agreement (on-bin 8 mHz carrier, ~0.5 yr, rectangular window, carrier
interior to the orbit):

    STFT-domain / FD (orbit)      ~ 1.0 %
    WDM-domain  / FD (scirdv1)    ~ 1.5 %
    Fresnel kernel (h|h) / FD     ~ 0.6 %
    Fresnel self-recovery mismatch ~ 6e-5

Residuals are time-frequency tiling + rectangular-window spectral leakage -- the
same error budget decomposed in ``test_stft_gb_accuracy.py`` -- NOT a modeling
error. Tolerances below carry a comfortable margin over the measured values.
"""

import unittest

import numpy as np

try:
    from lisatools.detector import ESAOrbits
    from lisatools.response.tdiconfig import TDIConfig
    from lisatools.response.tdionfly import GBTDIonTheFly
    from lisatools.domains import (TDSettings, TDSignal, FDSettings, WDMSettings,
                                   get_stft_settings)
    from lisatools.domaincomputation import STFTComputationGroup
    from lisatools.datacontainer import DataResidualArray
    from lisatools.sensitivity import XYZSensitivityBackend, XYZ2SensitivityMatrix
    from lisatools.analysiscontainer import AnalysisContainer, AnalysisContainerArray
    from lisatools.diagnostic import inner_product
    from lisatools.utils.constants import YRSID_SI

    from gbgpu.gbcomps import STFTGBComputations

    HAVE_XDOMAIN = True
except (ImportError, ModuleNotFoundError) as _e:
    HAVE_XDOMAIN = False
    _IMPORT_ERR = repr(_e)


# Single GB injection: verification-binary-style source, carrier ON an STFT bin.
AMP, FDOT, FDDOT = 1.0e-22, 1.0e-17, 0.0
PHI0, INC, PSI, LAM, BETA = 1.4, np.pi / 3.0, 0.7, 2.1, 0.5

# orbit-based instrument noise knobs (OMS displacement / acceleration).
OMS, ACC = 15e-12, 3e-15

# Cross-domain SNR^2 ratios agree to ~1-1.5%; allow a generous margin so the
# test is robust to grid / orbit / FP variation without being meaningless.
CROSS_TOL = 0.03          # FD<->STFT (same PSD) and FD<->WDM (same PSD)
CROSS_TOL_PSDMIX = 0.05   # Fresnel(orbit) <-> WDM(scirdv1): + ~1.4% PSD-model gap
MM_TOL = 1.0e-3           # Fresnel matched-filter self-recovery (measured ~6e-5)


@unittest.skipUnless(HAVE_XDOMAIN,
                     "requires GBGPU STFT-GB build + LAT response/sensitivity/domaincomputation")
class STFTGBCrossDomainTest(unittest.TestCase):
    """Inject one GB; recover its optimal SNR^2 in FD, STFT, and WDM."""

    @classmethod
    def setUpClass(cls):
        fb = "cpu"
        dt = 15.0
        Nobs = 2 ** 20                       # 1,048,576 = NT*nperseg = Nf*Nt
        Tobs = Nobs * dt
        t_start = int(0.5 * YRSID_SI / dt) * dt    # carrier interior to the orbit

        orbits = ESAOrbits(); orbits.configure(linear_interp_setup=True)
        tdi_config = TDIConfig("2nd generation")

        nperseg = 4096
        stft_dt = nperseg * dt
        df_stft = 1.0 / stft_dt
        f0_idx = int(round(8.0e-3 / df_stft))
        f0 = f0_idx * df_stft                # ON a STFT bin centre (no leakage)
        params = np.array([[AMP, f0, FDOT, FDDOT, PHI0, INC, PSI, LAM, BETA]])

        # --- inject the GB TDI time series once (full grid, no edge padding) ---
        t_tdi = np.linspace(t_start, t_start + Tobs, 16384)
        gb_gen = GBTDIonTheFly(t_tdi, Tobs, t_start, 1.0 / dt, 1,
                               tdi_config=tdi_config, orbits=orbits,
                               tdi_chan="XYZ", force_backend=fb)
        spline = gb_gen(*[np.array([p]) for p in params[0]],
                        convert_to_ra_dec=False, return_spline=True)
        t_arr = np.arange(Nobs) * dt + t_start
        td_inj = np.asarray(spline.eval_tdi(t_arr))[0]      # (3, Nobs)
        td_set = TDSettings(N=Nobs, dt=dt)

        # --- domain settings (all bracket the 8 mHz carrier) ---
        fd_settings = FDSettings(N=Nobs // 2 + 1, df=1.0 / Tobs,
                                 min_freq=7.0e-3, max_freq=9.0e-3, force_backend=fb)
        stft_settings = get_stft_settings(t_arr, stft_dt,
                                          min_freq=(f0_idx - 30) * df_stft,
                                          max_freq=(f0_idx + 30) * df_stft,
                                          force_backend=fb)
        Nf = Nt = 1024
        EC = 8                               # edge-cut wavelet columns
        wdm_settings = WDMSettings(Nf, Nt, dt, t0=t_start,
                                   min_freq=7.0e-3, max_freq=9.0e-3,
                                   min_time=EC * Nf * dt, max_time=(Nt - EC) * Nf * dt,
                                   force_backend=fb)

        # --- transform the SAME injection into each domain ---
        fd_sig = TDSignal(td_inj, settings=td_set).transform(fd_settings)
        stft_sig = TDSignal(td_inj, settings=td_set).stft(
            settings=stft_settings, window=np.ones(nperseg))
        wdm_sig = TDSignal(td_inj, settings=td_set).transform(wdm_settings)

        def orbit_sens(settings):
            s = XYZSensitivityBackend(orbits=orbits, settings=settings, force_backend=fb)
            s.sens_mat = s.compute_sensitivity_matrix(s.basis_settings.f_arr, OMS, ACC)
            return s

        def snr2(sig, settings, sens):
            return float(np.real(inner_product(sig, sig, basis_settings=settings, psd=sens)))

        # FD anchor under BOTH PSDs (orbit for the STFT leg, scirdv1 for the WDM leg)
        cls.sens_stft = orbit_sens(stft_settings)
        cls.fd_orbit = snr2(fd_sig, fd_settings, orbit_sens(fd_settings))
        cls.fd_scird = snr2(fd_sig, fd_settings,
                            XYZ2SensitivityMatrix(fd_settings, model="scirdv1"))
        cls.stft_orbit = snr2(stft_sig, stft_settings, cls.sens_stft)
        cls.wdm_scird = snr2(wdm_sig, wdm_settings,
                             XYZ2SensitivityMatrix(wdm_settings, model="scirdv1"))

        # --- STFT/Fresnel GB kernel (the implementation under test) ---
        data_res = DataResidualArray(stft_sig)
        ac = AnalysisContainer(data_res, cls.sens_stft)
        acs = AnalysisContainerArray([ac], gpus=None)
        grp = STFTComputationGroup(acs, split_index=0, window_alpha=0.0, force_backend=fb)
        grp.compute_d_d_term()
        cls.d_d_stft = float(np.asarray(grp.d_d).reshape(-1)[0].real)
        gb = STFTGBComputations(stft_comps=grp, T=Tobs, t_ref=t_start, orbits=orbits,
                                tdi_config=tdi_config, force_backend=fb, n_side_bins=20,
                                window_factor=1.0, freq_from_tdi_phase=True)
        gb.get_ll_stft(params)
        cls.d_h_stft = complex(np.asarray(gb.d_h_out).reshape(-1)[0])
        cls.h_h_stft = float(np.asarray(gb.h_h_out).reshape(-1)[0].real)

        print(f"\n[xdomain] FD orbit={cls.fd_orbit:.5e} scirdv1={cls.fd_scird:.5e} "
              f"(PSD ratio {cls.fd_scird/cls.fd_orbit:.4f})")
        print(f"[xdomain] STFT={cls.stft_orbit:.5e} ({100*(cls.stft_orbit/cls.fd_orbit-1):+.2f}% vs FD) "
              f"WDM={cls.wdm_scird:.5e} ({100*(cls.wdm_scird/cls.fd_scird-1):+.2f}% vs FD)")
        print(f"[xdomain] Fresnel (h|h)={cls.h_h_stft:.5e} d_d={cls.d_d_stft:.5e} "
              f"(d|h).re={cls.d_h_stft.real:.5e}")

    # ---- ground-truth domain SNR^2 cross-checks (FD common anchor) ----

    def test_stft_domain_snr2_matches_fd(self):
        """STFT-domain optimal SNR^2 == FD optimal SNR^2 (same orbit PSD)."""
        ratio = self.stft_orbit / self.fd_orbit
        print(f"[xdomain] STFT/FD (orbit PSD) = {ratio:.5f}")
        self.assertLess(abs(1.0 - ratio), CROSS_TOL)

    def test_wdm_domain_snr2_matches_fd(self):
        """WDM-domain optimal SNR^2 == FD optimal SNR^2 (same scirdv1 folded PSD)."""
        ratio = self.wdm_scird / self.fd_scird
        print(f"[xdomain] WDM/FD (scirdv1 PSD) = {ratio:.5f}")
        self.assertLess(abs(1.0 - ratio), CROSS_TOL)

    def test_fd_anchor_consistent_across_psd_models(self):
        """The two FD anchors differ only by the PSD model (small, bounded)."""
        ratio = self.fd_scird / self.fd_orbit
        print(f"[xdomain] FD scirdv1/orbit (PSD-model gap) = {ratio:.5f}")
        self.assertLess(abs(1.0 - ratio), CROSS_TOL)   # ~1.4% measured

    # ---- the Fresnel GB kernel sits at the cross-domain SNR^2 ----

    def test_fresnel_kernel_snr2_matches_fd(self):
        """Fresnel get_ll_stft (h|h) reproduces the FD likelihood SNR^2 (orbit PSD)."""
        ratio = self.h_h_stft / self.fd_orbit
        print(f"[xdomain] Fresnel (h|h)/FD = {ratio:.5f}")
        self.assertLess(abs(1.0 - ratio), CROSS_TOL)

    def test_fresnel_kernel_snr2_matches_wdm(self):
        """Fresnel get_ll_stft (h|h) reproduces the WDM likelihood SNR^2.

        Cross-PSD comparison (Fresnel uses the orbit PSD, WDM the scirdv1 folded
        PSD), so the budget includes the ~1.4% PSD-model gap on top of tiling.
        """
        ratio = self.h_h_stft / self.wdm_scird
        print(f"[xdomain] Fresnel (h|h)/WDM = {ratio:.5f}")
        self.assertLess(abs(1.0 - ratio), CROSS_TOL_PSDMIX)

    def test_fresnel_kernel_self_recovery(self):
        """Matched-filter self-recovery: (h|h)~(d|h)~d_d and mismatch -> 0.

        Also pins the STFT/Fresnel kernel's (h|h) to the independent
        STFTComputationGroup d_d, which itself matches the inner_product STFT
        SNR^2 -- so the kernel, the STFT domain, and FD/WDM all tie together.
        """
        mm = 1.0 - self.d_h_stft.real / np.sqrt(self.d_d_stft * self.h_h_stft)
        print(f"[xdomain] Fresnel (h|h)/d_d = {self.h_h_stft/self.d_d_stft:.5f}  "
              f"mismatch = {mm:+.3e}")
        self.assertLess(abs(mm), MM_TOL)
        self.assertAlmostEqual(self.h_h_stft / self.d_d_stft, 1.0, delta=0.02)
        self.assertAlmostEqual(self.d_h_stft.real / self.d_d_stft, 1.0, delta=0.02)


if __name__ == "__main__":
    unittest.main()
