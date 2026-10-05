"""Sig-het v5 / F-stat at the ACTIVE-BAND EDGE: out-of-band m rows are skipped, not clamped.

The scorers form a candidate's m band as m_floor - m_half .. m_floor + m_half. Rows
below ind_min_f (above ind_max_f) do not exist; they used to be CLAMPED onto the edge
row, which then entered the folds 2-3 times for every source within m_half layers of a
band edge (first cluster gate: sig-het delta errors of 1-3 lnL at 0.3 mHz on the
production 2.5e-4 Hz band floor, all inclinations). Pinned here against the exact
chunked scorer for a source whose carrier sits 0.3 layers above ind_min_f: <h|h> at
the reference with zero data (the first-order metric) and the ln L of a displaced
candidate with data = the source. CPU.
"""

import os
import unittest

import numpy as np

from lisatools.detector import ESAOrbits
from lisatools.domains import WDMSettings
from lisatools.utils.constants import YRSID_SI

from gbgpu.gbcomps import GBWDMComputations
from gbgpu.gbsignalhetcomputations import GBSignalHetComputations

V5_KNOBS = dict(v3_n_nodes=32, v4_knots=64, v4_band=16, v5=1)


class _Holder:
    def __init__(self, data, invc):
        self.linear_data_arr = [np.ascontiguousarray(data).ravel()]
        self.linear_psd_arr = [np.ascontiguousarray(invc).ravel()]

    def __len__(self):
        return 1


class _Env:
    def __init__(self, **kw):
        self.kw = kw

    def __enter__(self):
        self.saved = {k: os.environ.get(k) for k in self.kw}
        os.environ.update(self.kw)

    def __exit__(self, *a):
        for k, v in self.saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


class SighetBandEdgeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        dt, Nf, Nt, edge = 10.0, 256, 512, 40
        t0 = int(0.5 * YRSID_SI / dt) * dt
        wdm = WDMSettings(Nf, Nt, dt, t0=t0, min_freq=3e-4, max_freq=2e-2,
                          min_time=edge * Nf * dt, max_time=(Nt - edge) * Nf * dt,
                          force_backend="cpu")
        cls.ch = ch = GBWDMComputations(
            wdm, t_ref=t0, Nt_sub=128, n_pad=16, N_sparse=256, N_cp_sig=0, N_cp_orbit=0,
            orbits=ESAOrbits(force_backend="cpu"), tdi_config="2nd generation",
            force_backend="cpu", d_d=0.0, tdi_type="XYZ")
        ch.convert_to_ra_dec = False
        df = wdm.layer_df
        m0 = int(wdm.ind_min_f)
        cls.m0 = m0
        # carrier 0.3 layers above the band floor: m_floor = ind_min_f, so the rows
        # m_floor - 2 and m_floor - 1 are out of band
        p = np.array([1e-21, (m0 + 0.3) * df, 1e-18, 0.0, 0.8, 0.9, 0.4, 2.0, 0.3])
        cls.p = p
        Nfa = int(wdm.ind_max_f - wdm.ind_min_f + 1)
        T = int(wdm.Nt_active)
        rng = np.random.default_rng(3)
        iC = np.empty((3, 3, Nfa, T))
        for c in range(3):
            iC[c, c] = rng.uniform(1.0, 2.0, (Nfa, T))
        for c, d in ((0, 1), (0, 2), (1, 2)):
            iC[c, d] = iC[d, c] = rng.uniform(-0.3, 0.3, (Nfa, T))
        cls.iC = iC
        d = np.zeros((1, 3, Nfa, T))
        ch.fill_global_wdm(p[None], d.reshape(-1), data_index=np.zeros(1, dtype=np.int32),
                           factors=np.ones(1))
        cls.data = d
        cls.zero = _Holder(np.zeros_like(d), iC[None])
        cls.full = _Holder(d, iC[None])

    def _comp(self, **kw):
        return GBSignalHetComputations.for_band_engine(self.ch, cp_repr="carrier",
                                                       **V5_KNOBS, **kw)

    def _exact(self, P, holder):
        z = np.zeros(len(P), dtype=np.int32)
        ll = np.asarray(self.ch.get_ll_wdm(P, holder, data_index=z, noise_index=z)).ravel()
        return ll, np.asarray(self.ch.h_h_out).copy()

    def test_reference_hh_at_the_band_floor(self):
        """Zero data, correction off: sig-het <h|h> at the reference == exact."""
        comp = self._comp()
        P = self.p[None]
        with _Env(SIGHET_ANCHOR_CORRECT="0"):
            comp.setup_in_model(self.zero, P, np.zeros(1, dtype=np.int32))
        try:
            comp.get_ll(P, data_index=np.zeros(1, dtype=np.int32))
            hh_sig = float(np.asarray(comp.last_h_h)[0])
        finally:
            comp.clear_in_model()
        hh_ex = float(self._exact(P, self.zero)[1][0])
        self.assertLess(abs(hh_sig / hh_ex - 1.0), 1e-3,
                        f"band-floor reference h_h ratio {hh_sig / hh_ex:.6f}")

    def test_legacy_scorers_reference_hh_at_the_band_floor(self):
        """The v2 / v3 / v4 scorers (v5 = 0, Re/Im) skip the out-of-band rows too.

        Their reference is the stash's own B0 sum, so the ratio sits at the sparse-grid
        accuracy (loose 1e-2 here); a duplicated edge row reads 2-3.
        """
        for name, knobs in (("v4", dict(v3_n_nodes=32, v4_knots=64, v4_band=16, v5=0)),
                            ("v3", dict(v3_n_nodes=32, v4_knots=0, v5=0)),
                            ("v2", dict(v3_n_nodes=0, v4_knots=0, v5=0))):
            comp = GBSignalHetComputations.for_band_engine(self.ch, **knobs)
            P = self.p[None]
            with _Env(SIGHET_ANCHOR_CORRECT="0"):
                comp.setup_in_model(self.zero, P, np.zeros(1, dtype=np.int32))
            try:
                comp.get_ll(P, data_index=np.zeros(1, dtype=np.int32))
                hh_sig = float(np.asarray(comp.last_h_h)[0])
            finally:
                comp.clear_in_model()
            hh_ex = float(self._exact(P, self.zero)[1][0])
            with self.subTest(scorer=name):
                self.assertLess(abs(hh_sig / hh_ex - 1.0), 1e-2,
                                f"{name} band-floor reference h_h ratio {hh_sig / hh_ex:.6f}")

    def test_displaced_candidate_delta_at_the_band_floor(self):
        comp = self._comp()
        P0 = self.p[None]
        P1 = P0.copy()
        P1[0, 0] *= 1.05
        P1[0, 4] += 0.05
        z = np.zeros(1, dtype=np.int32)
        with _Env(SIGHET_ANCHOR_CORRECT="0"):
            comp.setup_in_model(self.full, P0, z)
        try:
            d_sig = float(np.asarray(comp.get_ll(P1, data_index=z))[0]
                          - np.asarray(comp.get_ll(P0, data_index=z))[0])
        finally:
            comp.clear_in_model()
        d_ex = float(self._exact(P1, self.full)[0][0] - self._exact(P0, self.full)[0][0])
        self.assertLess(abs(d_sig - d_ex), max(1e-3 * abs(d_ex), 1e-6 * abs(self._exact(P0, self.full)[1][0])),
                        f"band-floor delta: sig-het {d_sig:.6g} vs exact {d_ex:.6g}")


if __name__ == "__main__":
    unittest.main()
