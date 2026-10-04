"""Sig-het F-statistic with the CARRIER reference + SYM Gram fold (SIGHET_FSTAT_CARRIER).

The legacy sig-het F-stat fits each channel's filter/reference ratio in log-polar
form against a circular reference; per-channel fit errors are incoherent across
X, Y, Z and the near-singular low-f X+Y+Z direction of the inverse covariance
amplifies them into the Gram M (measured on the unequal-arm production noise: F
low by up to 70 % at 0.6 mHz). The carrier path builds the reference as the
common carrier (+ the c1 packet moment), the filter ratio is the filter's own
demodulated envelope (Re / Im, linear in the channels), and M folds with the
SYM moments (second moment + c1).

Pinned against the exact chunked F-stat (``GBWDMComputations.get_fstat_ll_wdm``)
on a channel-symmetric, NOT a*I + b*J inverse covariance with the low-f null
(the analytic XYZ covariance with per-channel diagonal offsets): carrier F to
<1e-3 relative; the legacy path (SIGHET_FSTAT_CARRIER=0) is the negative control
and must miss by >1e-2 at low f. CPU.
"""

import os
import unittest

import numpy as np

from lisatools.detector import ESAOrbits
from lisatools.domains import WDMSettings
from lisatools.sensitivity import X2TDISens, XY2TDISens
from lisatools.utils.constants import YRSID_SI

from gbgpu.gbcomps import GBWDMComputations
from gbgpu.gbsignalhetcomputations import GBSignalHetComputations

NF, DT, NT, EDGE = 180, 20.0, 2160, 60


def _F(N, M):
    out = []
    iu = np.triu_indices(4)
    for n_, m_ in zip(np.asarray(N), np.asarray(M)):
        Mm = np.zeros((4, 4))
        Mm[iu] = m_
        Mm = Mm + Mm.T - np.diag(np.diag(Mm))
        out.append(float(n_ @ np.linalg.solve(Mm, n_)))
    return np.array(out)


class _H:
    def __init__(self, d, iC):
        self.linear_data_arr = [np.ascontiguousarray(d).ravel()]
        self.linear_psd_arr = [np.ascontiguousarray(iC[None]).ravel()]

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


class SighetFstatCarrierTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        t0 = int(0.4 * YRSID_SI / DT) * DT
        wdm = WDMSettings(NF, NT, DT, t0=t0, min_freq=1e-4, max_freq=2.5e-2,
                          min_time=EDGE * NF * DT, max_time=(NT - EDGE) * NF * DT,
                          force_backend="cpu")
        cls.ch = ch = GBWDMComputations(
            wdm, t_ref=t0, Nt_sub=256, n_pad=32, N_sparse=256, N_cp_sig=48, N_cp_orbit=32,
            orbits=ESAOrbits(force_backend="cpu"), tdi_config="2nd generation",
            force_backend="cpu", d_d=0.0, tdi_type="XYZ")
        ch.convert_to_ra_dec = False
        Nfa = int(wdm.ind_max_f - wdm.ind_min_f + 1)
        T = int(wdm.Nt_active)
        fl = np.maximum((wdm.ind_min_f + np.arange(Nfa)) * wdm.layer_df, 1e-5)
        sxx = np.asarray(X2TDISens.get_Sn(fl, model="scirdv1"), dtype=float)
        sxy = np.asarray(XY2TDISens.get_Sn(fl, model="scirdv1"), dtype=float)
        C = np.empty((Nfa, 3, 3))
        for c in range(3):
            for d in range(3):
                C[:, c, d] = sxx if c == d else sxy
        # unequal channel gains: a congruence D C D keeps C positive definite
        # (a raw diagonal offset makes the near-singular low-f C indefinite)
        D = np.diag([0.99, 1.0, 1.01])
        C = D[None] @ C @ D[None]
        cls.iC = np.broadcast_to(np.linalg.inv(C).transpose(1, 2, 0)[:, :, :, None],
                                 (3, 3, Nfa, T)).copy()
        rng = np.random.default_rng(13)
        cls.cases = []
        for f0, ci in ((6.3e-4, 0.02), (6.3e-4, 0.8), (1.25e-3, 0.3), (4e-3, 0.5)):
            p = np.array([[1e-22, f0, 1e-18 * (f0 / 1e-3) ** (11 / 3), 0.0, rng.uniform(0, 6),
                           np.arccos(ci), rng.uniform(0, 3), rng.uniform(0, 6),
                           np.arcsin(rng.uniform(-1, 1))]])
            d = np.zeros((1, 3, Nfa, T))
            ch.fill_global_wdm(p, d.reshape(-1), data_index=np.zeros(1, dtype=np.int32),
                               factors=np.ones(1))
            d *= 30.0 / np.sqrt(np.einsum("cft,cdft,dft->", d[0], cls.iC, d[0]))
            P = np.stack([p[0], p[0] + np.array([0, 0.5 / wdm.Tobs] + [0] * 7)])
            cls.cases.append((P, _H(d, cls.iC)))
        cls.sig = GBSignalHetComputations.for_band_engine(
            ch, nt_layer=-1, n_sparse_fd=1024, m_active_half_width=2, max_r=0.0,
            n_cp_build=256, v3_n_nodes=64, v4_knots=128, v4_band=16, v5=1, tukey_alpha=0.01)

    def _rel(self, carrier):
        out = []
        for P, h in self.cases:
            z = np.zeros(len(P), dtype=np.int32)
            Fc = _F(*self.ch.get_fstat_ll_wdm(P, h, data_index=z, noise_index=z))
            with _Env(SIGHET_FSTAT_CARRIER="1" if carrier else "0"):
                self.sig.setup_fstat_references(P[:1], h, data_index=0, noise_index=0)
            try:
                self.assertEqual(bool(self.sig._fstat["carrier"]), carrier)
                Fs = _F(*self.sig.get_fstat_ll_wdm(P, data_index=z))
            finally:
                self.sig.clear_fstat_references()
            out.append(np.abs(Fs / Fc - 1.0))
        return np.array(out)

    def test_carrier_fstat_matches_chunked(self):
        rel = self._rel(True)
        self.assertLess(rel.max(), 1e-3, f"carrier F rel err {rel}")

    def test_legacy_fstat_misses_at_low_f(self):
        """Negative control: the legacy log-polar ratio is far off at 0.63 mHz."""
        rel = self._rel(False)
        self.assertGreater(rel[:2].max(), 1e-2, f"legacy F rel err {rel}")


if __name__ == "__main__":
    unittest.main()
