"""Sig-het v5 CARRIER reference built from the WDM lookup table (SIGHET_REF_BUILD).

The carrier-only reference is a unit envelope on the source's common phase -- the
chirping tone the lookup table stores -- so c0 is one table read per (pixel, layer)
and the packet first moment c1 the table's f-derivative. Pinned here:

* c0 / c1 from ``GBLookupComputations.make_carrier_reference`` against the FD
  producer ``gb_signal_het_make_reference_carrier`` (carriers across the Meyer flat
  top and transition band);
* the v5 scores with the lookup-built reference against the FD-built one and against
  the exact chunked deltas (channel-symmetric, NOT a I + b J inverse covariance: the
  sym layout);
* the knob: ``fd`` forces the FD build, ``lookup`` without a table refuses.

CPU. Shares the GB lookup table (built once into ``$GB_LOOKUP_TABLE_DIR``) with
test_gb_lookup_kernel.
"""

import os
import unittest

import numpy as np

from lisatools.detector import ESAOrbits
from lisatools.domains import WDMSettings
from lisatools.sensitivity import X2TDISens, XY2TDISens
from lisatools.utils.constants import YRSID_SI

from gbgpu.gbcomps import GBWDMComputations
from gbgpu.gbsignalhetcomputations import (GBSignalHetComputations, _c1_scale,
                                           _n_cp_kernel_arg, _window_dj)

from .test_gb_lookup_kernel import _table

NF, DT, NT, EDGE = 180, 20.0, 720, 60
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


class SighetLookupReferenceTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        t0 = int(0.4 * YRSID_SI / DT) * DT
        cls.wdm = wdm = WDMSettings(NF, NT, DT, t0=t0, min_freq=1e-4, max_freq=2.5e-2,
                                    min_time=EDGE * NF * DT, max_time=(NT - EDGE) * NF * DT,
                                    force_backend="cpu")
        cls.ch = ch = GBWDMComputations(
            wdm, t_ref=t0, Nt_sub=256, n_pad=32, N_sparse=256, N_cp_sig=48, N_cp_orbit=32,
            orbits=ESAOrbits(force_backend="cpu"), tdi_config="2nd generation",
            force_backend="cpu", d_d=0.0, tdi_type="XYZ")
        ch.convert_to_ra_dec = False
        cls.table = _table()
        df = wdm.layer_df
        rng = np.random.default_rng(5)
        # carrier in-layer offsets across the Meyer flat top and transition band
        f0 = np.array([(9 + 0.45) * df, (17 + 0.2) * df, (31 + 0.55) * df, (58 + 0.8) * df])
        R = len(f0)
        cls.p = np.column_stack([
            np.full(R, 1e-22), f0, 1e-18 * (f0 / 1e-3) ** (11 / 3), np.zeros(R),
            rng.uniform(0, 2 * np.pi, R), np.arccos(np.array([0.02, 0.4, 0.8, -0.3])),
            rng.uniform(0, np.pi, R), rng.uniform(0, 2 * np.pi, R),
            np.arcsin(rng.uniform(-1, 1, R))])
        Nfa = int(wdm.ind_max_f - wdm.ind_min_f + 1)
        T = int(wdm.Nt_active)
        fl = np.maximum((wdm.ind_min_f + np.arange(Nfa)) * df, 1e-5)
        sxx = np.asarray(X2TDISens.get_Sn(fl, model="scirdv1"), dtype=float)
        sxy = np.asarray(XY2TDISens.get_Sn(fl, model="scirdv1"), dtype=float)
        C = np.empty((Nfa, 3, 3))
        for c in range(3):
            for d in range(3):
                C[:, c, d] = sxx if c == d else sxy
        D = np.diag([0.99, 1.0, 1.01])          # unequal gains: sym, not a I + b J
        C = D[None] @ C @ D[None]
        iC = np.broadcast_to(np.linalg.inv(C).transpose(1, 2, 0)[None, :, :, :, None],
                             (R, 3, 3, Nfa, T)).copy()
        idx = np.arange(R, dtype=np.int32)
        unit = np.zeros((R, 3, Nfa, T))
        ch.fill_global_wdm(cls.p, unit.reshape(-1), data_index=idx, factors=np.ones(R))
        # SNR 300 per source: data = the source, the reference at the source
        snr = np.sqrt(np.einsum("rcft,rcdft,rdft->r", unit, iC, unit))
        cls.ref = cls.p.copy()
        cls.ref[:, 0] *= 300.0 / snr
        data = np.zeros_like(unit)
        ch.fill_global_wdm(cls.ref, data.reshape(-1), data_index=idx, factors=np.ones(R))
        cls.holder = _Holder(data, iC)
        cls.R = R
        q = cls.ref.copy()
        q[:, 0] *= 1.04
        q[:, 4] += 0.05
        q[:, 1] += 0.2 / wdm.Tobs
        cls.cand = q

    def _comp(self, table=True):
        return GBSignalHetComputations.for_band_engine(
            self.ch, cp_repr="carrier", lookup_table=self.table if table else None, **V5_KNOBS)

    def test_reference_matches_fd_producer(self):
        comp = self._comp()
        g = comp._g
        R, W, Nta = self.R, 9, g["Nt_active"]
        w_lo = (np.floor(self.p[:, 1] / self.wdm.layer_df).astype(int)
                - g["ind_min_f"] - W // 2).astype(np.int32)
        sp = np.zeros((R, 1, W, g["N_sparse_t"]), dtype=complex)
        de = np.zeros((R, 1, W, Nta), dtype=complex)
        c1 = np.zeros_like(de)
        comp.cpp.gb_signal_het_make_reference_carrier(
            comp.tdi_wrap, sp, de, c1, comp.window_full, _window_dj(comp.window_full, np),
            comp.n_sparse_local, w_lo, np.ascontiguousarray(self.p), R, 9, 1, 2,
            g["Nf"], g["Nt"], W, Nta, g["nt_layer"], g["N_sparse_t"], g["stride"],
            g["ind_min_t"], g["ind_min_f"], g["layer_df"], g["dt"], g["Tobs"], g["t0"], 3,
            g["n_sparse_fd"], g["tukey_alpha"], _n_cp_kernel_arg(g, allow_carrier=True))
        c1 *= _c1_scale(g["Nt"])
        c0L, c1L = comp._ref_lookup.make_carrier_reference(self.p, w_lo, W)
        for i in range(R):
            n0 = np.linalg.norm(de[i, 0])
            self.assertLess(np.linalg.norm(c0L[i] - de[i, 0]) / n0, 1e-5, f"c0 source {i}")
            self.assertLess(np.linalg.norm(c1L[i] - c1[i, 0]) / n0, 1e-4, f"c1 source {i}")

    def _scores(self, comp, **env):
        di = np.arange(self.R, dtype=np.int32)
        with _Env(SIGHET_ANCHOR_CORRECT="0", **env):
            comp.setup_in_model(self.holder, self.ref, di)
        try:
            ll_c = np.asarray(comp.get_ll(self.cand, data_index=di)).copy()
            ll_r = np.asarray(comp.get_ll(self.ref, data_index=di)).copy()
            hh = np.asarray(comp.last_h_h).copy()
            build = comp._stash_ref_build
        finally:
            comp.clear_in_model()
        return ll_c - ll_r, hh, build

    def test_scores_match_fd_reference_and_chunked(self):
        comp = self._comp()
        d_lk, hh, b_lk = self._scores(comp)
        d_fd, _, b_fd = self._scores(comp, SIGHET_REF_BUILD="fd")
        self.assertEqual((b_lk, b_fd), ("lookup", "fd"))
        di = np.arange(self.R, dtype=np.int32)
        ex = (np.asarray(self.ch.get_ll_wdm(self.cand, self.holder, data_index=di,
                                            noise_index=di)).ravel()
              - np.asarray(self.ch.get_ll_wdm(self.ref, self.holder, data_index=di,
                                              noise_index=di)).ravel())
        # Measured (SNR 300, h_h 9.0e4, steps of ~2e4 lnL): lookup - FD reference
        # <= 8e-4 lnL; both within 0.73 lnL of chunked; c1 off: up to 8.8 lnL.
        np.testing.assert_allclose(d_lk, d_fd, rtol=0, atol=1e-7 * hh.max())
        for name, d in (("lookup", d_lk), ("fd", d_fd)):
            np.testing.assert_allclose(d, ex, rtol=0, atol=2e-5 * hh.max(),
                                       err_msg=f"{name}-built reference vs chunked")
        # negative control: the lookup's c1 is load-bearing (transition-band carriers)
        d_noc1, _, _ = self._scores(comp, SIGHET_CARRIER_C1="0")
        self.assertGreater(np.abs(d_noc1 - ex).max(), 4e-5 * hh.max())

    def test_knob(self):
        with _Env(SIGHET_REF_BUILD="lookup"):
            with self.assertRaises(RuntimeError):
                self._scores(self._comp(table=False))
        _, _, b = self._scores(self._comp(table=False))
        self.assertEqual(b, "fd")


if __name__ == "__main__":
    unittest.main()
