"""Carrier-mode COLLAPSED sig-het stash + the second (B2) fold moment.

The v5 fold models each candidate's WDM coefficients in a time bin as
``c0(n) * (r + dr * n_off)`` and forms ``<h|h>`` from template moments
``sum_bin conj(c0) iC c0 n_off^q``. The full layout carried q = 0, 1 only: the
``conj(dr) dr * n_off^2`` term was dropped. Harmless while r ~ 1 near the
reference; with the carrier-only reference r is the source's whole demodulated
envelope, and the missing term was the carrier-mode anchor offset (0.5-1 ln L
at SNR 100, 6 months).

In carrier mode c0 is the same in every channel, and the equal-arm XYZ inverse
covariance is ``iC_cd = a delta_cd + b``, so each nch x nch block is
``a_q delta_cd + b_q``: the stash keeps (a_q, b_q) for q = 0, 1, 2 -- 12
complex moments per pixel instead of 36, B2 included.

Pinned here:

* **Moments vs brute force** and the **full-layout identity** (the collapsed
  q = 0, 1 moments reproduce ``bin_fold_real``'s nch x nch blocks).
* **The fold is exact for its own piecewise-linear r** -- the identity the B2
  term restores; dropping it breaks the test (mutation check).
* **Negative controls**: an asymmetric invC is detected (and the collapse
  would be wrong for it); a channel-dependent c0 raises.
* **Compiled kernel**: collapsed with q = 2 zeroed == the full layout; the q = 2
  moments move h_h; an asymmetric buffer falls back loudly to the full layout,
  and a mid-block patch cannot switch layouts.

CPU-only (numpy); the compiled tests use the small grid of
``test_sighet_inmodel_window``.
"""

import os
import unittest

import numpy as np

from lisatools.detector import ESAOrbits
from lisatools.domains import WDMSettings
from lisatools.signal_het import bin_fold_real, sparse_time_grid
from lisatools.utils.constants import YRSID_SI

from gbgpu.gbcomps import GBWDMComputations
from gbgpu.gbsignalhetcomputations import (
    GBSignalHetComputations,
    _collapsed_carrier_fold,
    _invc_channel_symmetric,
)

V5_KNOBS = dict(v3_n_nodes=32, v4_knots=64, v4_band=16, v5=1)


def _sym_invc(rng, k, W, Nt):
    """(k, 3, 3, W, Nt) a*I + b*J, positive definite, varying per pixel."""
    a = rng.uniform(0.5, 2.0, (k, W, Nt))
    b = -rng.uniform(0.0, 0.3, (k, W, Nt)) * a
    iC = np.empty((k, 3, 3, W, Nt))
    for c in range(3):
        for d in range(3):
            iC[:, c, d] = b + (a if c == d else 0.0)
    return iC


class _Toy:
    """Random carrier c0 / residual / symmetric invC on a short sparse grid."""

    def __init__(self, seed=7, k=2, W=3, Nt_layer=8, stride=9, Nt_active=70):
        rng = np.random.default_rng(seed)
        self.k, self.W, self.Nt = k, W, Nt_active
        self.stride = stride
        self.Ns = Nt_active // stride
        self.nb = (stride // 2 + np.arange(self.Ns) * stride).astype(np.int32)
        c0 = rng.standard_normal((k, W, Nt_active)) + 1j * rng.standard_normal((k, W, Nt_active))
        self.c0 = np.repeat(c0[:, None], 3, axis=1)                     # same in every channel
        self.res = rng.standard_normal((k, 3, W, Nt_active)) + 0j
        self.iC = _sym_invc(rng, k, W, Nt_active)
        edges = np.arange(self.Ns + 1) * stride
        edges[-1] = Nt_active
        self.bin_idx = np.repeat(np.arange(self.Ns), np.diff(edges))
        self.n_off = np.arange(Nt_active) - self.nb[self.bin_idx]
        self.rng = rng

    def fold(self):
        return _collapsed_carrier_fold(self.res, self.c0, self.iC, self.nb, self.stride, self.Nt)


def _fold_hh(P, r, dr, drop_q2=False):
    """Python mirror of the kernel's collapsed fold: 0.5 Re(sum ...) per source.
    ``r, dr`` (k, 3, W, Ns)."""
    Pa, Pb, Pna, Pnb = P
    q2 = 0.0 if drop_q2 else 1.0
    X = [np.sum(np.conj(r) * r, 1), np.sum(np.conj(r) * dr + np.conj(dr) * r, 1),
         q2 * np.sum(np.conj(dr) * dr, 1)]
    Sr, Sdr = r.sum(1), dr.sum(1)
    Y = [np.conj(Sr) * Sr, np.conj(Sr) * Sdr + np.conj(Sdr) * Sr, q2 * np.conj(Sdr) * Sdr]
    Z = [np.sum(r * r, 1), 2 * np.sum(r * dr, 1), q2 * np.sum(dr * dr, 1)]
    ZJ = [Sr * Sr, 2 * Sr * Sdr, q2 * Sdr * Sdr]
    tot = sum(Pa[:, q] * X[q] + Pb[:, q] * Y[q] + Pna[:, q] * Z[q] + Pnb[:, q] * ZJ[q]
              for q in range(3))
    return 0.5 * np.real(tot.sum(axis=(-2, -1)))


class CollapsedMomentsTest(unittest.TestCase):
    def setUp(self):
        self.t = _Toy()

    def test_moments_vs_brute_force(self):
        t = self.t
        A0, A1, Pa, Pb, Pna, Pnb = t.fold()
        c0 = t.c0[:, 0]
        a = t.iC[:, 0, 0] - t.iC[:, 0, 1]
        b = t.iC[:, 0, 1]
        for q in range(3):
            w = t.n_off.astype(float) ** q
            for bb in range(t.Ns):
                sel = t.bin_idx == bb
                E = np.abs(c0[..., sel]) ** 2 * w[sel]
                En = c0[..., sel] ** 2 * w[sel]
                for got, want in ((Pa, (E * a[..., sel]).sum(-1)),
                                  (Pb, (E * b[..., sel]).sum(-1)),
                                  (Pna, (En * a[..., sel]).sum(-1)),
                                  (Pnb, (En * b[..., sel]).sum(-1))):
                    np.testing.assert_allclose(got[:, q, :, bb], want, rtol=1e-12,
                                               atol=1e-12 * np.abs(want).max())

    def test_full_layout_identity(self):
        """q = 0, 1 collapsed moments == bin_fold_real's nch x nch blocks; A0/A1 equal."""
        t = self.t
        A0, A1, Pa, Pb, Pna, Pnb = t.fold()
        fA0, fA1, B0, B1, B0nc, B1nc = bin_fold_real(t.res, t.c0, t.iC, t.nb, t.stride, t.Nt,
                                                      tdi_type="XYZ")
        np.testing.assert_allclose(A0, fA0, rtol=1e-13, atol=1e-13 * np.abs(fA0).max())
        np.testing.assert_allclose(A1, fA1, rtol=1e-13, atol=1e-13 * np.abs(fA1).max())
        for c in range(3):
            for d in range(3):
                dl = 1.0 if c == d else 0.0
                for full, pa, pb, q in ((B0, Pa, Pb, 0), (B1, Pa, Pb, 1),
                                        (B0nc, Pna, Pnb, 0), (B1nc, Pna, Pnb, 1)):
                    want = full[:, c, d]
                    np.testing.assert_allclose(dl * pa[:, q] + pb[:, q], want, rtol=1e-12,
                                               atol=1e-12 * np.abs(want).max())

    def test_fold_exact_for_piecewise_linear_r(self):
        """<h|h> of h = c0 (r + dr n_off) == the collapsed fold, only WITH the q = 2 term."""
        t = self.t
        P = t.fold()[2:]
        rng = t.rng
        r = rng.standard_normal((t.k, 3, t.W, t.Ns)) + 1j * rng.standard_normal((t.k, 3, t.W, t.Ns))
        dr = 0.3 * (rng.standard_normal(r.shape) + 1j * rng.standard_normal(r.shape))
        h = t.c0 * (r[..., t.bin_idx] + dr[..., t.bin_idx] * t.n_off)
        dense = np.einsum("kcwn,kcdwn,kdwn->k", h.real, t.iC, h.real)
        np.testing.assert_allclose(_fold_hh(P, r, dr), dense, rtol=1e-12)
        miss = np.abs(_fold_hh(P, r, dr, drop_q2=True) / dense - 1)
        self.assertGreater(miss.min(), 1e-3,
                           "dropping the q = 2 moment must break exactness (mutation check)")


class NegativeControlTest(unittest.TestCase):
    def test_asymmetric_invc_detected(self):
        t = _Toy()
        self.assertTrue(_invc_channel_symmetric(t.iC, np))
        bad = t.iC.copy()
        bad[:, 1, 1] *= 1.0 + 1e-6                 # unequal arm noise in Y
        self.assertFalse(_invc_channel_symmetric(bad, np))
        bad2 = t.iC.copy()
        bad2[:, 0, 2] *= 1.0 + 1e-6
        bad2[:, 2, 0] *= 1.0 + 1e-6
        self.assertFalse(_invc_channel_symmetric(bad2, np))

    def test_collapse_wrong_for_asymmetric_invc(self):
        """The guard is load-bearing: on asymmetric noise the (a, b) moments miss the blocks."""
        t = _Toy()
        t.iC[:, 1, 1] *= 1.5
        _, _, Pa, Pb, _, _ = t.fold()
        _, _, B0, _, _, _ = bin_fold_real(t.res, t.c0, t.iC, t.nb, t.stride, t.Nt, tdi_type="XYZ")
        err = np.abs(Pa[:, 0] + Pb[:, 0] - B0[:, 1, 1]).max() / np.abs(B0[:, 1, 1]).max()
        self.assertGreater(err, 1e-2)

    def test_channel_dependent_c0_raises(self):
        t = _Toy()
        t.c0[:, 2] *= 1.01
        with self.assertRaises(RuntimeError):
            t.fold()


class _SlotHolder:
    def __init__(self, data_slabs, invC_slabs):
        self.linear_data_arr = [np.ascontiguousarray(data_slabs).ravel()]
        self.linear_psd_arr = [np.ascontiguousarray(invC_slabs).ravel()]

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


class CompiledCollapseTest(unittest.TestCase):
    """The compiled v5 scorer on the collapsed stash (small 15-day grid)."""

    @classmethod
    def setUpClass(cls):
        dt, Nf, Nt, edge = 10.0, 256, 512, 40
        t0 = int(0.5 * YRSID_SI / dt) * dt
        layer_df = 1.0 / (2.0 * Nf * dt)
        orbits = ESAOrbits(force_backend="cpu")
        wdm = WDMSettings(Nf, Nt, dt, t0=t0, min_freq=1e-4, max_freq=2e-2,
                          min_time=edge * Nf * dt, max_time=(Nt - edge) * Nf * dt,
                          force_backend="cpu")
        cls.chunked = chunked = GBWDMComputations(
            wdm, t_ref=t0, Nt_sub=128, n_pad=16, N_sparse=256, N_cp_sig=0, N_cp_orbit=0,
            orbits=orbits, tdi_config="2nd generation", force_backend="cpu", d_d=0.0,
            tdi_type="XYZ")
        chunked.convert_to_ra_dec = False
        A = np.array([1e-21, (int(3e-3 / layer_df) + 0.37) * layer_df, 1e-17, 0.0, 1.2, 0.7,
                      0.4, 2.0, 0.5])
        C = np.array([8e-22, (int(5e-3 / layer_df) + 0.62) * layer_df, 2e-17, 0.0, 0.4, 1.53,
                      0.9, 4.0, -0.3])
        cls.params_ref = np.stack([A, C])
        ilo, ihi = wdm.ind_min_f, wdm.ind_max_f + 1
        rng = np.random.default_rng(11)
        slabs, invCs = [], []
        for p in (A, C):
            h = np.zeros((3, Nf, Nt))
            chunked.fill_global_wdm(p[None, :], h, convert_to_ra_dec=False)
            h_act = np.ascontiguousarray(h[:, ilo:ihi, wdm.active_slice_t])
            slabs.append(h_act)
            invCs.append(_sym_invc(rng, 1, h_act.shape[1], h_act.shape[2])[0])
        cls.slabs, cls.invCs = np.stack(slabs), np.stack(invCs)
        cls.holder = _SlotHolder(cls.slabs, cls.invCs)
        cls.slots = np.array([0, 1], dtype=np.int32)
        rows = [A, C]
        for _ in range(2):
            for p in (A, C):
                q = p.copy()
                q[0] *= 1.0 + 0.15 * rng.standard_normal()
                q[1] += 0.05 * layer_df * rng.standard_normal()
                q[4] = rng.uniform(0.0, 2 * np.pi)
                rows.append(q)
        cls.params = np.stack(rows)
        cls.di = np.array([0, 1, 0, 1, 0, 1], dtype=np.int32)

    def _comp(self):
        return GBSignalHetComputations.for_band_engine(self.chunked, cp_repr="carrier",
                                                       **V5_KNOBS)

    def _scores(self, comp, holder=None, collapse="1", zero_q2=False):
        with _Env(SIGHET_CARRIER_COLLAPSE=collapse, SIGHET_ANCHOR_CORRECT="0"):
            comp.setup_in_model(self.holder if holder is None else holder, self.params_ref,
                                self.slots)
        try:
            if zero_q2:
                for arr in (comp.B0_all, comp.B1_all, comp.B0nc_all, comp.B1nc_all):
                    arr[:, 2] = 0.0
            comp.get_ll(self.params, data_index=self.di)
            return (bool(comp._stash_collapsed), np.asarray(comp.last_d_h).copy(),
                    np.asarray(comp.last_h_h).copy())
        finally:
            comp.clear_in_model()

    def test_collapsed_without_q2_equals_full_layout(self):
        comp = self._comp()
        c_full, dh_f, hh_f = self._scores(comp, collapse="0")
        c_col, dh_c, hh_c = self._scores(comp, collapse="1", zero_q2=True)
        self.assertFalse(c_full)
        self.assertTrue(c_col)
        np.testing.assert_allclose(dh_c, dh_f, rtol=1e-11, atol=1e-11 * np.abs(dh_f).max())
        np.testing.assert_allclose(hh_c, hh_f, rtol=1e-11, atol=1e-11 * np.abs(hh_f).max())

    def test_q2_moments_reach_the_score(self):
        comp = self._comp()
        _, dh0, hh0 = self._scores(comp, collapse="1", zero_q2=True)
        _, dh2, hh2 = self._scores(comp, collapse="1")
        np.testing.assert_array_equal(dh2, dh0)          # <d|h> has no q = 2 term
        self.assertGreater(np.min(hh2 - hh0), 0.0,
                           "conj(dr) dr n_off^2 is a positive-definite addition to <h|h>")

    def test_asymmetric_buffer_falls_back_loudly(self):
        bad = self.invCs.copy()
        bad[:, 1, 1] *= 1.2
        holder = _SlotHolder(self.slabs, bad)
        comp = self._comp()
        with self.assertLogs("gbgpu.gbsignalhetcomputations", level="WARNING") as cm:
            collapsed, dh, hh = self._scores(comp, holder=holder, collapse="1")
        self.assertFalse(collapsed)
        self.assertTrue(any("NOT channel-symmetric" in m for m in cm.output))
        self.assertTrue(np.all(np.isfinite(hh)) and np.all(hh > 0))

    def test_patch_cannot_switch_layout(self):
        bad = self.invCs.copy()
        bad[:, 1, 1] *= 1.2
        comp = self._comp()
        with _Env(SIGHET_CARRIER_COLLAPSE="1", SIGHET_ANCHOR_CORRECT="0"):
            comp.setup_in_model(self.holder, self.params_ref, self.slots)
            try:
                self.assertTrue(comp._stash_collapsed)
                with self.assertRaises(RuntimeError):
                    comp.setup_in_model(_SlotHolder(self.slabs, bad), self.params_ref[:1],
                                        self.slots[:1])
            finally:
                comp.clear_in_model()
        self.assertFalse(comp._stash_collapsed)


if __name__ == "__main__":
    unittest.main()
