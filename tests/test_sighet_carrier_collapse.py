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
    _SYM_PAIRS,
    _collapsed_carrier_fold,
    _invc_channel_symmetric,
    _invc_pair_symmetric,
    _window_dj,
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


def _unequal_invc(rng, k, W, Nt):
    """(k, 3, 3, W, Nt) channel-symmetric but NOT a*I + b*J (unequal arms): distinct
    diagonals and distinct off-diagonal pairs, diagonally dominant (positive definite)."""
    iC = np.empty((k, 3, 3, W, Nt))
    for c in range(3):
        iC[:, c, c] = rng.uniform(1.0, 2.0, (k, W, Nt))
    for c, d in ((0, 1), (0, 2), (1, 2)):
        iC[:, c, d] = iC[:, d, c] = rng.uniform(-0.3, 0.3, (k, W, Nt))
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

    def fold(self, c1=None):
        return _collapsed_carrier_fold(self.res, self.c0, self.iC, self.nb, self.stride, self.Nt,
                                       c1_dense=c1)

    def random_c1(self, scale=0.4):
        shp = (self.k, self.W, self.Nt)
        c1 = scale * (self.rng.standard_normal(shp) + 1j * self.rng.standard_normal(shp))
        return np.repeat(c1[:, None], 3, axis=1)


def _fold_hh(P, r, dr, drop_q2=False):
    """Python mirror of the kernel's collapsed fold: 0.5 Re(sum ...) per source.
    ``r, dr`` (k, 3, W, Ns). The q = 1 conj moments pair with 2 sum conj(r) dr."""
    Pa, Pb, Pna, Pnb = P
    q2 = 0.0 if drop_q2 else 1.0
    X = [np.sum(np.conj(r) * r, 1), 2 * np.sum(np.conj(r) * dr, 1),
         q2 * np.sum(np.conj(dr) * dr, 1)]
    Sr, Sdr = r.sum(1), dr.sum(1)
    Y = [np.conj(Sr) * Sr, 2 * np.conj(Sr) * Sdr, q2 * np.conj(Sdr) * Sdr]
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


class FirstMomentTest(unittest.TestCase):
    """The packet first moment c1: template c0 (r + dr n_off) + c1 dr folds exactly."""

    def _rdr(self, t):
        rng = t.rng
        r = rng.standard_normal((t.k, 3, t.W, t.Ns)) + 1j * rng.standard_normal((t.k, 3, t.W, t.Ns))
        dr = 0.3 * (rng.standard_normal(r.shape) + 1j * rng.standard_normal(r.shape))
        return r, dr

    def test_hh_exact_with_c1(self):
        t = _Toy(seed=3)
        c1 = t.random_c1()
        P = t.fold(c1)[2:]
        r, dr = self._rdr(t)
        h = t.c0 * (r[..., t.bin_idx] + dr[..., t.bin_idx] * t.n_off) + c1 * dr[..., t.bin_idx]
        dense = np.einsum("kcwn,kcdwn,kdwn->k", h.real, t.iC, h.real)
        np.testing.assert_allclose(_fold_hh(P, r, dr), dense, rtol=1e-12)
        miss = np.abs(_fold_hh(t.fold()[2:], r, dr) / dense - 1)
        self.assertGreater(miss.min(), 1e-3, "dropping c1 must break exactness (mutation check)")

    def test_dh_exact_with_c1(self):
        t = _Toy(seed=4)
        c1 = t.random_c1()
        A0, A1 = t.fold(c1)[:2]
        r, dr = self._rdr(t)
        h = t.c0 * (r[..., t.bin_idx] + dr[..., t.bin_idx] * t.n_off) + c1 * dr[..., t.bin_idx]
        dense = np.einsum("kcwn,kcdwn,kdwn->k", t.res.real, t.iC, h.real)
        fold = 0.5 * np.real(np.sum(A0 * r + A1 * dr, axis=(1, 2, 3)))
        np.testing.assert_allclose(fold, dense, rtol=1e-12)

    def test_window_derivative(self):
        """_window_dj against the analytic derivative of the Meyer window (WDMSettings.phitilde)."""
        from scipy import special
        wdm = WDMSettings(180, 1440, 20.0, force_backend="cpu")
        dO = np.pi / wdm.Nf
        A, B, nn = wdm.A, dO - 2 * wdm.A, wdm.WAVELET_FILTER_CONSTANT
        om = np.asarray(wdm.omega)
        x = np.clip((np.abs(om) - A) / B, 0.0, 1.0)
        trans = (np.abs(om) >= A) & (np.abs(om) < A + B)
        y = special.betainc(nn, nn, x)
        dy = x ** (nn - 1) * (1 - x) ** (nn - 1) / special.beta(nn, nn)
        dphi = np.where(trans, -np.sqrt(1 / dO) * np.sin(np.pi * y / 2) * (np.pi / 2) * dy / B
                        * np.sign(om), 0.0)
        want = dphi * 2 * np.pi / wdm.N            # d omega / d j
        got = np.asarray(_window_dj(wdm.window, np))
        # 4th-order difference across a C^3 kink: ~3e-7 at this Nt (1440), ~27x less at 4320
        self.assertLess(np.abs(got - want).max() / np.abs(want).max(), 1e-6)


def _fold_hh_sym(P, r, dr):
    """Python mirror of the kernel's SYM fold. ``P = (B0, B1, B0nc, B1nc)`` each
    (k, 3 q, 3 pair, W, Ns); ``r, dr`` (k, 3, W, Ns)."""
    B0, B1, B0nc, B1nc = P
    cj = np.conj
    tot = 0.0
    for p in range(3):
        rp, dp = r[:, p], dr[:, p]
        tot = tot + B0[:, 0, p] * (cj(rp) * rp) + B0[:, 1, p] * 2 * cj(rp) * dp \
            + B0[:, 2, p] * (cj(dp) * dp)
        tot = tot + B0nc[:, 0, p] * rp * rp + B0nc[:, 1, p] * 2 * rp * dp + B0nc[:, 2, p] * dp * dp
        c, d = _SYM_PAIRS[3 + p]
        rc, rd, dc, dd = r[:, c], r[:, d], dr[:, c], dr[:, d]
        tot = tot + B1[:, 0, p] * (cj(rc) * rd + cj(rd) * rc) \
            + B1[:, 1, p] * 2 * (cj(rc) * dd + cj(rd) * dc) + B1[:, 2, p] * (cj(dc) * dd + cj(dd) * dc)
        tot = tot + B1nc[:, 0, p] * 2 * rc * rd + B1nc[:, 1, p] * 2 * (rc * dd + dc * rd) \
            + B1nc[:, 2, p] * 2 * dc * dd
    return 0.5 * np.real(tot.sum(axis=(-2, -1)))


class SymLayoutTest(unittest.TestCase):
    """The carrier SYM layout: any channel-symmetric invC (the unequal-arm production noise)."""

    def _toy(self, seed=9):
        t = _Toy(seed=seed)
        t.iC = _unequal_invc(t.rng, t.k, t.W, t.Nt)
        return t

    def test_detection(self):
        t = self._toy()
        self.assertFalse(_invc_channel_symmetric(t.iC, np))
        self.assertTrue(_invc_pair_symmetric(t.iC, np))
        bad = t.iC.copy()
        bad[:, 0, 1] *= 1.0 + 1e-6
        self.assertFalse(_invc_pair_symmetric(bad, np))

    def test_full_layout_identity(self):
        """q = 0, 1 SYM moments (c1 off) == bin_fold_real's nch x nch blocks."""
        t = self._toy()
        A0, A1, B0, B1, B0nc, B1nc = _collapsed_carrier_fold(
            t.res, t.c0, t.iC, t.nb, t.stride, t.Nt, layout="sym")
        fA0, fA1, F0, F1, F0nc, F1nc = bin_fold_real(t.res, t.c0, t.iC, t.nb, t.stride, t.Nt,
                                                      tdi_type="XYZ")
        np.testing.assert_allclose(A0, fA0, rtol=1e-13, atol=1e-13 * np.abs(fA0).max())
        for full, dia, off, q in ((F0, B0, B1, 0), (F1, B0, B1, 1), (F0nc, B0nc, B1nc, 0),
                                  (F1nc, B0nc, B1nc, 1)):
            for p, (c, d) in enumerate(_SYM_PAIRS):
                got = dia[:, q, p] if p < 3 else off[:, q, p - 3]
                for cc, dd in ((c, d), (d, c)):
                    np.testing.assert_allclose(got, full[:, cc, dd], rtol=1e-12,
                                               atol=1e-12 * np.abs(full).max())

    def test_hh_exact_with_c1(self):
        t = self._toy(seed=10)
        c1 = t.random_c1()
        P = _collapsed_carrier_fold(t.res, t.c0, t.iC, t.nb, t.stride, t.Nt, c1_dense=c1,
                                    layout="sym")[2:]
        rng = t.rng
        r = rng.standard_normal((t.k, 3, t.W, t.Ns)) + 1j * rng.standard_normal((t.k, 3, t.W, t.Ns))
        dr = 0.3 * (rng.standard_normal(r.shape) + 1j * rng.standard_normal(r.shape))
        h = t.c0 * (r[..., t.bin_idx] + dr[..., t.bin_idx] * t.n_off) + c1 * dr[..., t.bin_idx]
        dense = np.einsum("kcwn,kcdwn,kdwn->k", h.real, t.iC, h.real)
        np.testing.assert_allclose(_fold_hh_sym(P, r, dr), dense, rtol=1e-12)

    def test_sym_equals_collapsed_on_equal_arm(self):
        """On a*I + b*J noise the SYM moments are the collapsed ones: diag = a + b, off = b."""
        t = _Toy(seed=11)
        c1 = t.random_c1()
        _, _, Pa, Pb, Pna, Pnb = t.fold(c1)
        _, _, B0, B1, B0nc, B1nc = _collapsed_carrier_fold(
            t.res, t.c0, t.iC, t.nb, t.stride, t.Nt, c1_dense=c1, layout="sym")
        for p in range(3):
            np.testing.assert_allclose(B0[:, :, p], Pa + Pb, rtol=1e-12, atol=1e-12 * np.abs(Pa).max())
            np.testing.assert_allclose(B1[:, :, p], Pb, rtol=1e-12, atol=1e-12 * np.abs(Pa).max())
            np.testing.assert_allclose(B0nc[:, :, p], Pna + Pnb, rtol=1e-12,
                                       atol=1e-12 * np.abs(Pna).max())
            np.testing.assert_allclose(B1nc[:, :, p], Pnb, rtol=1e-12, atol=1e-12 * np.abs(Pna).max())


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
        cls.invCs_un = np.stack([_unequal_invc(rng, 1, s.shape[1], s.shape[2])[0] for s in slabs])
        cls.holder_un = _SlotHolder(cls.slabs, cls.invCs_un)
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

    def _scores(self, comp, holder=None, layout="collapsed", zero_q2=False, c1="0"):
        with _Env(SIGHET_CARRIER_LAYOUT=layout, SIGHET_ANCHOR_CORRECT="0",
                  SIGHET_CARRIER_C1=c1):
            comp.setup_in_model(self.holder if holder is None else holder, self.params_ref,
                                self.slots)
        try:
            if zero_q2:
                for arr in (comp.B0_all, comp.B1_all, comp.B0nc_all, comp.B1nc_all):
                    arr[:, 2] = 0.0
            comp.get_ll(self.params, data_index=self.di)
            return (comp._stash_layout, np.asarray(comp.last_d_h).copy(),
                    np.asarray(comp.last_h_h).copy())
        finally:
            comp.clear_in_model()

    def test_collapsed_without_q2_equals_full_layout(self):
        comp = self._comp()
        c_full, dh_f, hh_f = self._scores(comp, layout="full")
        c_col, dh_c, hh_c = self._scores(comp, layout="collapsed", zero_q2=True)
        self.assertEqual((c_full, c_col), ("full", "collapsed"))
        np.testing.assert_allclose(dh_c, dh_f, rtol=1e-11, atol=1e-11 * np.abs(dh_f).max())
        np.testing.assert_allclose(hh_c, hh_f, rtol=1e-11, atol=1e-11 * np.abs(hh_f).max())

    def test_q2_moments_reach_the_score(self):
        comp = self._comp()
        _, dh0, hh0 = self._scores(comp, layout="collapsed", zero_q2=True)
        _, dh2, hh2 = self._scores(comp, layout="collapsed")
        np.testing.assert_array_equal(dh2, dh0)          # <d|h> has no q = 2 term
        self.assertGreater(np.min(hh2 - hh0), 0.0,
                           "conj(dr) dr n_off^2 is a positive-definite addition to <h|h>")

    def test_c1_makes_the_anchor_exact(self):
        """At the references (rows 0, 1) the c1 fold matches the exact chunked ln L ~100x
        better than without it (mutation: SIGHET_CARRIER_C1=0 brings the offset back)."""
        comp = self._comp()
        ex = np.asarray(self.chunked.get_ll_wdm(self.params, self.holder, data_index=self.di,
                                                noise_index=self.di)).ravel()
        err = {}
        for c1 in ("0", "1"):
            with _Env(SIGHET_ANCHOR_CORRECT="0", SIGHET_CARRIER_C1=c1):
                comp.setup_in_model(self.holder, self.params_ref, self.slots)
            try:
                self.assertEqual(comp._stash_c1, c1 == "1")
                ll = np.asarray(comp.get_ll(self.params, data_index=self.di)).ravel()
            finally:
                comp.clear_in_model()
            err[c1] = np.abs(ll - ex)[:2] / np.abs(ex[:2])
        self.assertGreater(err["0"].min(), 1e-8, f"no-c1 anchor {err['0']}")
        self.assertLess(err["1"].max(), err["0"].min() / 100.0,
                        f"c1 anchor {err['1']} vs without {err['0']}")

    def test_one_channel_carrier_build(self):
        """The collapsed setup's carrier producer (gb_signal_het_make_reference_carrier)
        emits channel 0 of c0 and the dW/dj (c1) transform from ONE FD build: it must equal
        every channel of the 3-channel build with each window (carrier = channel-free)."""
        from gbgpu.gbsignalhetcomputations import _n_cp_kernel_arg
        comp = self._comp()
        g = comp._g
        win, dwin = np.asarray(comp.window_full), np.asarray(_window_dj(comp.window_full, np))
        common = (np.asarray(comp.n_sparse_local), np.zeros(2, dtype=np.int32),
                  np.ascontiguousarray(self.params_ref), 2, 9, 1, 2,
                  g["Nf"], g["Nt"], g["Nf_active"], g["Nt_active"], g["nt_layer"],
                  g["N_sparse_t"], g["stride"], g["ind_min_t"], g["ind_min_f"],
                  g["layer_df"], g["dt"], g["Tobs"], g["t0"], 3, g["n_sparse_fd"],
                  g["tukey_alpha"], _n_cp_kernel_arg(g, allow_carrier=True))
        full = {}
        for name, w in (("c0", win), ("c1", dwin)):
            sp = np.zeros((2, 3, g["Nf_active"], g["N_sparse_t"]), dtype=np.complex128)
            de = np.zeros((2, 3, g["Nf_active"], g["Nt_active"]), dtype=np.complex128)
            comp.cpp.gb_signal_het_make_reference(comp.tdi_wrap, sp, de, w, *common)
            full[name] = (sp, de)
        sp1 = np.zeros((2, 1, g["Nf_active"], g["N_sparse_t"]), dtype=np.complex128)
        de1 = np.zeros((2, 1, g["Nf_active"], g["Nt_active"]), dtype=np.complex128)
        c11 = np.zeros_like(de1)
        comp.cpp.gb_signal_het_make_reference_carrier(comp.tdi_wrap, sp1, de1, c11, win, dwin,
                                                      *common)
        self.assertGreater(np.abs(de1).max(), 0.0)
        self.assertGreater(np.abs(c11).max(), 0.0)
        for c in range(3):
            np.testing.assert_array_equal(full["c0"][1][:, c], de1[:, 0], err_msg=f"c0 dense ch {c}")
            np.testing.assert_array_equal(full["c0"][0][:, c], sp1[:, 0], err_msg=f"c0 sparse ch {c}")
            np.testing.assert_allclose(full["c1"][1][:, c], c11[:, 0], rtol=0,
                                       atol=1e-14 * np.abs(c11).max(), err_msg=f"c1 ch {c}")

    def _scores_layout(self, comp, holder, zero_q2=False, **env):
        e = dict(SIGHET_ANCHOR_CORRECT="0")
        e.update(env)
        with _Env(**e):
            comp.setup_in_model(holder, self.params_ref, self.slots)
        try:
            if zero_q2:
                for arr in (comp.B0_all, comp.B1_all, comp.B0nc_all, comp.B1nc_all):
                    arr[:, 2] = 0.0
            comp.get_ll(self.params, data_index=self.di)
            return (comp._stash_layout, np.asarray(comp.last_d_h).copy(),
                    np.asarray(comp.last_h_h).copy())
        finally:
            comp.clear_in_model()

    def test_unequal_arm_takes_the_sym_layout(self):
        """Unequal-arm noise: SYM layout, and with q = 2 and c1 off it IS the full layout."""
        comp = self._comp()
        lay_f, dh_f, hh_f = self._scores_layout(comp, self.holder_un, SIGHET_CARRIER_COLLAPSE="0")
        lay_s, dh_s, hh_s = self._scores_layout(comp, self.holder_un, zero_q2=True,
                                                SIGHET_CARRIER_C1="0")
        self.assertEqual((lay_f, lay_s), ("full", "sym"))
        np.testing.assert_allclose(dh_s, dh_f, rtol=1e-11, atol=1e-11 * np.abs(dh_f).max())
        np.testing.assert_allclose(hh_s, hh_f, rtol=1e-11, atol=1e-11 * np.abs(hh_f).max())

    def test_sym_equals_collapsed_compiled(self):
        """On equal-arm noise the SYM kernel path reproduces the collapsed one (same model)."""
        comp = self._comp()
        lay_c, dh_c, hh_c = self._scores_layout(comp, self.holder,
                                                SIGHET_CARRIER_LAYOUT="collapsed")
        lay_s, dh_s, hh_s = self._scores_layout(comp, self.holder)    # default: sym
        self.assertEqual((lay_c, lay_s), ("collapsed", "sym"))
        np.testing.assert_allclose(dh_s, dh_c, rtol=1e-12, atol=1e-12 * np.abs(dh_c).max())
        np.testing.assert_allclose(hh_s, hh_c, rtol=1e-11, atol=1e-11 * np.abs(hh_c).max())

    def test_sym_anchor_exact_with_c1(self):
        """Unequal-arm noise: the SYM fold's reference ln L matches the exact chunked one
        ~100x better than the full layout (no second moment, no c1)."""
        comp = self._comp()
        ex = np.asarray(self.chunked.get_ll_wdm(self.params, self.holder_un, data_index=self.di,
                                                noise_index=self.di)).ravel()
        err = {}
        for name, env in (("full", dict(SIGHET_CARRIER_COLLAPSE="0")), ("sym", {})):
            e = dict(SIGHET_ANCHOR_CORRECT="0")
            e.update(env)
            with _Env(**e):
                comp.setup_in_model(self.holder_un, self.params_ref, self.slots)
            try:
                self.assertEqual(comp._stash_layout, name)
                ll = np.asarray(comp.get_ll(self.params, data_index=self.di)).ravel()
            finally:
                comp.clear_in_model()
            err[name] = np.abs(ll - ex)[:2] / np.abs(ex[:2])
        self.assertLess(err["sym"].max(), err["full"].min() / 100.0,
                        f"sym anchor {err['sym']} vs full {err['full']}")

    def test_default_layout_is_sym(self):
        """SYM is the default on any channel-symmetric noise, equal-arm included (the
        collapsed layout is opt-in only)."""
        comp = self._comp()
        self.assertEqual(self._scores_layout(comp, self.holder)[0], "sym")
        self.assertEqual(self._scores_layout(comp, self.holder_un)[0], "sym")
        self.assertEqual(self._scores_layout(comp, self.holder_un,
                                             SIGHET_CARRIER_LAYOUT="collapsed")[0], "sym")

    def test_asymmetric_buffer_falls_back_loudly(self):
        bad = self.invCs.copy()
        bad[:, 0, 1] *= 1.2                       # iC_01 != iC_10: not even pair-symmetric
        holder = _SlotHolder(self.slabs, bad)
        comp = self._comp()
        with self.assertLogs("gbgpu.gbsignalhetcomputations", level="WARNING") as cm:
            lay, dh, hh = self._scores_layout(comp, holder)
        self.assertEqual(lay, "full")
        self.assertTrue(any("not symmetric" in m for m in cm.output))
        self.assertTrue(np.all(np.isfinite(hh)) and np.all(hh > 0))

    def test_patch_cannot_switch_layout(self):
        bad = self.invCs_un                       # collapsed block -> a SYM patch
        comp = self._comp()
        with _Env(SIGHET_CARRIER_LAYOUT="collapsed", SIGHET_ANCHOR_CORRECT="0"):
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
