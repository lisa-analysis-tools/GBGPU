"""Re/Im control-point representation of the sig-het spline build.

``gbfd_build_one_source`` rebuilds each channel's envelope between ``n_cp``
control points. The ``"ampph"`` representation splines each XYZ channel's
(signed) amplitude and phase; where an envelope passes near zero (edge-on /
pattern-null sources) the phase swings steeply and DIFFERENTLY per channel,
so the rebuilt channels no longer cancel in the low-f X+Y+Z null direction,
which the near-singular low-f XYZ inverse covariance then amplifies. The
``"reim"`` representation splines Re/Im of each channel's envelope
demodulated by the common reference phase: identical at the nodes and LINEAR
in the channel values between them, so the null is interpolated as itself.

Pinned here, all through the real kernels (CPU):

* **Binding level.** ``gb_signal_het_make_reference`` at ``n_cp`` nodes vs
  the DIRECT build (``n_cp = 0``, the exact per-sample evaluation), noise-
  weighted with the low-f XYZ covariance: Re/Im is accurate where amp/phase
  is not, and never worse.
* **Engine level (the production contract).** A v5 engine's anchor ``h_h``
  -- ``setup_in_model`` + ``get_ll`` at the reference, narrow-slab buffer
  layout, full XYZ invC -- against the exact chunked ``h_h``: the amp/phase
  build inflates it for an edge-on low-f source, the Re/Im build does not.
* **v5 node-ratio fit.** The same knob switches the v5 scorer's candidate
  ratio fit from log-polar to Re/Im: delta-vs-delta against the exact
  chunked deltas, the catastrophic log-polar tail is gone and the anchor is
  exact.
* **Knob plumbing.** ``cp_repr`` / ``SIGHET_CP_REPR`` resolution (default
  ``"carrier"`` on v5, ``"reim"`` elsewhere, ``"ampph"`` the rollback) and the
  negative ``n_cp_sig`` / ``n_nodes`` the kernels decode.
* **Second fold moment.** The carrier-mode collapsed stash carries the
  ``conj(dr) dr`` moment the full layout lacks: the uncorrected anchor offset
  drops from ~0.5 lnL to ~1e-4 (median, SNR 100) and the deltas stay inside
  the tier bar.
"""

import os
import unittest

import numpy as np

from lisatools.detector import ESAOrbits
from lisatools.domains import WDMSettings
from lisatools.sensitivity import X2TDISens, XY2TDISens
from lisatools.utils.constants import YRSID_SI

from gbgpu.gbcomps import GBWDMComputations
from gbgpu.gbsignalhetcomputations import (
    GBSignalHetComputations,
    _SIGHET_CARRIER,
    _n_cp_kernel_arg,
    _resolve_cp_repr,
    _v5_nodes_arg,
)

#: Six months on the production WDM layer (3600 s) with a cheap sampling:
#: the effect needs the annual response modulation, i.e. a long Tobs.
DT, NF, NT, EDGE = 20.0, 180, 4320, 30
V5_KNOBS = dict(v3_n_nodes=64, v4_knots=128, v4_band=16, v5=1)

#: Edge-on, low-f sources (the population the amp/phase build corrupts) plus
#: one generic-inclination control.
SOURCES = np.array([
    # amp     f0       fdot   fddot phi0 inc   psi  lam  beta
    [1e-22, 6.3e-4, 1e-18, 0.0, 0.7, 1.55, 0.40, 1.0, 0.30],
    [1e-22, 6.3e-4, 1e-18, 0.0, 2.1, 1.55, 1.30, 4.0, -0.80],
    [1e-22, 1.1e-3, 1e-18, 0.0, 1.2, 1.53, 0.90, 2.5, 0.10],
    [1e-22, 6.3e-4, 1e-18, 0.0, 0.3, 0.70, 0.20, 5.5, 0.60],
])

_FIX = {}


def _fixture():
    if _FIX:
        return _FIX
    t0 = int(0.5 * YRSID_SI / DT) * DT
    orbits = ESAOrbits(force_backend="cpu")
    wdm = WDMSettings(NF, NT, DT, t0=t0, min_freq=1e-4, max_freq=2e-3,
                      min_time=EDGE * NF * DT, max_time=(NT - EDGE) * NF * DT,
                      force_backend="cpu")
    chunked = GBWDMComputations(
        wdm, t_ref=t0, Nt_sub=256, n_pad=32, N_sparse=256,
        N_cp_sig=0, N_cp_orbit=0, orbits=orbits, tdi_config="2nd generation",
        force_backend="cpu", d_d=0.0, tdi_type="XYZ")
    chunked.convert_to_ra_dec = False
    _FIX.update(wdm=wdm, chunked=chunked)
    return _FIX


def _engine(n_cp, cp_repr):
    fx = _fixture()
    return GBSignalHetComputations.for_band_engine(
        fx["chunked"], nt_layer=120, n_sparse_fd=1024, m_active_half_width=2,
        max_r=0.0, n_cp_build=n_cp, tukey_alpha=0.01, cp_repr=cp_repr,
        **V5_KNOBS)


def _xyz_invC(f):
    sxx = float(np.atleast_1d(X2TDISens.get_Sn(f, model="scirdv1"))[0])
    sxy = float(np.atleast_1d(XY2TDISens.get_Sn(f, model="scirdv1"))[0])
    C = np.full((3, 3), sxy)
    np.fill_diagonal(C, sxx)
    return np.linalg.inv(C)


def _make_reference(sig, params9):
    """Dense c0 (3, Nf_active, Nt_active) from the REAL producer."""
    g = sig._g
    c_sp = np.zeros((1, 3, g["Nf_active"], g["N_sparse_t"]), dtype=np.complex128)
    c_de = np.zeros((1, 3, g["Nf_active"], g["Nt_active"]), dtype=np.complex128)
    sig.cpp.gb_signal_het_make_reference(
        sig.tdi_wrap, c_sp, c_de, np.asarray(sig.window_full),
        np.asarray(sig.n_sparse_local), np.zeros(1, dtype=np.int32),
        np.ascontiguousarray(np.asarray(params9, float).reshape(1, 9)),
        1, 9, 1, 2,
        g["Nf"], g["Nt"], g["Nf_active"], g["Nt_active"],
        g["nt_layer"], g["N_sparse_t"], g["stride"],
        g["ind_min_t"], g["ind_min_f"], g["layer_df"], g["dt"],
        g["Tobs"], g["t0"], 3, g["n_sparse_fd"], g["tukey_alpha"],
        _n_cp_kernel_arg(g))
    return c_de[0]


def _rel_err(c, ref, iC):
    """<c - ref | c - ref> / <ref | ref> under a constant XYZ invC."""
    e = c - ref
    num = np.real(np.einsum("cft,cd,dft->", e.conj(), iC, e))
    den = np.real(np.einsum("cft,cd,dft->", ref.conj(), iC, ref))
    return num / den


class CpReprKnobTest(unittest.TestCase):
    def test_resolution_and_sign(self):
        saved = os.environ.pop("SIGHET_CP_REPR", None)
        try:
            self.assertEqual(_resolve_cp_repr(None), "carrier")    # default
            os.environ["SIGHET_CP_REPR"] = "AmpPh"
            self.assertEqual(_resolve_cp_repr(None), "ampph")      # rollback
            self.assertEqual(_resolve_cp_repr("reim"), "reim")     # explicit wins
            with self.assertRaises(ValueError):
                _resolve_cp_repr("polar")
        finally:
            os.environ.pop("SIGHET_CP_REPR", None)
            if saved is not None:
                os.environ["SIGHET_CP_REPR"] = saved
        self.assertEqual(_n_cp_kernel_arg(dict(n_cp_build=64, cp_repr="reim")), -64)
        self.assertEqual(_n_cp_kernel_arg(dict(n_cp_build=64, cp_repr="ampph")), 64)
        self.assertEqual(_n_cp_kernel_arg(dict(n_cp_build=0, cp_repr="reim")), 0)
        self.assertEqual(_n_cp_kernel_arg(dict(n_cp_build=64)), 64)
        # carrier: the reference build takes the carrier code only where allowed
        g = dict(n_cp_build=64, cp_repr="carrier")
        self.assertEqual(_n_cp_kernel_arg(g), -64)
        self.assertEqual(_n_cp_kernel_arg(g, allow_carrier=True), -(64 + _SIGHET_CARRIER))
        self.assertEqual(_v5_nodes_arg(g, 32), -(32 + _SIGHET_CARRIER))
        self.assertEqual(_v5_nodes_arg(dict(cp_repr="reim"), 32), -32)
        self.assertEqual(_v5_nodes_arg(dict(cp_repr="ampph"), 32), 32)

    def test_carrier_is_v5_only(self):
        fx = _fixture()
        with self.assertRaises(ValueError):
            GBSignalHetComputations.for_band_engine(
                fx["chunked"], nt_layer=120, n_cp_build=32, tukey_alpha=0.01,
                cp_repr="carrier")                       # explicit, v2 engine
        saved = os.environ.pop("SIGHET_CP_REPR", None)
        try:   # the DEFAULT on a non-v5 engine falls back to reim
            sig = GBSignalHetComputations.for_band_engine(
                fx["chunked"], nt_layer=120, n_cp_build=32, tukey_alpha=0.01)
            self.assertEqual(sig._g["cp_repr"], "reim")
        finally:
            if saved is not None:
                os.environ["SIGHET_CP_REPR"] = saved

    def test_engine_records_repr(self):
        self.assertEqual(_engine(32, "reim")._g["cp_repr"], "reim")
        self.assertEqual(_engine(32, "ampph")._g["cp_repr"], "ampph")


class MakeReferenceReImTest(unittest.TestCase):
    """Binding level: the build at n_cp nodes vs the DIRECT build."""

    @classmethod
    def setUpClass(cls):
        direct = _engine(0, "ampph")
        cls.err = {}
        for n_cp in (32, 256):
            for rep in ("ampph", "reim"):
                sig = _engine(n_cp, rep)
                cls.err[(n_cp, rep)] = [
                    _rel_err(_make_reference(sig, p), _make_reference(direct, p),
                             _xyz_invC(p[1]))
                    for p in SOURCES]

    def test_reim_matches_direct(self):
        for n_cp in (32, 256):
            worst = max(self.err[(n_cp, "reim")])
            self.assertLess(worst, 1e-6, f"n_cp={n_cp}: Re/Im build err {worst:.2e}")

    def test_reim_beats_ampph_on_edge_on_low_f(self):
        a, r = max(self.err[(32, "ampph")]), max(self.err[(32, "reim")])
        self.assertGreater(a, 1e-3, f"amp/phase err {a:.2e} (the defect should show)")
        self.assertLess(r * 1e3, a, f"Re/Im {r:.2e} vs amp/phase {a:.2e}")

    def test_reim_never_worse(self):
        for n_cp in (32, 256):
            for a, r in zip(self.err[(n_cp, "ampph")], self.err[(n_cp, "reim")]):
                self.assertLessEqual(r, max(a, 1e-9) * 1.5)


class _Slabs:
    """Narrow per-slot slab holder (the production buffer layout)."""

    def __init__(self, data, invc, slab_min_f, W):
        self.linear_data_arr = [np.ascontiguousarray(data, dtype=np.float64).ravel()]
        self.linear_psd_arr = [np.ascontiguousarray(invc, dtype=np.float64).ravel()]
        self.band_slab_Nf = int(W)
        self.slab_min_f = np.asarray(slab_min_f, dtype=np.int32)
        self.min_freq_inds = self.slab_min_f
        self._n = len(data)

    def __len__(self):
        return self._n


class EngineAnchorHhTest(unittest.TestCase):
    """The v5 engine's anchor h_h against the exact chunked h_h."""

    @classmethod
    def setUpClass(cls):
        from gbgpu.gb_likelihood import make_band_likelihood_engine

        fx = _fixture()
        wdm = fx["wdm"]
        W = 5
        n = len(SOURCES)
        T = int(wdm.ind_max_t - wdm.ind_min_t + 1)
        m_car = np.floor(SOURCES[:, 1] / wdm.layer_df).astype(int)
        slab_lo = (m_car - W // 2).astype(np.int32)
        invc = np.empty((n, 3, 3, W, T))
        for i in range(n):
            for j in range(W):
                invc[i, :, :, j, :] = _xyz_invC((slab_lo[i] + j) * wdm.layer_df)[:, :, None]
        holder = _Slabs(np.zeros((n, 3, W, T)), invc, slab_lo, W)
        idx = np.arange(n)
        NV = np.full(n, 1024)
        cls.ratio = {}
        for rep in ("ampph", "reim"):
            eng = make_band_likelihood_engine(
                wdm, gb_wdm_comp=_engine(32, rep), nchannels=3,
                tdi_channel_setup="XYZ")
            eng.setup_in_model(holder, SOURCES, idx)
            eng.get_ll(holder, SOURCES, data_index=idx, noise_index=idx,
                       N_vals=NV, waveform_kwargs={})
            hh_sig = np.asarray(eng.h_h_out).real.ravel()[:n].copy()
            eng.clear_in_model()
            eng.get_ll(holder, SOURCES, data_index=idx, noise_index=idx,
                       N_vals=NV, waveform_kwargs={})
            hh_ex = np.asarray(eng.h_h_out).real.ravel()[:n].copy()
            cls.ratio[rep] = hh_sig / hh_ex

    def test_reim_anchor_exact(self):
        worst = np.abs(np.log(self.ratio["reim"])).max()
        self.assertLess(worst, 1e-4, f"Re/Im anchor |log hh ratio| {worst:.2e}")

    def test_ampph_anchor_inflated(self):
        worst = np.abs(np.log(self.ratio["ampph"])).max()
        self.assertGreater(worst, 1e-3, f"amp/phase anchor |log hh ratio| {worst:.2e}")


class V5RatioDeltaTest(unittest.TestCase):
    """The v5 scorer's candidate NODE-RATIO fit, through the real engine.

    Each slot's slab holds its reference source (exact chunked fill, SNR
    ~100); displaced candidates are scored delta-vs-delta against the exact
    chunked deltas. The log-polar fit exp()-amplifies spline overshoot near
    envelope minima and flip spans; the Re/Im fit has no exp() and returns
    r == 1 at the reference.
    """

    @classmethod
    def setUpClass(cls):
        from gbgpu.gb_likelihood import make_band_likelihood_engine

        fx = _fixture()
        wdm = fx["wdm"]
        W = 5
        rng = np.random.default_rng(20261003)
        n_src = 12
        f0 = rng.choice([6.3e-4, 1.1e-3, 1.7e-3], n_src)
        cosi = np.concatenate([np.full(6, 0.02), rng.uniform(0.1, 0.4, 6)])
        p0 = np.column_stack([
            np.full(n_src, 1e-22), f0, np.full(n_src, 1e-18), np.zeros(n_src),
            rng.uniform(0, 2 * np.pi, n_src), np.arccos(cosi),
            rng.uniform(0, np.pi, n_src), rng.uniform(0, 2 * np.pi, n_src),
            np.arcsin(rng.uniform(-1, 1, n_src))])
        T = int(wdm.ind_max_t - wdm.ind_min_t + 1)
        slab_lo = (np.floor(f0 / wdm.layer_df).astype(int) - W // 2).astype(np.int32)
        invc = np.empty((n_src, 3, 3, W, T))
        for i in range(n_src):
            for j in range(W):
                invc[i, :, :, j, :] = _xyz_invC((slab_lo[i] + j) * wdm.layer_df)[:, :, None]
        idx = np.arange(n_src)
        NV = np.full(n_src, 1024)
        eng = {rep: make_band_likelihood_engine(
            wdm, gb_wdm_comp=_engine(256, rep), nchannels=3, tdi_channel_setup="XYZ")
            for rep in ("ampph", "reim", "carrier")}
        e0 = eng["ampph"]
        zero = _Slabs(np.zeros((n_src, 3, W, T)), invc, slab_lo, W)
        e0.get_ll(zero, p0, data_index=idx, noise_index=idx, N_vals=NV, waveform_kwargs={})
        p0[:, 0] *= 100.0 / np.sqrt(np.asarray(e0.h_h_out).real.ravel()[:n_src])
        data = _Slabs(np.zeros((n_src, 3, W, T)), invc, slab_lo, W)
        e0.fill_template(data, p0, idx, NV, factor=+1, waveform_kwargs={},
                         band_slab_Nf=W, slab_min_f=slab_lo)
        cands = []
        for _ in range(3):
            p1 = p0.copy()
            p1[:, 0] *= np.exp(0.05 * rng.standard_normal(n_src))
            p1[:, 1] += 0.05 / wdm.Tobs * rng.standard_normal(n_src)
            p1[:, 4] += 0.1 * rng.standard_normal(n_src)
            p1[:, 5] = np.clip(p1[:, 5] + 0.03 * rng.standard_normal(n_src), 1e-3, np.pi - 1e-3)
            p1[:, 6] += 0.03 * rng.standard_normal(n_src)
            p1[:, 7] += 0.01 * rng.standard_normal(n_src)
            p1[:, 8] = np.clip(p1[:, 8] + 0.01 * rng.standard_normal(n_src), -1.5, 1.5)
            cands.append(p1)

        def ll(e, P):
            out = e.get_ll(data, P, data_index=idx, noise_index=idx, N_vals=NV,
                           waveform_kwargs={})
            return np.asarray(out, dtype=float).ravel()[:n_src]

        le0 = ll(e0, p0)                       # exact (no reference active)
        D_ex = np.array([ll(e0, p1) - le0 for p1 in cands])
        cls.eps, cls.anchor = {}, {}
        # carrier_nocorr: the collapsed stash WITH the second fold moment, no
        # correction; carrier_full_nocorr: the full nch x nch layout, which has
        # no second moment -- the pre-B2 fold.
        runs = list(eng.items()) + [
            ("carrier_nocorr", eng["carrier"], dict(SIGHET_ANCHOR_CORRECT="0")),
            ("carrier_full_nocorr", eng["carrier"],
             dict(SIGHET_ANCHOR_CORRECT="0", SIGHET_CARRIER_COLLAPSE="0"))]
        for run in runs:
            rep, e = run[:2]
            env = run[2] if len(run) > 2 else {}
            saved = {k: os.environ.get(k) for k in env}
            os.environ.update(env)
            try:
                e.setup_in_model(data, p0, idx)
                ls0 = ll(e, p0)
                D = np.array([ll(e, p1) - ls0 for p1 in cands])
                e.clear_in_model()
            finally:
                for k, v in saved.items():
                    if v is None:
                        os.environ.pop(k, None)
                    else:
                        os.environ[k] = v
            cls.eps[rep] = np.abs(D - D_ex)
            cls.anchor[rep] = np.abs(ls0 - le0)
        cls.T = np.abs(D_ex)
        cls.edge = np.abs(np.cos(p0[:, 5])) < 0.05

    def test_reim_anchor_exact(self):
        worst = self.anchor["reim"].max()
        self.assertLess(worst, 1e-2, f"Re/Im v5 anchor offset {worst:.2e} lnL")

    def test_reim_kills_the_catastrophic_tail(self):
        a, r = self.eps["ampph"].max(), self.eps["reim"].max()
        self.assertGreater(a, 1e3, f"log-polar worst eps {a:.2e} (the defect should show)")
        self.assertLess(r, 1e-2 * a, f"Re/Im worst eps {r:.2e} vs log-polar {a:.2e}")

    def test_reim_not_worse_in_bulk(self):
        a, r = np.median(self.eps["ampph"]), np.median(self.eps["reim"])
        self.assertLessEqual(r, 1.5 * a + 1e-6, f"median eps Re/Im {r:.2e} vs {a:.2e}")

    def test_carrier_fixes_edge_on(self):
        """Carrier-only reference: every delta inside the tier bar, edge-on included,
        where the per-channel Re/Im ratio still fails (its error sits in X+Y+Z)."""
        e = self.eps["carrier"]
        bar = np.maximum(0.1, self.T / 100.0)
        self.assertTrue(np.all(e <= bar), f"carrier worst eps/bar {np.max(e / bar):.2f}")
        r = self.eps["reim"][:, self.edge]
        self.assertGreater(np.max(r / bar[:, self.edge]), 3.0,
                           "the per-channel Re/Im ratio should still fail edge-on here")

    def test_carrier_anchor_corrected(self):
        self.assertLess(self.anchor["carrier"].max(), 1e-8,
                        f"carrier anchor {self.anchor['carrier'].max():.2e} with the correction")

    def test_second_moment_removes_the_anchor(self):
        """The pre-B2 fold (full layout) is off by ~0.5 lnL at the reference at SNR 100;
        the second moment takes the uncorrected offset down by >~ 30x."""
        full, coll = self.anchor["carrier_full_nocorr"], self.anchor["carrier_nocorr"]
        self.assertGreater(np.median(full), 0.2, f"pre-B2 anchor median {np.median(full):.2e}")
        self.assertLess(coll.max(), 0.1, f"B2 anchor max {coll.max():.2e}")
        self.assertLess(np.median(coll), np.median(full) / 30.0,
                        f"B2 anchor median {np.median(coll):.2e} vs {np.median(full):.2e}")

    def test_second_moment_keeps_the_deltas(self):
        e = self.eps["carrier_nocorr"]
        bar = np.maximum(0.1, self.T / 100.0)
        self.assertTrue(np.all(e <= bar), f"B2 worst eps/bar {np.max(e / bar):.2f}")


if __name__ == "__main__":
    unittest.main()
