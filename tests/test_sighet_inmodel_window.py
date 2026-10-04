"""Compact-window in-model stash: the v5 ``(W_slab, w_lo_arr)`` contract.

``setup_in_model`` builds each source's heterodyne reference in a narrow
layer window and used to SCATTER the result into full-band arrays spanning
every active layer, purely so the scorer could keep a single
``Nf_active``-strided index. At production settings that made ~97% of the
seven coefficient arrays exact zeros (measured W = 5 of Nf_active = 179).
The v5 scorer now takes the window width and the per-reference
active-local origins directly -- the same contract the F-stat scorer has
carried since it was written -- and the stash stays compact.

What is pinned here:

* **Degenerate case is bit-identical.** ``W_slab == Nf_active`` with
  all-zero ``w_lo`` must reproduce the pre-window full-band indexing
  EXACTLY (``assert_array_equal``, no tolerance): that call IS the old
  call. Asserted at the binding, against a stash expanded in the test.
* **Windowed == full-band.** The real narrow-W path must agree with the
  full-band expansion of the same references. This is the test that
  actually proves the port; it is also exact, because the full-band stash
  was identically zero off the window and skipping exact zeros cannot
  change a sum that starts at +0.0.
* **The mid-block refresh.** ``setup_in_model`` is incremental: mid-block
  it re-anchors a SUBSET of slots, and a refreshed source's window can
  MOVE. The compact layout only stays correct if that source's ``w_lo``
  moves with it, so the refresh is exercised with a subset whose window
  demonstrably shifts, and compared against the full-band twin.
* **The stash is RELEASED on clear.** ``clear_in_model`` runs after every
  repeat block; holding the block's coefficient arrays until the next
  ``setup_in_model`` reassigns them kept the run's largest transient
  resident and doubled the peak at the next build.

CPU-only (numpy), same small grid as ``test_sighet_infomat`` /
``test_phase_max_fused``.
"""

import os
import unittest
import weakref

import numpy as np

from lisatools.detector import ESAOrbits
from lisatools.domains import WDMSettings
from lisatools.utils.constants import YRSID_SI

from gbgpu.gbcomps import GBWDMComputations
from gbgpu.gbsignalhetcomputations import (
    GBSignalHetComputations,
    _V5_COLLAPSED,
    _v5_nodes_arg,
)

#: v5 knobs. v5 is the only in-model scorer that takes the windowed
#: contract (and the only one production runs use), so every windowed comp
#: here is built with them.
V5_KNOBS = dict(v3_n_nodes=32, v4_knots=64, v4_band=16, v5=1)


class _SlotHolder:
    """Minimal wdm_holder: N buffer slots (residual slab + XYZ invC slab).

    No ``band_slab_Nf``, so ``setup_in_model`` takes the FULL-BAND buffer
    branch: a carrier-centred window per reference, which is what lets the
    refresh test move a window by moving a reference's f0.
    """

    def __init__(self, data_slabs, invC_slabs):
        self.linear_data_arr = [np.ascontiguousarray(data_slabs).ravel()]
        self.linear_psd_arr = [np.ascontiguousarray(invC_slabs).ravel()]

    def __len__(self):
        return 1


class _NarrowSlabHolder(_SlotHolder):
    """The PRODUCTION buffer layout: per-slot narrow band slabs.

    ``band_slab_Nf`` + ``slab_min_f`` send ``setup_in_model`` down the
    narrow-slab branch, where the fold window IS the slab -- so the window
    is NOT carrier-centred and a candidate's active m-band routinely runs
    off its edges. That is the case the compact stash has to get right, and
    the one where the F-stat scorer's edge-CLAMP would be wrong (the slab's
    edge rows carry real, nonzero coefficients).
    """

    def __init__(self, data_slabs, invC_slabs, slab_Nf, slab_min_f):
        super().__init__(data_slabs, invC_slabs)
        self.band_slab_Nf = int(slab_Nf)
        self.slab_min_f = np.asarray(slab_min_f, dtype=np.int32)


def _expand_A(vals, w_lo, Nf_active):
    """(n, nch, W, Nsp) compact -> (n, nch, Nf_active, Nsp) full band."""
    n, nch, W, nsp = vals.shape
    out = np.zeros((n, nch, Nf_active, nsp), dtype=vals.dtype)
    for i in range(n):
        lo = int(w_lo[i])
        out[i, :, lo:lo + W, :] = vals[i]
    return np.ascontiguousarray(out)


def _expand_B(vals, w_lo, Nf_active):
    """(n, nch, nch, W, Nsp) compact -> (n, nch, nch, Nf_active, Nsp).
    The carrier-mode COLLAPSED moments are (n, 3 [q], W, Nsp): A-shaped."""
    if vals.ndim == 4:
        return _expand_A(vals, w_lo, Nf_active)
    n, nch, nch2, W, nsp = vals.shape
    out = np.zeros((n, nch, nch2, Nf_active, nsp), dtype=vals.dtype)
    for i in range(n):
        lo = int(w_lo[i])
        out[i, :, :, lo:lo + W, :] = vals[i]
    return np.ascontiguousarray(out)


#: Built ONCE for the whole module. The grid costs a chunked
#: ``fill_global_wdm`` per source and the suite has several classes; on the
#: 8 GB dev laptop rebuilding it per class is the difference between a
#: quick run and an OOM kill.
_FIXTURE = {}


class _Grid(unittest.TestCase):
    """Shared small-grid scaffolding."""

    @classmethod
    def setUpClass(cls):
        if _FIXTURE:
            for k, v in _FIXTURE.items():
                setattr(cls, k, v)
            return
        cls._build_grid()
        for k in ("layer_df", "wdm_set", "chunked", "params_ref", "holder",
                  "slots", "params", "di", "slab_Nf", "slab_min_f",
                  "slab_holder"):
            _FIXTURE[k] = getattr(cls, k)

    @classmethod
    def _build_grid(cls):
        backend = "cpu"
        dt = 10.0
        Nf, Nt = 256, 512
        t_start = int(0.5 * YRSID_SI / dt) * dt
        cls.layer_df = layer_df = 1.0 / (2.0 * Nf * dt)
        edge = 40

        orbits = ESAOrbits(force_backend=backend)
        cls.wdm_set = wdm_set = WDMSettings(
            Nf, Nt, dt, t0=t_start,
            min_freq=1e-4, max_freq=2e-2,
            min_time=edge * Nf * dt, max_time=(Nt - edge) * Nf * dt,
            force_backend=backend,
        )
        cls.chunked = chunked = GBWDMComputations(
            wdm_set, t_ref=t_start,
            Nt_sub=128, n_pad=16, N_sparse=256,
            N_cp_sig=0, N_cp_orbit=0,
            orbits=orbits, tdi_config="2nd generation",
            force_backend=backend, d_d=0.0, tdi_type="XYZ",
        )
        chunked.convert_to_ra_dec = False

        # Two picked sources in separate buffer slots, far apart in f0 so
        # their windows are disjoint and a slot/window mix-up is visible.
        f0_A = (int(3e-3 / layer_df) + 0.37) * layer_df
        f0_C = (int(5e-3 / layer_df) + 0.62) * layer_df
        A = np.array([1e-21, f0_A, 1e-17, 0.0, 1.2, 0.7, 0.4, 2.0, 0.5])
        C = np.array([8e-22, f0_C, 2e-17, 0.0, 0.4, 1.1, 0.9, 4.0, -0.3])
        cls.params_ref = np.stack([A, C])

        ilo, ihi = wdm_set.ind_min_f, wdm_set.ind_max_f + 1
        slabs, invCs = [], []
        for p in (A, C):
            h = np.zeros((3, Nf, Nt))
            chunked.fill_global_wdm(p[None, :], h, convert_to_ra_dec=False)
            h_act = np.ascontiguousarray(h[:, ilo:ihi, wdm_set.active_slice_t])
            slabs.append(h_act)
            nch, nfa, nta = h_act.shape
            invC = np.zeros((nch, nch, nfa, nta))
            for c in range(nch):
                invC[c, c] = 1.0
            invCs.append(invC)
        cls.holder = _SlotHolder(np.stack(slabs), np.stack(invCs))
        cls.slots = np.array([0, 1], dtype=np.int32)

        # ---- narrow-slab twin (the production buffer layout) -------------
        # W = 5, matching the measured production band_slab_Nf. Slot 0's
        # slab is CENTRED on its carrier; slot 1's is deliberately offset so
        # the candidate's active m-band (carrier +- m_half = 2) runs off the
        # slab's right edge -- i.e. the off-window rows the compact stash
        # must skip and the full-band stash held as exact zeros.
        cls.slab_Nf = slab_Nf = 5
        m_car = np.floor(cls.params_ref[:, 1] / layer_df).astype(int)
        slab_min_f = np.array([m_car[0] - 2, m_car[1] - 4], dtype=np.int32)
        cls.slab_min_f = slab_min_f
        lo = slab_min_f - int(wdm_set.ind_min_f)
        assert (lo >= 0).all() and (lo + slab_Nf <= slabs[0].shape[1]).all()
        cls.slab_holder = _NarrowSlabHolder(
            np.stack([s[:, lo[i]:lo[i] + slab_Nf, :]
                      for i, s in enumerate(slabs)]),
            np.stack([c[:, :, lo[i]:lo[i] + slab_Nf, :]
                      for i, c in enumerate(invCs)]),
            slab_Nf, slab_min_f)

        # Scoring batch: the references plus jittered copies (generic
        # phases/amplitudes, inside the heterodyne validity range).
        rng = np.random.default_rng(20260909)
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

    # ---- helpers -----------------------------------------------------

    @staticmethod
    def _build(chunked, windowed, **knobs):
        """A comp whose in-model stash layout is FORCED either way."""
        saved = os.environ.get("GB_SIGHET_INMODEL_WINDOWED")
        os.environ["GB_SIGHET_INMODEL_WINDOWED"] = "1" if windowed else "0"
        try:
            return GBSignalHetComputations.for_band_engine(chunked, **knobs)
        finally:
            if saved is None:
                os.environ.pop("GB_SIGHET_INMODEL_WINDOWED", None)
            else:
                os.environ["GB_SIGHET_INMODEL_WINDOWED"] = saved

    def _setup(self, comp, windowed, params, slots, holder=None):
        saved = os.environ.get("GB_SIGHET_INMODEL_WINDOWED")
        os.environ["GB_SIGHET_INMODEL_WINDOWED"] = "1" if windowed else "0"
        try:
            return comp.setup_in_model(
                self.holder if holder is None else holder, params, slots)
        finally:
            if saved is None:
                os.environ.pop("GB_SIGHET_INMODEL_WINDOWED", None)
            else:
                os.environ["GB_SIGHET_INMODEL_WINDOWED"] = saved


class DegenerateFullBandTest(_Grid):
    """W_slab == Nf_active + all-zero w_lo == the pre-window call, exactly."""

    def test_degenerate_window_is_bit_identical_to_full_band(self):
        # Production layout (W = 5 band slabs), so the comparison also
        # covers active rows that fall OFF the window.
        comp = self._build(self.chunked, True, **V5_KNOBS)
        self._setup(comp, True, self.params_ref, self.slots,
                    holder=self.slab_holder)
        try:
            g = comp._g
            Nf_active = int(g["Nf_active"])
            W = int(comp._stash_W)
            self.assertLess(W, Nf_active,
                            "fixture must actually window, else this test "
                            "compares a stash against itself")
            w_lo = np.asarray(comp._stash_w_lo)

            # The stash the OLD code would have handed the kernel.
            mask_full = _expand_A(np.asarray(comp.c0_mask_all), w_lo,
                                  Nf_active)
            full = dict(
                A0=_expand_A(np.asarray(comp.A0_all), w_lo, Nf_active),
                A1=_expand_A(np.asarray(comp.A1_all), w_lo, Nf_active),
                B0=_expand_B(np.asarray(comp.B0_all), w_lo, Nf_active),
                B1=_expand_B(np.asarray(comp.B1_all), w_lo, Nf_active),
                B0nc=_expand_B(np.asarray(comp.B0nc_all), w_lo, Nf_active),
                B1nc=_expand_B(np.asarray(comp.B1nc_all), w_lo, Nf_active),
            )

            win = self._score(comp, W, w_lo.astype(np.int32),
                              comp.c0_mask_all, comp.A0_all, comp.A1_all,
                              comp.B0_all, comp.B1_all, comp.B0nc_all,
                              comp.B1nc_all)
            deg = self._score(
                comp, Nf_active,
                np.zeros(int(comp.params_ref_all.shape[0]), dtype=np.int32),
                mask_full, full["A0"], full["A1"], full["B0"], full["B1"],
                full["B0nc"], full["B1nc"])

            for k in ("d_h", "h_h", "d_h_im"):
                np.testing.assert_array_equal(
                    win[k], deg[k],
                    err_msg=f"{k}: windowed != degenerate full-band stash "
                            "(the compact index path must be exact, not "
                            "merely close)")
            # ...and the reference values must not be trivially zero.
            self.assertGreater(float(np.max(np.abs(deg["h_h"]))), 0.0)
        finally:
            comp.clear_in_model()

    def _score(self, comp, W_slab, w_lo, mask, A0, A1, B0, B1, B0nc, B1nc):
        """One direct binding call at an explicit (W_slab, w_lo)."""
        xp = comp.xp
        g = comp._g
        if comp._v4_band_arrays is None:
            comp._v4_band_arrays = comp._make_v4_band_arrays()
        x = xp.ascontiguousarray(xp.atleast_2d(
            xp.asarray(self.params, dtype=float)))
        di = xp.ascontiguousarray(xp.asarray(self.di, dtype=xp.int32))
        n = x.shape[0]
        d_h = xp.zeros(n, dtype=xp.float64)
        h_h = xp.zeros(n, dtype=xp.float64)
        d_h_im = xp.zeros(n, dtype=xp.float64)
        comp.cpp.gb_signal_het_v5_get_ll(
            comp.tdi_wrap, d_h, h_h, mask,
            A0, A1, B0, B1, B0nc, B1nc,
            comp.n_sparse_local,
            comp._v4_band_arrays[0], comp._v4_band_arrays[1],
            comp._v4_band_arrays[2],
            x, comp.params_ref_all, di,
            xp.ascontiguousarray(xp.asarray(w_lo, dtype=xp.int32)),
            n, int(comp.params_ref_all.shape[0]),
            _v5_nodes_arg(g, comp._resolve_v3_nodes(x, di)), int(g["v4_knots"]),
            9, 1, 2,
            g["Nf"], g["Nt"], g["Nf_active"], int(W_slab), g["Nt_active"],
            g["nt_layer"], g["N_sparse_t"], g["stride"],
            g["ind_min_t"], g["ind_min_f"], g["m_half"],
            g["layer_df"], g["dt"], g["Tobs"], g["t0"],
            3, 0, 1,
            1 | (_V5_COLLAPSED if comp._stash_collapsed else 0), d_h_im)
        return dict(d_h=np.asarray(d_h).copy(), h_h=np.asarray(h_h).copy(),
                    d_h_im=np.asarray(d_h_im).copy())


class WindowedVsFullBandTest(_Grid):
    """The port's real gate: narrow-W stash == full-band expansion."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.win = cls._build(cls.chunked, True, **V5_KNOBS)
        cls.full = cls._build(cls.chunked, False, **V5_KNOBS)

    @classmethod
    def tearDownClass(cls):
        cls.win.clear_in_model()
        cls.full.clear_in_model()

    def setUp(self):
        self._setup(self.win, True, self.params_ref, self.slots)
        self._setup(self.full, False, self.params_ref, self.slots)

    def tearDown(self):
        self.win.clear_in_model()
        self.full.clear_in_model()

    def test_layouts_actually_differ(self):
        """Guard against a vacuous comparison."""
        Nf_active = int(self.win._g["Nf_active"])
        self.assertEqual(int(self.full._stash_W), Nf_active)
        self.assertLess(int(self.win._stash_W), Nf_active)
        self.assertEqual(self.full.A0_all.shape[2], Nf_active)
        self.assertEqual(self.win.A0_all.shape[2], int(self.win._stash_W))
        # (n, nch, nch, W, Ns), or (n, 3 [q], W, Ns) when carrier-collapsed
        self.assertEqual(self.win.B0_all.shape[-2], int(self.win._stash_W))
        # the windowed origins are the real per-reference offsets
        self.assertTrue(bool(np.any(np.asarray(self.win._stash_w_lo) > 0)))
        np.testing.assert_array_equal(np.asarray(self.full._stash_w_lo), 0)

    def test_windowed_matches_full_band(self):
        ll_w = np.asarray(self.win.get_ll(self.params, data_index=self.di))
        ll_f = np.asarray(self.full.get_ll(self.params, data_index=self.di))
        np.testing.assert_array_equal(
            ll_w, ll_f,
            err_msg="windowed stash != full-band stash (get_ll)")
        np.testing.assert_array_equal(np.asarray(self.win.last_d_h),
                                      np.asarray(self.full.last_d_h))
        np.testing.assert_array_equal(np.asarray(self.win.last_h_h),
                                      np.asarray(self.full.last_h_h))
        np.testing.assert_array_equal(np.asarray(self.win.last_d_h_im),
                                      np.asarray(self.full.last_d_h_im))
        self.assertGreater(float(np.max(np.abs(np.asarray(ll_f)))), 0.0)

    def test_windowed_matches_full_band_phase_maximized(self):
        ll_w = np.asarray(self.win.get_ll(self.params, data_index=self.di,
                                          phase_maximize=True))
        ll_f = np.asarray(self.full.get_ll(self.params, data_index=self.di,
                                           phase_maximize=True))
        np.testing.assert_array_equal(ll_w, ll_f)

    def test_windowed_matches_full_band_through_get_ll_wdm(self):
        """The engine-facing route (slot -> reference mapping) too."""
        ll_w = np.asarray(self.win.get_ll_wdm(
            self.params, self.holder, data_index=self.di, noise_index=self.di))
        ll_f = np.asarray(self.full.get_ll_wdm(
            self.params, self.holder, data_index=self.di, noise_index=self.di))
        np.testing.assert_array_equal(ll_w, ll_f)


class NarrowSlabWindowTest(_Grid):
    """The production layout: W = 5 slabs, active bands running off them.

    This is the case that forced the one documented deviation from the
    F-stat scorer. F-stat CLAMPS an off-window row onto the window edge,
    which is exact only because its carrier-centred window has >= 1 layer
    of zero margin. Here the window is the band slab and its edge rows are
    nonzero, so clamping would fold an edge layer in two or three times
    over -- the compact scorer SKIPS instead, which is what makes it match
    the full-band expansion.
    """

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.win = cls._build(cls.chunked, True, **V5_KNOBS)
        cls.full = cls._build(cls.chunked, False, **V5_KNOBS)

    @classmethod
    def tearDownClass(cls):
        cls.win.clear_in_model()
        cls.full.clear_in_model()

    def setUp(self):
        self._setup(self.win, True, self.params_ref, self.slots,
                    holder=self.slab_holder)
        self._setup(self.full, False, self.params_ref, self.slots,
                    holder=self.slab_holder)

    def tearDown(self):
        self.win.clear_in_model()
        self.full.clear_in_model()

    def test_slab_window_is_the_slab(self):
        self.assertEqual(int(self.win._stash_W), self.slab_Nf)
        np.testing.assert_array_equal(
            np.asarray(self.win._stash_w_lo),
            self.slab_min_f - int(self.win._g["ind_min_f"]))
        self.assertEqual(int(self.full._stash_W),
                         int(self.full._g["Nf_active"]))

    def test_active_band_really_runs_off_the_slab(self):
        """Fixture guard: without this the skip path is never taken."""
        g = self.win._g
        m_half = int(g["m_half"])
        w_lo = np.asarray(self.win._stash_w_lo)
        m_car = np.floor(self.params_ref[:, 1] / g["layer_df"]).astype(int)
        off = 0
        for i in range(len(m_car)):
            lo = int(w_lo[i]) + int(g["ind_min_f"])
            hi = lo + self.slab_Nf - 1
            for k in range(-m_half, m_half + 1):
                if not (lo <= m_car[i] + k <= hi):
                    off += 1
        self.assertGreater(
            off, 0, "no candidate row falls off its slab -- the fixture "
                    "does not exercise the off-window skip")

    def test_windowed_matches_full_band_on_slabs(self):
        ll_w = np.asarray(self.win.get_ll(self.params, data_index=self.di))
        ll_f = np.asarray(self.full.get_ll(self.params, data_index=self.di))
        self.assertTrue(np.all(np.isfinite(ll_f)))
        np.testing.assert_array_equal(
            ll_w, ll_f,
            err_msg="narrow-slab windowed != full-band expansion: an "
                    "off-slab active row is not being treated as zero")
        np.testing.assert_array_equal(np.asarray(self.win.last_h_h),
                                      np.asarray(self.full.last_h_h))
        np.testing.assert_array_equal(np.asarray(self.win.last_d_h_im),
                                      np.asarray(self.full.last_d_h_im))

    def test_refresh_subset_on_slabs(self):
        """Slab windows do not move with f0, but the patch must still work."""
        p = self.params_ref.copy()
        p[1, 0] *= 1.4
        p[1, 4] += 0.5
        self._setup(self.win, True, p[1][None, :], self.slots[1:],
                    holder=self.slab_holder)
        self._setup(self.full, False, p[1][None, :], self.slots[1:],
                    holder=self.slab_holder)
        ll_w = np.asarray(self.win.get_ll(self.params, data_index=self.di))
        ll_f = np.asarray(self.full.get_ll(self.params, data_index=self.di))
        np.testing.assert_array_equal(ll_w, ll_f)


class MidBlockRefreshTest(_Grid):
    """Incremental refresh of a SUBSET whose window moves."""

    def _refresh_params(self, shift_layers=6):
        """Reference row 0 walked far enough in f0 to move its window."""
        p = self.params_ref.copy()
        p[0, 1] = p[0, 1] + shift_layers * self.layer_df
        return p

    def test_refresh_shifts_the_stored_window_origin(self):
        comp = self._build(self.chunked, True, **V5_KNOBS)
        self._setup(comp, True, self.params_ref, self.slots)
        try:
            w0 = np.asarray(comp._stash_w_lo).copy()
            # refresh ONLY slot 0 (a strict subset), with a moved carrier
            moved = self._refresh_params()[0][None, :]
            self._setup(comp, True, moved, self.slots[:1])
            w1 = np.asarray(comp._stash_w_lo)
            self.assertNotEqual(int(w0[0]), int(w1[0]),
                                "the refreshed source's window must MOVE, "
                                "else this fixture does not test the hazard")
            self.assertEqual(int(w0[1]), int(w1[1]),
                             "an untouched reference's window must not move")
        finally:
            comp.clear_in_model()

    def test_refresh_subset_matches_full_band_twin(self):
        """Windowed and full-band comps must agree AFTER a partial refresh.

        This is the test the stale-coefficient hazard lives in: with a
        compact stash a moved window reads its coefficients at different
        ABSOLUTE layers, so a refresh that rewrote the rows but not
        ``w_lo`` would score the refreshed source against the wrong band.
        """
        win = self._build(self.chunked, True, **V5_KNOBS)
        full = self._build(self.chunked, False, **V5_KNOBS)
        try:
            self._setup(win, True, self.params_ref, self.slots)
            self._setup(full, False, self.params_ref, self.slots)

            moved = self._refresh_params()
            self._setup(win, True, moved[0][None, :], self.slots[:1])
            self._setup(full, False, moved[0][None, :], self.slots[:1])

            # score against the MOVED reference (row 0's candidates must
            # now sit near the new expansion point) and the untouched one
            params = np.stack([moved[0], self.params_ref[1],
                               moved[0], self.params_ref[1]])
            params[2, 4] += 0.3
            params[3, 0] *= 1.1
            di = np.array([0, 1, 0, 1], dtype=np.int32)

            ll_w = np.asarray(win.get_ll(params, data_index=di))
            ll_f = np.asarray(full.get_ll(params, data_index=di))
            np.testing.assert_array_equal(
                ll_w, ll_f,
                err_msg="post-refresh windowed != full-band: a shifted "
                        "window is reading the wrong absolute layers")
            self.assertGreater(float(np.max(np.abs(ll_f))), 0.0)

            # And the refreshed reference really is the moved one.
            np.testing.assert_allclose(
                np.asarray(win.params_ref_all)[0, 1], moved[0, 1], rtol=0,
                atol=0)
        finally:
            win.clear_in_model()
            full.clear_in_model()

    def test_refresh_width_change_raises(self):
        """A mid-block layout flip must be loud, never a silent misindex."""
        comp = self._build(self.chunked, True, **V5_KNOBS)
        self._setup(comp, True, self.params_ref, self.slots)
        try:
            comp._stash_windowed = None       # simulate a layout flip
            with self.assertRaises(RuntimeError):
                self._setup(comp, False, self.params_ref[:1], self.slots[:1])
        finally:
            comp.clear_in_model()


class ClearInModelReleaseTest(_Grid):
    """clear_in_model must DROP the block's coefficient stash."""

    STASH = ("c0_sparse_all", "A0_all", "A1_all", "B0_all", "B1_all",
             "B0nc_all", "B1nc_all", "params_ref_all")

    def test_clear_nulls_the_stash(self):
        comp = self._build(self.chunked, True, **V5_KNOBS)
        self._setup(comp, True, self.params_ref, self.slots)
        for name in self.STASH:
            self.assertIsNotNone(getattr(comp, name), name)
        comp.clear_in_model()
        for name in self.STASH:
            self.assertIsNone(getattr(comp, name),
                              f"{name} survived clear_in_model")
        self.assertIsNone(comp.c0_mask_all)
        self.assertIsNone(comp._stash_W)
        self.assertIsNone(comp._stash_w_lo)

    def test_clear_actually_frees_the_arrays(self):
        """Not just re-pointed: the buffers must become collectable."""
        comp = self._build(self.chunked, True, **V5_KNOBS)
        self._setup(comp, True, self.params_ref, self.slots)
        refs = [weakref.ref(getattr(comp, n)) for n in
                ("A0_all", "B0_all", "B1nc_all")]
        self.assertTrue(all(r() is not None for r in refs))
        comp.clear_in_model()
        import gc
        gc.collect()
        for r in refs:
            self.assertIsNone(
                r(), "a stash array is still referenced after "
                     "clear_in_model -- the block's memory is not freed")

    def test_get_ll_after_clear_raises(self):
        comp = self._build(self.chunked, True, **V5_KNOBS)
        self._setup(comp, True, self.params_ref, self.slots)
        comp.clear_in_model()
        with self.assertRaises(RuntimeError):
            comp.get_ll(self.params, data_index=self.di)

    def test_get_ll_wdm_after_clear_routes_to_chunked(self):
        comp = self._build(self.chunked, True, **V5_KNOBS)
        self._setup(comp, True, self.params_ref, self.slots)
        comp.clear_in_model()
        ll = np.asarray(comp.get_ll_wdm(
            self.params, self.holder, data_index=self.di,
            noise_index=self.di))
        self.assertEqual(ll.shape, (self.params.shape[0],))
        self.assertTrue(np.all(np.isfinite(ll)))

    def test_setup_clear_setup_cycle_is_reproducible(self):
        comp = self._build(self.chunked, True, **V5_KNOBS)
        try:
            self._setup(comp, True, self.params_ref, self.slots)
            ll_1 = np.asarray(comp.get_ll(self.params, data_index=self.di))
            comp.clear_in_model()
            self._setup(comp, True, self.params_ref, self.slots)
            ll_2 = np.asarray(comp.get_ll(self.params, data_index=self.di))
            np.testing.assert_array_equal(
                ll_1, ll_2,
                err_msg="setup -> clear -> setup changed the answer")
        finally:
            comp.clear_in_model()

    def test_standalone_style_stash_is_not_released(self):
        """A comp with no ACTIVE in-model reference keeps its arrays.

        The standalone ``__init__`` construction builds a single-reference
        stash into the same attributes without ever marking it in-model;
        ``clear_in_model`` must leave that alone.
        """
        comp = self._build(self.chunked, True, **V5_KNOBS)
        sentinel = np.zeros((1, 3, 4, 5), dtype=np.complex128)
        comp.A0_all = sentinel
        comp._in_model = None
        comp.clear_in_model()
        self.assertIs(comp.A0_all, sentinel)


class NonV5LayoutTest(_Grid):
    """The scorers that were NOT ported must keep the full-band stash."""

    def test_v4_comp_stays_full_band(self):
        comp = GBSignalHetComputations.for_band_engine(
            self.chunked, v3_n_nodes=32, v4_knots=64)
        comp.setup_in_model(self.holder, self.params_ref, self.slots)
        try:
            self.assertEqual(int(comp._stash_W),
                             int(comp._g["Nf_active"]))
            ll = np.asarray(comp.get_ll(self.params, data_index=self.di))
            self.assertTrue(np.all(np.isfinite(ll)))
        finally:
            comp.clear_in_model()

    def test_v2_comp_stays_full_band(self):
        comp = GBSignalHetComputations.for_band_engine(self.chunked)
        comp.setup_in_model(self.holder, self.params_ref, self.slots)
        try:
            self.assertEqual(int(comp._stash_W),
                             int(comp._g["Nf_active"]))
            ll = np.asarray(comp.get_ll(self.params, data_index=self.di))
            self.assertTrue(np.all(np.isfinite(ll)))
        finally:
            comp.clear_in_model()

    def test_forcing_windowed_on_a_non_v5_comp_raises(self):
        comp = self._build(self.chunked, True, v3_n_nodes=32, v4_knots=64)
        with self.assertRaises(RuntimeError):
            self._setup(comp, True, self.params_ref, self.slots)


_INMODEL_STASH = ("c0_sparse_all", "c0_mask_all", "A0_all", "A1_all",
                  "B0_all", "B1_all", "B0nc_all", "B1nc_all",
                  "params_ref_all")
_FSTAT_STASH = ("A0", "A1", "B0", "B1", "B0nc", "B1nc", "mask", "refs",
                "w_lo")


class DenseTransientChunkingTest(_Grid):
    """The dense c0 is built ONE CHUNK AT A TIME; nothing may be dropped.

    User ruling 2026-09-09: ``c0_dense_w`` is needed only between
    ``make_reference`` and the bin-fold, so both build paths now run that
    pair chunk by chunk under ``GB_SIGHET_FOLD_MAX_BYTES`` instead of
    allocating the dense c0 for the whole batch (2.04 MB/source at 1 yr;
    ~520 MB at n=256, and the largest single allocation of the epoch-0
    F-stat phase). The ONLY acceptable outcome of that change is a stash
    that is bit-identical to the single-chunk build -- a reference whose
    contribution went missing, a stale row read from the reused dense
    buffer, or a mis-sliced partial chunk would all show up here.

    ``_SIGHET_FOLD_MAX_BYTES`` is read from the env ONCE at import, so the
    chunk width is forced by patching the module constant, not the env.
    """

    @staticmethod
    def _per_src_bytes(W, Nt_active, nch=3):
        # mirrors the accounting in both build paths
        return (2 * nch * nch * W * Nt_active + nch * W * Nt_active) * 16

    def _snapshot(self, comp, keys, src=None):
        out = {}
        for k in keys:
            v = getattr(comp, k) if src is None else src[k]
            out[k] = np.array(v, copy=True)
        return out

    def _assert_bit_identical(self, a, b):
        self.assertEqual(set(a), set(b))
        for k in a:
            self.assertEqual(a[k].shape, b[k].shape, k)
            self.assertTrue(np.array_equal(a[k], b[k]),
                            f"{k}: chunked build differs from single-chunk")

    def test_inmodel_stash_is_chunk_invariant(self):
        """n=2 refs: one chunk of 2 vs two chunks of 1 (reused buffer)."""
        import gbgpu.gbsignalhetcomputations as gsh
        from unittest import mock

        one = self._build(self.chunked, True, **V5_KNOBS)
        with mock.patch.object(gsh, "_SIGHET_FOLD_MAX_BYTES", 1 << 40):
            self._setup(one, True, self.params_ref, self.slots)
        try:
            ref = self._snapshot(one, _INMODEL_STASH)
        finally:
            one.clear_in_model()

        many = self._build(self.chunked, True, **V5_KNOBS)
        with mock.patch.object(gsh, "_SIGHET_FOLD_MAX_BYTES", 1):
            self._setup(many, True, self.params_ref, self.slots)
        try:
            got = self._snapshot(many, _INMODEL_STASH)
            # the scorer must see the same thing, not just the stash
            ll_many = np.asarray(many.get_ll(self.params, data_index=self.di))
        finally:
            many.clear_in_model()
        self._assert_bit_identical(ref, got)

        again = self._build(self.chunked, True, **V5_KNOBS)
        with mock.patch.object(gsh, "_SIGHET_FOLD_MAX_BYTES", 1 << 40):
            self._setup(again, True, self.params_ref, self.slots)
        try:
            ll_one = np.asarray(again.get_ll(self.params, data_index=self.di))
        finally:
            again.clear_in_model()
        self.assertTrue(np.array_equal(ll_one, ll_many))

    def test_fstat_stash_is_chunk_invariant_with_a_partial_chunk(self):
        """n=3 refs, chunk=2: exercises the final PARTIAL chunk (k < chunk),
        i.e. the exact-size buffer path, against the single-chunk build."""
        import gbgpu.gbsignalhetcomputations as gsh
        from unittest import mock

        A, C = self.params_ref[0], self.params_ref[1]
        A2 = A.copy()
        A2[1] += 0.05 * self.layer_df
        refs3 = np.stack([A, C, A2])

        one = self._build(self.chunked, True, **V5_KNOBS)
        with mock.patch.object(gsh, "_SIGHET_FOLD_MAX_BYTES", 1 << 40):
            one.setup_fstat_references(refs3, self.holder, data_index=0)
        ref = self._snapshot(one, _FSTAT_STASH, src=one._fstat)
        self.assertEqual(int(one._fstat["n"]), 3)
        W = int(one._fstat["W"])

        budget = 2 * self._per_src_bytes(W, int(one._g["Nt_active"])) + 1
        many = self._build(self.chunked, True, **V5_KNOBS)
        with mock.patch.object(gsh, "_SIGHET_FOLD_MAX_BYTES", budget):
            # chunk = max(1, min(3, budget // per_src)) == 2 -> k = 2, then 1
            many.setup_fstat_references(refs3, self.holder, data_index=0)
        got = self._snapshot(many, _FSTAT_STASH, src=many._fstat)
        self.assertEqual(int(many._fstat["W"]), W)
        self._assert_bit_identical(ref, got)


if __name__ == "__main__":
    unittest.main()
