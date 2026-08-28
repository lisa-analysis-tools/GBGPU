"""Fused two-quadrature phase maximisation: parity vs the explicit double call.

The GB kernels accumulate the candidate-linear inner product as a COMPLEX
sum whose real part is ``<d|h>``; because a physical ``phi0`` shift is a
unit phasor on every candidate-linear term (division, magnitude clip,
floor, spline fit and resample all commute with it exactly), the imaginary
part of that same sum IS the quadrature ``<d|h>(phi0 + pi/2)`` up to a
path-fixed sign. The fusion exposes it as a second real output
(``d_h_im_out`` / ``last_d_h_im``), so one kernel call replaces the
two-call ``TwoQuadraturePhaseMaxMixin`` evaluation.

Contracts pinned here:

* **Stash parity** -- each comp's stashed quadrature equals the ``d_h`` of
  an explicit second call at ``phi0 + pi/2`` (this is what pins the
  per-family ``_QUAD_SIGN_*`` constants: ``|D|`` is sign-blind, only a
  signed comparison can catch a wrong sign).
* **Fused = double-call** at float precision (``rtol ~ 1e-9``): the two
  paths are equal in exact arithmetic and differ only through the second
  pipeline's rounding, NEVER assert bit-identity between them.
* **Invariants** -- ``h_h``, ``kept`` and ``non_marg_d_h`` of the fused
  call are the first (unshifted) call's values BIT-exactly.
* **One kernel call** -- the fused engine path hits the underlying comp
  exactly once per phase-maximised batch.
* **Fallback** -- a comp that stashes no quadrature (``None``; e.g. the
  JAX chunked backend) silently falls back to the legacy two-call body.

CPU-only (numpy), same small grid as ``test_sighet_infomat``.
"""

import os
import unittest

import numpy as np

from lisatools.detector import ESAOrbits
from lisatools.domains import WDMSettings
from lisatools.utils.constants import YRSID_SI

from gbgpu.gbcomps import GBWDMComputations
from gbgpu.gbsignalhetcomputations import GBSignalHetComputations
from gbgpu.gb_likelihood import (
    PHYS_IDX_PHI0,
    TwoQuadraturePhaseMaxMixin,
    WDMBandLikelihoodEngine,
)

#: fused-vs-double tolerance: exact-arithmetic-equal, FP-rounding-different.
#: The two paths each carry absolute rounding at ~1e-11 of the BATCH's
#: largest magnitude (long pipelines: polyphase/spline fits/floors), so
#: small-|D| rows need an absolute floor scaled to the batch max -- see
#: _dyn_atol. A sign flip moves a row by 2|value| >= 1e-3 of batch max,
#: 6 orders above the floor, so the signed pins stay sharp.
RTOL = 1e-9
ANGLE_ATOL = 1e-6


class _TwoSlotHolder:
    """Minimal wdm_holder: N buffer slots (residual slab + XYZ invC slab)."""

    def __init__(self, data_slabs, invC_slabs):
        self.linear_data_arr = [np.ascontiguousarray(data_slabs).ravel()]
        self.linear_psd_arr = [np.ascontiguousarray(invC_slabs).ravel()]

    def __len__(self):
        return 1


def _shift_quadrature(params):
    q = np.array(params, dtype=float, copy=True)
    q[:, PHYS_IDX_PHI0] = q[:, PHYS_IDX_PHI0] + np.pi / 2
    return q


def _assert_angles_close(a, b, atol=ANGLE_ATOL, msg=""):
    d = np.mod(np.asarray(a) - np.asarray(b) + np.pi, 2 * np.pi) - np.pi
    np.testing.assert_allclose(d, 0.0, atol=atol, err_msg=msg)


def _dyn_atol(*arrays):
    """Absolute floor scaled to the batch max (guards small-|D| rows whose
    absolute rounding is set by the batch's LARGEST accumulation scale;
    measured ~3e-11 of it on the sig-het path -- 1e-9 gives 30x margin
    while staying 6 orders below any sign-flip signal)."""
    m = max(float(np.max(np.abs(np.asarray(a)))) for a in arrays)
    return 1e-9 * max(m, 1e-300)


class _CountingComp:
    """Forwarding proxy that counts underlying kernel-batch calls."""

    def __init__(self, comp):
        object.__setattr__(self, "_comp", comp)
        object.__setattr__(self, "n_get_ll", 0)
        object.__setattr__(self, "n_swap_ll", 0)

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "_comp"), name)

    def __setattr__(self, name, value):
        setattr(object.__getattribute__(self, "_comp"), name, value)

    def get_ll_wdm(self, *args, **kwargs):
        object.__setattr__(self, "n_get_ll", self.n_get_ll + 1)
        return self._comp.get_ll_wdm(*args, **kwargs)

    def get_swap_ll_wdm(self, *args, **kwargs):
        object.__setattr__(self, "n_swap_ll", self.n_swap_ll + 1)
        return self._comp.get_swap_ll_wdm(*args, **kwargs)


class _GridFixture(unittest.TestCase):
    """Shared small-grid scaffolding (same shape as test_sighet_infomat)."""

    @classmethod
    def setUpClass(cls):
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

        f0_A = (int(3e-3 / layer_df) + 0.37) * layer_df
        f0_C = (int(5e-3 / layer_df) + 0.62) * layer_df
        A = np.array([1e-21, f0_A, 1e-17, 0.0, 1.2, 0.7, 0.4, 2.0, 0.5])
        C = np.array([8e-22, f0_C, 2e-17, 0.0, 0.4, 1.1, 0.9, 4.0, -0.3])
        cls.params_ref = np.stack([A, C])

        # Residual slab per slot = exactly that slot's source signal
        # (in-model repeat-block configuration; also gives the chunked
        # scorer a strong, generic <d|h>).
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
        cls.holder = _TwoSlotHolder(np.stack(slabs), np.stack(invCs))
        cls.slots = np.array([0, 1], dtype=np.int32)

        # Scoring batch: the two references + jittered copies (generic
        # phases/amplitudes, still inside the heterodyne validity range).
        rng = np.random.default_rng(20260827)
        rows = [A, C]
        for k in range(2):
            for p in (A, C):
                q = p.copy()
                q[0] *= 1.0 + 0.15 * rng.standard_normal()
                q[1] += 0.05 * layer_df * rng.standard_normal()
                q[4] = rng.uniform(0.0, 2 * np.pi)
                rows.append(q)
        cls.params = np.stack(rows)
        cls.di = np.array([0, 1, 0, 1, 0, 1], dtype=np.int32)

    # ---- shared assertion bundles ------------------------------------

    def _stash_parity_sighet(self, comp):
        """comp.last_d_h_im must equal the explicit quadrature call's d_h."""
        comp.get_ll(self.params, data_index=self.di)
        d_h_0 = np.asarray(comp.last_d_h).copy()
        h_h_0 = np.asarray(comp.last_h_h).copy()
        im = np.asarray(comp.last_d_h_im).copy()

        comp.get_ll(_shift_quadrature(self.params), data_index=self.di)
        d_h_90 = np.asarray(comp.last_d_h).copy()

        np.testing.assert_allclose(
            im, d_h_90, rtol=RTOL, atol=_dyn_atol(d_h_90),
            err_msg="stashed quadrature != explicit phi0+pi/2 call "
                    "(wrong _QUAD_SIGN if it matches with a flipped sign)")
        return d_h_0, d_h_90, h_h_0

    def _fused_parity_sighet(self, comp):
        d_h_0, d_h_90, h_h_0 = self._stash_parity_sighet(comp)
        ll_0 = np.asarray(comp.get_ll(self.params, data_index=self.di))

        ll_f = np.asarray(
            comp.get_ll(self.params, data_index=self.di, phase_maximize=True))
        D = d_h_0 + 1j * d_h_90
        np.testing.assert_allclose(
            np.asarray(comp.last_d_h), np.abs(D),
            rtol=RTOL, atol=_dyn_atol(d_h_0))
        np.testing.assert_allclose(
            ll_f, ll_0 + (np.abs(D) - d_h_0),
            rtol=RTOL, atol=_dyn_atol(ll_0))
        _assert_angles_close(
            np.asarray(comp.phase_angle), np.arctan2(d_h_90, d_h_0))
        # invariants of the single fused call
        np.testing.assert_array_equal(np.asarray(comp.last_h_h), h_h_0)
        np.testing.assert_allclose(
            np.asarray(comp.non_marg_d_h), d_h_0,
            rtol=RTOL, atol=_dyn_atol(d_h_0))


class SigHetFusedTest(_GridFixture):
    """Production in-model scorer branches v2 / v3 / v4 / v5."""

    KNOBS = {
        "v2": dict(),
        "v3": dict(v3_n_nodes=32),
        "v4": dict(v3_n_nodes=32, v4_knots=64),
        "v5": dict(v3_n_nodes=32, v4_knots=64, v5=1),
    }

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.sighet = {}
        for name, knobs in cls.KNOBS.items():
            comp = GBSignalHetComputations.for_band_engine(
                cls.chunked, **knobs)
            comp.setup_in_model(cls.holder, cls.params_ref, cls.slots)
            cls.sighet[name] = comp

    @classmethod
    def tearDownClass(cls):
        for comp in cls.sighet.values():
            comp.clear_in_model()
        super().tearDownClass()

    def test_v2_fused(self):
        self._fused_parity_sighet(self.sighet["v2"])

    def test_v3_fused(self):
        self._fused_parity_sighet(self.sighet["v3"])

    def test_v4_fused(self):
        self._fused_parity_sighet(self.sighet["v4"])

    def test_v5_fused(self):
        self._fused_parity_sighet(self.sighet["v5"])

    def test_kill_switch_comp_two_call_matches_fused(self):
        """GB_PHASE_MAX_FUSED=0 on the sig-het comp's own phase-max branch:
        the legacy two-call body must run and agree with the fused result."""
        comp = self.sighet["v2"]
        ll_f = np.asarray(comp.get_ll(self.params, data_index=self.di,
                                      phase_maximize=True))
        d_h_f = np.asarray(comp.last_d_h).copy()
        ang_f = np.asarray(comp.phase_angle).copy()
        saved = os.environ.get("GB_PHASE_MAX_FUSED")
        os.environ["GB_PHASE_MAX_FUSED"] = "0"
        try:
            ll_2 = np.asarray(comp.get_ll(self.params, data_index=self.di,
                                          phase_maximize=True))
            d_h_2 = np.asarray(comp.last_d_h).copy()
            ang_2 = np.asarray(comp.phase_angle).copy()
        finally:
            if saved is None:
                os.environ.pop("GB_PHASE_MAX_FUSED", None)
            else:
                os.environ["GB_PHASE_MAX_FUSED"] = saved
        np.testing.assert_allclose(ll_2, ll_f, rtol=RTOL,
                                   atol=_dyn_atol(ll_f))
        np.testing.assert_allclose(d_h_2, d_h_f, rtol=RTOL,
                                   atol=_dyn_atol(d_h_f))
        _assert_angles_close(ang_2, ang_f)

    def test_get_ll_wdm_stashes_quadrature_in_model(self):
        """The engine-facing route must carry the stash on both paths."""
        comp = self.sighet["v2"]
        comp.get_ll_wdm(self.params, self.holder, data_index=self.di,
                        noise_index=self.di)
        im = np.asarray(comp.d_h_im_out).copy()
        comp.get_ll_wdm(_shift_quadrature(self.params), self.holder,
                        data_index=self.di, noise_index=self.di)
        np.testing.assert_allclose(
            im, np.asarray(comp.d_h_out), rtol=RTOL,
            atol=_dyn_atol(comp.d_h_out))


class ChunkedFusedTest(_GridFixture):
    """Chunked-heterodyne comp (the RJ scorer): get_ll + swap quadratures."""

    def test_get_ll_stash_parity(self):
        self.chunked.get_ll_wdm(self.params, self.holder,
                                data_index=self.di, noise_index=self.di)
        im = np.asarray(self.chunked.d_h_im_out).copy()
        self.chunked.get_ll_wdm(_shift_quadrature(self.params), self.holder,
                                data_index=self.di, noise_index=self.di)
        d_h_90 = np.asarray(self.chunked.d_h_out).copy()
        np.testing.assert_allclose(
            im, d_h_90, rtol=RTOL, atol=_dyn_atol(d_h_90),
            err_msg="chunked d_h_im_out != explicit quadrature call")

    def test_swap_stash_parity(self):
        p_rm = self.params[:2]
        p_add = self.params[2:4]
        di = self.slots
        self.chunked.get_swap_ll_wdm(p_add, p_rm, self.holder,
                                     data_index=di, noise_index=di)
        im_a = np.asarray(self.chunked.d_h_add_im_out).copy()
        im_ar = np.asarray(self.chunked.add_remove_im_out).copy()
        self.chunked.get_swap_ll_wdm(_shift_quadrature(p_add), p_rm,
                                     self.holder, data_index=di,
                                     noise_index=di)
        np.testing.assert_allclose(
            im_a, np.asarray(self.chunked.d_h_add_out), rtol=RTOL,
            atol=_dyn_atol(self.chunked.d_h_add_out),
            err_msg="swap d_h_add quadrature stash mismatch")
        np.testing.assert_allclose(
            im_ar, np.asarray(self.chunked.add_remove_out), rtol=RTOL,
            atol=_dyn_atol(self.chunked.add_remove_out),
            err_msg="swap add_remove quadrature stash mismatch")


class WDMEngineFusedTest(_GridFixture):
    """Engine-level fused path: one kernel call, double-call parity."""

    def _engine(self, comp):
        return WDMBandLikelihoodEngine(
            _CountingComp(comp), self.wdm_set, 3, "XYZ")

    def _fused_vs_double(self, eng):
        kw = dict(data_index=self.di, noise_index=self.di, N_vals=None,
                  waveform_kwargs={})
        ll_0 = np.asarray(eng.get_ll(self.holder, self.params,
                                     phase_maximize=False, **kw))
        d_h_0 = np.asarray(eng.d_h_out).copy()
        h_h_0 = np.asarray(eng.h_h_out).copy()
        kept_0 = np.asarray(eng.kept_out).copy()
        eng.get_ll(self.holder, _shift_quadrature(self.params),
                   phase_maximize=False, **kw)
        d_h_90 = np.asarray(eng.d_h_out).copy()

        n_before = eng.gb_comps.n_get_ll
        ll_f = np.asarray(eng.get_ll(self.holder, self.params,
                                     phase_maximize=True, **kw))
        self.assertEqual(
            eng.gb_comps.n_get_ll - n_before, 1,
            "fused phase-max must hit the underlying comp exactly once")

        D = d_h_0 + 1j * d_h_90
        np.testing.assert_allclose(
            ll_f, np.where(ll_0 > -1e290, ll_0 + (np.abs(D) - d_h_0), ll_0),
            rtol=RTOL, atol=_dyn_atol(ll_0))
        np.testing.assert_allclose(
            np.asarray(eng.d_h_out), np.abs(D), rtol=RTOL,
            atol=_dyn_atol(d_h_0))
        _assert_angles_close(np.asarray(eng.phase_angle),
                             np.arctan2(d_h_90, d_h_0))
        np.testing.assert_array_equal(np.asarray(eng.h_h_out), h_h_0)
        np.testing.assert_array_equal(np.asarray(eng.kept_out), kept_0)
        np.testing.assert_allclose(
            np.asarray(eng.non_marg_d_h), d_h_0, rtol=RTOL,
            atol=_dyn_atol(d_h_0))

    def test_engine_over_chunked(self):
        self._fused_vs_double(self._engine(self.chunked))

    def test_engine_over_sighet_in_model(self):
        comp = GBSignalHetComputations.for_band_engine(self.chunked)
        comp.setup_in_model(self.holder, self.params_ref, self.slots)
        try:
            self._fused_vs_double(self._engine(comp))
        finally:
            comp.clear_in_model()

    def test_engine_swap_fused(self):
        eng = self._engine(self.chunked)
        p_rm, p_add = self.params[:2], self.params[2:4]
        kw = dict(data_index=self.slots, noise_index=self.slots,
                  N_vals=None, waveform_kwargs={})
        r0 = eng.get_swap_ll(self.holder, p_rm, p_add,
                             phase_maximize=False, **kw)
        r90 = eng.get_swap_ll(self.holder, p_rm, _shift_quadrature(p_add),
                              phase_maximize=False, **kw)

        n_before = eng.gb_comps.n_swap_ll
        rf = eng.get_swap_ll(self.holder, p_rm, p_add,
                             phase_maximize=True, **kw)
        self.assertEqual(
            eng.gb_comps.n_swap_ll - n_before, 1,
            "fused swap phase-max must hit the underlying comp exactly once")

        g0 = np.asarray(r0.d_h_add) - np.asarray(r0.hh_cross)
        g90 = np.asarray(r90.d_h_add) - np.asarray(r90.hh_cross)
        delta = np.arctan2(g90, g0)
        rot = np.exp(1j * delta)

        def at_max(a0, a90):
            return (np.conj(np.asarray(a0) + 1j * np.asarray(a90))
                    * rot).real

        d_h_add = at_max(r0.d_h_add, r90.d_h_add)
        hh_cross = at_max(r0.hh_cross, r90.hh_cross)
        gain = (d_h_add - hh_cross) - g0
        ll_ref = np.where(np.asarray(r0.ll_diff) > -1e290,
                          np.asarray(r0.ll_diff) + gain,
                          np.asarray(r0.ll_diff))

        np.testing.assert_allclose(np.asarray(rf.ll_diff), ll_ref,
                                   rtol=RTOL, atol=_dyn_atol(ll_ref))
        np.testing.assert_allclose(np.asarray(rf.d_h_add), d_h_add,
                                   rtol=RTOL, atol=_dyn_atol(d_h_add))
        np.testing.assert_allclose(np.asarray(rf.hh_cross), hh_cross,
                                   rtol=RTOL, atol=_dyn_atol(hh_cross))
        _assert_angles_close(np.asarray(rf.phase_angle), delta)
        # untouched outputs come from the single call bit-exactly
        np.testing.assert_array_equal(np.asarray(rf.d_h_remove),
                                      np.asarray(r0.d_h_remove))
        np.testing.assert_array_equal(np.asarray(rf.hh_add),
                                      np.asarray(r0.hh_add))
        np.testing.assert_array_equal(np.asarray(rf.hh_remove),
                                      np.asarray(r0.hh_remove))
        np.testing.assert_array_equal(np.asarray(rf.opt_snr_add),
                                      np.asarray(r0.opt_snr_add))
        np.testing.assert_array_equal(np.asarray(rf.kept),
                                      np.asarray(r0.kept))


class _FDHolder:
    """Minimal windowed FD holder: per-row complex data + XYZ invC slabs."""

    def __init__(self, data_rows, invC_rows, min_freq_inds):
        self.linear_data_arr = [np.ascontiguousarray(data_rows).ravel()]
        self.linear_psd_arr = [np.ascontiguousarray(invC_rows).ravel()]
        self.acs_total_entries = int(data_rows.shape[0])
        self.min_freq_inds = np.asarray(min_freq_inds, dtype=np.int32)


class FDCompFusedTest(unittest.TestCase):
    """FD comp family: quadrature stashes vs the explicit shifted call.

    Synthetic windowed holder (random data, identity invC) -- the parity
    only needs the kernel's candidate-linearity, not physical data.
    """

    @classmethod
    def setUpClass(cls):
        from lisatools.domains import FDSettings

        from gbgpu.gbcomps import GBFDComputations

        cls.comp = GBFDComputations(
            FDSettings(N=2048, df=1e-6), t_ref=0.0, N_sparse=64,
            tdi_config="1st generation", force_backend="cpu",
            d_d=0.0, tdi_type="XYZ")

        rng = np.random.default_rng(20260828)
        n_rows, nch, n_rfft = 2, 3, 256
        k0 = 500
        data = 1e-21 * (rng.standard_normal((n_rows, nch, n_rfft))
                        + 1j * rng.standard_normal((n_rows, nch, n_rfft)))
        invC = np.zeros((n_rows, nch, nch, n_rfft))
        for c in range(nch):
            invC[:, c, c, :] = 1.0
        cls.holder = _FDHolder(data.astype(np.complex128), invC,
                               np.full(n_rows, k0, dtype=np.int32))

        f0_A = (k0 + 100.37) * 1e-6
        f0_C = (k0 + 140.62) * 1e-6
        cls.params = np.stack([
            np.array([1e-21, f0_A, 1e-17, 0.0, 1.2, 0.7, 0.4, 2.0, 0.5]),
            np.array([8e-22, f0_C, 2e-17, 0.0, 0.4, 1.1, 0.9, 4.0, -0.3]),
        ])
        cls.di = np.array([0, 1], dtype=np.int32)

    def test_get_ll_stash_parity(self):
        self.comp.get_ll_fd(self.params, self.holder, data_index=self.di,
                            noise_index=self.di, convert_to_ra_dec=False)
        im = np.asarray(self.comp.d_h_im_out).copy()
        self.comp.get_ll_fd(_shift_quadrature(self.params), self.holder,
                            data_index=self.di, noise_index=self.di,
                            convert_to_ra_dec=False)
        d_h_90 = np.asarray(self.comp.d_h_out).copy()
        np.testing.assert_allclose(
            im, d_h_90, rtol=RTOL, atol=_dyn_atol(d_h_90),
            err_msg="FD d_h_im_out != explicit quadrature call")

    def test_swap_stash_parity(self):
        p_rm = self.params
        p_add = self.params.copy()
        p_add[:, 4] += 0.7
        p_add[:, 0] *= 1.1
        (_, _, _, _, _, _, ar_0) = self.comp.get_swap_ll_fd(
            p_add, p_rm, self.holder, data_index=self.di,
            noise_index=self.di, convert_to_ra_dec=False)
        im_a = np.asarray(self.comp.d_h_add_im_out).copy()
        im_ar = np.asarray(self.comp.add_remove_im_out).copy()
        (_, _, d_h_a_90, _, _, _, ar_90) = self.comp.get_swap_ll_fd(
            _shift_quadrature(p_add), p_rm, self.holder,
            data_index=self.di, noise_index=self.di,
            convert_to_ra_dec=False)
        np.testing.assert_allclose(
            im_a, np.asarray(d_h_a_90), rtol=RTOL,
            atol=_dyn_atol(d_h_a_90),
            err_msg="FD swap d_h_add quadrature stash mismatch")
        np.testing.assert_allclose(
            im_ar, np.asarray(ar_90), rtol=RTOL, atol=_dyn_atol(ar_90),
            err_msg="FD swap add_remove quadrature stash mismatch "
                    "(conjugated-slot sign constant if flipped)")


class _NoQuadStubEngine(TwoQuadraturePhaseMaxMixin):
    """Analytic engine WITHOUT a quadrature stash -> must take the
    legacy two-call fallback (contract for e.g. the JAX chunked backend).

    d_h(phi0) = R * cos(phi0 + theta); h_h constant; ll = -0.5*(h_h - 2 d_h).
    """

    xp = np

    def __init__(self, R, theta, h_h):
        self.R, self.theta, self.h_h = R, theta, h_h
        self.n_calls = 0

    def get_ll(self, buffer_aca, params_phys, *, data_index, noise_index,
               N_vals, phase_maximize=False, return_inner_products=False,
               waveform_kwargs):
        if phase_maximize:
            return self._get_ll_phase_max(
                buffer_aca, params_phys,
                data_index=data_index, noise_index=noise_index,
                N_vals=N_vals, return_inner_products=return_inner_products,
                waveform_kwargs=waveform_kwargs)
        self.n_calls += 1
        phi = np.asarray(params_phys)[:, PHYS_IDX_PHI0]
        d_h = self.R * np.cos(phi + self.theta)
        self.d_h_out = d_h
        self.h_h_out = np.full_like(d_h, self.h_h)
        self.kept_out = np.ones(d_h.shape, dtype=bool)
        self.d_h_im_out = None          # <- no fused quadrature available
        self.phase_angle = None
        return -0.5 * (self.h_h_out - 2.0 * d_h)


class _QuadStubEngine(_NoQuadStubEngine):
    """Stub engine WITH the fused quadrature stash (exact analytic value)."""

    def get_ll(self, buffer_aca, params_phys, *, data_index, noise_index,
               N_vals, phase_maximize=False, return_inner_products=False,
               waveform_kwargs):
        ll = super().get_ll(
            buffer_aca, params_phys, data_index=data_index,
            noise_index=noise_index, N_vals=N_vals,
            phase_maximize=phase_maximize,
            return_inner_products=return_inner_products,
            waveform_kwargs=waveform_kwargs)
        if not phase_maximize:
            phi = np.asarray(params_phys)[:, PHYS_IDX_PHI0]
            self.d_h_im_out = self.R * np.cos(phi + np.pi / 2 + self.theta)
        return ll


class KillSwitchTest(unittest.TestCase):
    """GB_PHASE_MAX_FUSED=0 must force the legacy two-call body even when
    the fused stash is available -- the production rollback lever for the
    first GPU-validated rows."""

    def _run(self, env_val):
        eng = _QuadStubEngine(R=7.5, theta=0.9, h_h=56.25)
        params = np.zeros((4, 9))
        params[:, PHYS_IDX_PHI0] = [0.0, 1.0, 2.5, 5.0]
        saved = os.environ.get("GB_PHASE_MAX_FUSED")
        try:
            if env_val is None:
                os.environ.pop("GB_PHASE_MAX_FUSED", None)
            else:
                os.environ["GB_PHASE_MAX_FUSED"] = env_val
            ll = eng.get_ll(None, params, data_index=None, noise_index=None,
                            N_vals=None, phase_maximize=True,
                            waveform_kwargs={})
        finally:
            if saved is None:
                os.environ.pop("GB_PHASE_MAX_FUSED", None)
            else:
                os.environ["GB_PHASE_MAX_FUSED"] = saved
        return eng, np.asarray(ll)

    def test_default_is_fused_single_call(self):
        eng, ll = self._run(None)
        self.assertEqual(eng.n_calls, 1)
        np.testing.assert_allclose(np.asarray(eng.d_h_out),
                                   np.full(4, 7.5), rtol=1e-12)

    def test_kill_switch_forces_two_call_same_answer(self):
        eng0, ll0 = self._run(None)
        eng, ll = self._run("0")
        self.assertEqual(eng.n_calls, 2,
                         "GB_PHASE_MAX_FUSED=0 must take the two-call body")
        np.testing.assert_allclose(ll, ll0, rtol=1e-12)
        np.testing.assert_allclose(np.asarray(eng.d_h_out),
                                   np.asarray(eng0.d_h_out), rtol=1e-12)
        _assert_angles_close(np.asarray(eng.phase_angle),
                             np.asarray(eng0.phase_angle), atol=1e-10)


class MixinFallbackTest(unittest.TestCase):
    def test_none_stash_falls_back_to_two_calls_and_exact_max(self):
        eng = _NoQuadStubEngine(R=7.5, theta=0.9, h_h=56.25)
        params = np.zeros((4, 9))
        params[:, PHYS_IDX_PHI0] = [0.0, 1.0, 2.5, 5.0]
        ll = eng.get_ll(None, params, data_index=None, noise_index=None,
                        N_vals=None, phase_maximize=True, waveform_kwargs={})
        self.assertEqual(eng.n_calls, 2,
                         "None stash must take the legacy two-call body")
        # analytic maximum of R cos(.) is R for every row
        np.testing.assert_allclose(np.asarray(eng.d_h_out),
                                   np.full(4, 7.5), rtol=1e-12)
        np.testing.assert_allclose(
            np.asarray(ll), -0.5 * (56.25 - 2 * 7.5) * np.ones(4),
            rtol=1e-12)
        # phase_angle really maximises: d_h(phi0 + delta*) == R
        phi_max = params[:, PHYS_IDX_PHI0] + np.asarray(eng.phase_angle)
        np.testing.assert_allclose(7.5 * np.cos(phi_max + 0.9),
                                   np.full(4, 7.5), rtol=1e-12)


if __name__ == "__main__":
    unittest.main()
