"""Lookup Fisher / Gram information matrix (``GBLookupComputations.information_matrix``,
the sig-het ``SIGHET_INFOMAT_ENGINE=lookup`` route) and the ``SIGHET_SETUP_PROFILE`` knob.

The reference is the EXACT-template Gram matrix: central differences of
``fill_global_wdm`` (the chunked template) with phase steps of 1e-4 rad (f0 / fdot /
fddot scaled by Tobs), contracted with the same invC. It is converged (8e-5 against
steps of 1e-3). The chunked delegate's own ``information_matrix`` is NOT the truth: its
default fdot step gives a NEGATIVE fdot-fdot entry on this fixture.

Same fixture as ``test_sighet_lookup_reference`` (unequal-gain XYZ invC, four carriers
across the Meyer flat top and transition band, SNR 300, 30 d). CPU here; the GPU == CPU
class runs on the cluster only (``SIGHET_GPU_TEST_BACKEND``).
"""

import unittest

import numpy as np

from gbgpu.gblookupcomputations import GBLookupComputations

from . import test_sighet_lookup_reference as _ref
from .test_gb_lookup_kernel import GPU

# physical slots the in-model proposal scores (fddot is a fill, never sampled)
INDS = np.array([0, 1, 2, 4, 5, 6, 7, 8])
ALL = np.arange(9)

#: worst element |G_lk - G_exact|_ab / sqrt(G_aa G_bb). Measured 7.3e-5 here (1.2e-4 at
#: 6 months, the lookup template's accuracy). Mutations it catches: the chunked default
#: f0 / fdot steps (5.06), its 1e-6 rad angle steps (7.5e-3), its fddot step (5.8e2).
BOUND = 1e-3


def _rel(a, b):
    """Per source, |A_ab - B_ab| / sqrt(B_aa B_bb) (the correlation-normalized error)."""
    d = np.sqrt(np.einsum("naa->na", np.abs(b)))
    return np.abs(a - b) / (d[:, :, None] * d[:, None, :])


class _DeviceLike:
    """Stand-in for a cupy array on a CPU box: converts only through ``.get()`` and
    refuses ``np.asarray`` as cupy does (job 735: the first GB_CHOL_CACHE refresh died
    on ``np.asarray`` of the chunked helper's cupy step array)."""

    def __init__(self, a):
        self.a = np.asarray(a)

    def get(self):
        return self.a

    def __array__(self, *a, **k):
        raise TypeError("Implicit conversion to a NumPy array is not allowed.")


class _MirrorHolder:
    """A shared-psd MIRROR holder: ``linear_psd_arr[0]`` is a per-walker full-band invC
    plane and slot ``s`` reads row ``psd_row_index[s]`` (narrow data slabs)."""

    def __init__(self, data, invc_rows, psd_row_index, band_slab_Nf=5):
        self.linear_data_arr = [np.ascontiguousarray(data).ravel()]
        self.linear_psd_arr = [np.ascontiguousarray(invc_rows).ravel()]
        self.psd_row_index = np.asarray(psd_row_index, dtype=np.int32)
        self.band_slab_Nf = band_slab_Nf

    def __len__(self):
        return 1


class LookupInformationMatrixTest(unittest.TestCase):
    _comp = _ref.SighetLookupReferenceTest._comp

    @classmethod
    def setUpClass(cls):
        _ref.SighetLookupReferenceTest.setUpClass.__func__(cls)
        wdm, R = cls.wdm, cls.R
        Nfa, Nta = int(wdm.ind_max_f - wdm.ind_min_f + 1), int(wdm.Nt_active)
        cls.iC = iC = cls.holder.linear_psd_arr[0].reshape(R, 3, 3, Nfa, Nta)

        def fill(p):
            out = np.zeros((R, 3, Nfa, Nta))
            cls.ch.fill_global_wdm(p, out.reshape(-1), data_index=np.arange(R, dtype=np.int32),
                                   factors=np.ones(R))
            return out

        T = wdm.Tobs
        E = np.array([1e-3 * cls.ref[0, 0], 1e-4 / T, 1e-4 / T**2, 1e-4 / T**3,
                      1e-4, 1e-4, 1e-4, 1e-4, 1e-4])
        dh = []
        for i in ALL:
            e = np.zeros(9)
            e[i] = E[i]
            dh.append((fill(cls.ref + e) - fill(cls.ref - e)) / (2 * E[i]))
        dh = np.array(dh)
        cls.truth_all = np.einsum("arcft,rcdft,brdft->rab", dh, iC, dh)
        cls.truth = cls.truth_all[:, INDS][:, :, INDS]
        cls.di = np.arange(R, dtype=np.int32)

    def _lookup(self, comp=None, inds=INDS, holder=None, **kw):
        comp = comp or self._comp()
        return np.asarray(comp._ref_lookup.information_matrix(
            self.ref, self.holder if holder is None else holder, inds=inds,
            noise_index=kw.pop("noise_index", self.di), **kw))

    def test_matches_exact_template_gram(self):
        lk = self._lookup(inds=ALL)
        rel = _rel(lk, self.truth_all)
        self.assertLess(rel.max(), BOUND, f"max correlation-normalized diff {rel.max():.2e}")
        np.testing.assert_array_equal(lk, lk.transpose(0, 2, 1))
        # the proposal uses the inverse over the sampled slots: compare the marginal
        # widths too (the A / inc / psi / phi0 block is near-degenerate at 30 d and
        # amplifies element errors). Measured 1.3e-4; 0.20 with the 1e-6 rad angle steps.
        lk = self._lookup()
        s_ex = np.sqrt(np.einsum("naa->na", np.linalg.inv(self.truth)))
        s_lk = np.sqrt(np.einsum("naa->na", np.linalg.inv(lk)))
        np.testing.assert_allclose(s_lk, s_ex, rtol=2e-3)

    def test_engine_knob_routes_without_an_in_model_block(self):
        comp = self._comp()
        with _ref._Env(SIGHET_INFOMAT_ENGINE="lookup"):
            via = np.asarray(comp.information_matrix(
                self.ref, self.holder, inds=INDS, noise_index=self.di))
        np.testing.assert_array_equal(via, self._lookup(comp))
        with _ref._Env(SIGHET_INFOMAT_ENGINE="lookup"):
            with self.assertRaises(RuntimeError):         # no table attached
                self._comp(table=False).information_matrix(
                    self.ref, self.holder, inds=INDS, noise_index=self.di)
        with _ref._Env(SIGHET_INFOMAT_ENGINE="lokup"):      # a typo must not fall through
            with self.assertRaises(ValueError):
                comp.information_matrix(self.ref, self.holder, inds=INDS,
                                        noise_index=self.di)

    def test_device_array_inputs(self):
        """Steps, inds and rows given as device arrays are read through ``.get()``."""
        comp = self._comp()
        lk = comp._ref_lookup
        host = self._lookup(comp)
        dev = np.asarray(lk.information_matrix(
            self.ref, self.holder, inds=_DeviceLike(INDS),
            param_eps=_DeviceLike(lk.info_matrix_param_eps()),
            noise_index=_DeviceLike(self.di)))
        np.testing.assert_array_equal(dev, host)

    def test_holder_layouts_and_guards(self):
        comp = self._comp()
        base = self._lookup(comp)
        R = self.R
        # shared-psd mirror: slots map to rows R..2R-1 of a 2R-row plane, which hold
        # 2 x invC -> exactly twice the matrix (unmapped rows would give 1x)
        planes = np.concatenate([self.iC, 2.0 * self.iC])
        mirror = _MirrorHolder(np.zeros(1), planes, np.arange(R) + R)
        np.testing.assert_allclose(self._lookup(comp, holder=mirror), 2.0 * base,
                                   rtol=1e-14, atol=0)
        narrow = _ref._Holder(np.zeros(1), np.zeros(1))
        narrow.band_slab_Nf = 5                      # per-slot narrow invC slabs
        with self.assertRaises(NotImplementedError):
            self._lookup(comp, holder=narrow)
        with self.assertRaises(NotImplementedError):
            self._lookup(comp, convert_to_ra_dec=True)
        with self.assertRaises(ValueError):
            self._lookup(comp, noise_index=self.di[:-1])
        with self.assertRaises(IndexError):
            self._lookup(comp, noise_index=self.di + R)
        with self.assertRaises(ValueError):
            self._lookup(comp, param_eps=np.ones(8))

    def test_NEGATIVE_CONTROL_a_wrong_noise_row_is_caught(self):
        """The comparison has teeth: scoring against a 2x invC must fail it."""
        self.assertGreater(_rel(2.0 * self._lookup(), self.truth).max(), BOUND)


class SetupProfileKnobTest(unittest.TestCase):
    """``SIGHET_SETUP_PROFILE=1``: per-phase setup_in_model times, one INFO line per
    ``_PROF_EVERY`` calls; off = no timer state."""

    _comp = _ref.SighetLookupReferenceTest._comp

    @classmethod
    def setUpClass(cls):
        _ref.SighetLookupReferenceTest.setUpClass.__func__(cls)

    def _setup(self, comp, profile):
        di = np.arange(self.R, dtype=np.int32)
        with _ref._Env(SIGHET_ANCHOR_CORRECT="0", SIGHET_SETUP_PROFILE=profile):
            comp.setup_in_model(self.holder, self.ref, di)
        comp.clear_in_model()

    def test_profile_knob(self):
        comp = self._comp()
        self._setup(comp, "0")
        self.assertNotIn("_setup_prof", comp.__dict__)
        comp._PROF_EVERY = 2
        with self.assertLogs("gbgpu.gbsignalhetcomputations", level="INFO") as cm:
            self._setup(comp, "1")
            self._setup(comp, "1")
        lines = [r for r in cm.output if "[SIGHET_SETUP_PROFILE]" in r]
        self.assertEqual(len(lines), 1, cm.output)
        self.assertIn(f"2 calls / {2 * self.R} refs (ref build lookup)", lines[0])
        self.assertIn("ms/ref", lines[0])
        for phase in ("gather", "checks", "ref_build", "fold", "stash", "anchor"):
            self.assertIn(f"{phase} ", lines[0])


@unittest.skipIf(GPU is None, "no CUDA backend (set SIGHET_GPU_TEST_BACKEND on the cluster)")
class LookupInformationMatrixGpuTest(unittest.TestCase):
    """GPU == CPU for the lookup information matrix, incl. device step arrays.

    Tolerance: the repo's GPU-vs-CPU convention is rtol 1e-9 on direct outputs
    (34a1210). A derivative here is a template difference divided by a step that moves
    the phase by ``s = INFOMAT_PHASE_STEP`` (1e-4 rad), so backend rounding in the two
    templates is amplified by ~1 / (2 s): compare at ``1e-9 / (2 s)`` = 5e-6, normalized
    per element by ``sqrt(G_aa G_bb)`` (the per-row/column scale, playing the role of the
    convention's floor at the batch's largest accumulation).
    """

    @classmethod
    def setUpClass(cls):
        import cupy as cp

        from lisatools.detector import ESAOrbits
        from lisatools.domains import WDMSettings

        from gbgpu.gbcomps import GBWDMComputations

        _ref.SighetLookupReferenceTest.setUpClass.__func__(cls)
        wdm = cls.wdm
        g_wdm = WDMSettings(_ref.NF, _ref.NT, _ref.DT, t0=wdm.t0, min_freq=1e-4,
                            max_freq=2.5e-2, min_time=_ref.EDGE * _ref.NF * _ref.DT,
                            max_time=(_ref.NT - _ref.EDGE) * _ref.NF * _ref.DT,
                            force_backend=GPU)
        cls.g_ch = GBWDMComputations(
            g_wdm, t_ref=cls.ch.t_ref, Nt_sub=256, n_pad=32, N_sparse=256, N_cp_sig=48,
            N_cp_orbit=32, orbits=ESAOrbits(force_backend=GPU), tdi_config="2nd generation",
            force_backend=GPU, d_d=0.0, tdi_type="XYZ")
        cls.g_ch.convert_to_ra_dec = False
        cls.g_lk = GBLookupComputations(cls.g_ch, cls.table)
        cls.c_lk = GBLookupComputations(cls.ch, cls.table)
        h = cls.holder

        class _G:
            linear_data_arr = [cp.asarray(h.linear_data_arr[0])]
            linear_psd_arr = [cp.asarray(h.linear_psd_arr[0])]

            def __len__(self):
                return 1

        cls.g_holder = _G()
        cls.di = np.arange(cls.R, dtype=np.int32)
        cls.tol = 1e-9 / (2.0 * GBLookupComputations.INFOMAT_PHASE_STEP)

    def _gpu(self, lk=None, **kw):
        import cupy as cp

        out = (lk or self.g_lk).information_matrix(
            cp.asarray(self.ref), self.g_holder, inds=cp.asarray(INDS),
            noise_index=cp.asarray(self.di), **kw)
        self.assertIsInstance(out, cp.ndarray)
        return cp.asnumpy(out)

    def test_gpu_matches_cpu(self):
        cpu = np.asarray(self.c_lk.information_matrix(self.ref, self.holder, inds=INDS,
                                                      noise_index=self.di))
        rel = _rel(self._gpu(), cpu)
        self.assertLess(rel.max(), self.tol, f"GPU vs CPU {rel.max():.2e}")

    def test_device_step_arrays(self):
        """cupy steps (the chunked helper's own output: the job-735 input) give exactly
        the host-step result."""
        import cupy as cp

        steps = self.g_lk.info_matrix_param_eps()
        host = self._gpu(param_eps=steps)
        np.testing.assert_array_equal(self._gpu(param_eps=cp.asarray(steps)), host)
        np.testing.assert_array_equal(self._gpu(), host)
        ch_steps = self.g_ch._info_matrix_param_eps(9, None)
        self.assertIsInstance(ch_steps, cp.ndarray)
        np.testing.assert_array_equal(self._gpu(param_eps=ch_steps),
                                      self._gpu(param_eps=cp.asnumpy(ch_steps)))

    def test_sighet_engine_route(self):
        from gbgpu.gbsignalhetcomputations import GBSignalHetComputations

        comp = GBSignalHetComputations.for_band_engine(
            self.g_ch, cp_repr="carrier", lookup_table=self.table, **_ref.V5_KNOBS)
        with _ref._Env(SIGHET_INFOMAT_ENGINE="lookup"):
            via = self._gpu(lk=comp)
        np.testing.assert_array_equal(via, self._gpu())


if __name__ == "__main__":
    unittest.main()
