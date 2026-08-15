"""SIGHET_INFOMAT: fast in-model information matrix vs the chunked route.

CPU-only (numpy), small grid -- same scaffolding as
``LISAanalysistools/scripts/gb_chunked_het/gb_sighet_inmodel_validate.py``.

The fast route (``SIGHET_INFOMAT=1``) second-differences the in-model
sig-het likelihood (observed information ``-d_i d_j lnL``); the chunked
delegate assembles the Fisher matrix ``<d_i h | d_j h>`` from swap cross
terms. The two coincide AT the reference when the residual contains
exactly the picked source's signal (``r - h(theta_ref) = 0``, so the
``<r - h | d_i d_j h>`` residual term vanishes) -- which is precisely the
in-model repeat-block configuration ``setup_in_model`` is called in. The
parity test is built there, so agreement is limited only by the two
heterodyne schemes' approximation errors and the finite-difference
truncation.

Also regression-tests the misindex family from the 2026-08 runbook audit:

* ``data_index=None`` under a live in-model reference MUST fall back to
  the chunked delegate (never reinterpret walker indices as slots);
* a permuted ``data_index`` MUST change the answer (proves the slot index
  is consumed, i.e. the covariance really is built per-slot);
* wrong-length / reference-less slot arrays MUST raise (the loud
  tripwire for the multi-shard global-vs-intra slot-space hazard).
"""

import os
import unittest

import numpy as np

from lisatools.detector import ESAOrbits
from lisatools.domains import WDMSettings
from lisatools.utils.constants import YRSID_SI

from gbgpu.gbcomps import GBWDMComputations
from gbgpu.gbsignalhetcomputations import GBSignalHetComputations


class _TwoSlotHolder:
    """Minimal wdm_holder: 2 buffer slots (residual slab + XYZ invC slab)."""

    def __init__(self, data_slabs, invC_slabs):
        # (n_slots, nch, Nf_active, Nt_active) / (n_slots, nch, nch, ...)
        self.linear_data_arr = [np.ascontiguousarray(data_slabs).ravel()]
        self.linear_psd_arr = [np.ascontiguousarray(invC_slabs).ravel()]

    def __len__(self):
        return 1


def _norm_diff(G1, G2):
    """Max per-element difference normalized by sqrt(diag_i diag_j)."""
    d1 = np.sqrt(np.abs(np.einsum("nii->ni", G2)))
    scale = d1[:, :, None] * d1[:, None, :]
    return float(np.max(np.abs(G1 - G2) / np.maximum(scale, 1e-300)))


class SigHetInfomatTest(unittest.TestCase):
    #: physical-parameter columns exercised (amp, f0, phi0): keeps the
    #: chunked reference route light (6 pairs x 4 swap launches) while
    #: covering an absolute-scale, a frequency and an angle column.
    INDS = [0, 1, 4]

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
        cls.sighet = GBSignalHetComputations.for_band_engine(chunked)

        # Two picked sources in separate buffer slots, well separated in
        # frequency so a slot permutation is unambiguously wrong.
        f0_A = (int(3e-3 / layer_df) + 0.37) * layer_df
        f0_C = (int(5e-3 / layer_df) + 0.62) * layer_df
        A = np.array([1e-21, f0_A, 1e-17, 0.0, 1.2, 0.7, 0.4, 2.0, 0.5])
        C = np.array([8e-22, f0_C, 2e-17, 0.0, 0.4, 1.1, 0.9, 4.0, -0.3])
        cls.params = np.stack([A, C])

        # In-model repeat-block configuration: each slot's residual slab
        # contains EXACTLY its own source's signal (the move removed the
        # template from the model, so the signal is in the residual), so
        # at the reference r - h = 0 and observed == Fisher.
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
        cls.sighet.setup_in_model(cls.holder, cls.params, cls.slots)

        # Explicit steps sized for the second-difference route on this
        # grid's likelihood scale (see the runbook: the default table +
        # SIGHET_INFOMAT_EPS_SCALE serve the same purpose in production).
        cls.eps = np.array(
            [1e-24, 1e-12, 1e-21, 1e-28, 1e-4, 1e-4, 1e-4, 1e-4, 1e-4]
        )

    @classmethod
    def tearDownClass(cls):
        cls.sighet.clear_in_model()

    def setUp(self):
        self._saved = os.environ.get("SIGHET_INFOMAT")
        os.environ["SIGHET_INFOMAT"] = "1"

    def tearDown(self):
        if self._saved is None:
            os.environ.pop("SIGHET_INFOMAT", None)
        else:
            os.environ["SIGHET_INFOMAT"] = self._saved

    # ------------------------------------------------------------------

    def _fast(self, data_index, **kw):
        return np.asarray(self.sighet.information_matrix(
            self.params, self.holder, inds=self.INDS,
            param_eps=self.eps, noise_index=self.slots,
            data_index=data_index, **kw))

    def _chunked_ref(self):
        return np.asarray(self.chunked.information_matrix(
            self.params, self.holder, inds=self.INDS,
            param_eps=self.eps, noise_index=self.slots))

    def test_fast_route_matches_chunked_at_reference(self):
        G_fast = self._fast(self.slots)
        G_ref = self._chunked_ref()
        self.assertEqual(G_fast.shape, (2, len(self.INDS), len(self.INDS)))
        # Observed == Fisher at the reference; residual budget = het
        # approximation (~1e-6 on ll) + finite-difference cancellation.
        err = _norm_diff(G_fast, G_ref)
        self.assertLess(
            err, 5e-3,
            f"fast-vs-chunked normalized mismatch {err:.3e} at the "
            "reference (expected tight agreement: residual term is zero)")

    def test_permuted_data_index_changes_the_answer(self):
        """The slot index must be CONSUMED: swapping the two sources'
        slots scores each against the other's reference and must move the
        result far outside the parity budget (this is the misindex
        regression -- a route that ignored data_index would pass parity
        while building covariances from the wrong references)."""
        G_good = self._fast(self.slots)
        G_perm = self._fast(self.slots[::-1].copy())
        err = _norm_diff(G_perm, G_good)
        self.assertGreater(
            err, 1e-1,
            f"permuted data_index changed the matrix by only {err:.3e}; "
            "the slot index is not being consumed")

    def test_none_data_index_falls_back_to_chunked(self):
        """With the in-model reference LIVE but no slots supplied, the
        route MUST be the chunked delegate -- never a silent
        reinterpretation of noise_index (walker indices) as slots."""
        G_none = np.asarray(self.sighet.information_matrix(
            self.params, self.holder, inds=self.INDS,
            param_eps=self.eps, noise_index=self.slots))
        G_ref = self._chunked_ref()
        np.testing.assert_array_equal(G_none, G_ref)

    def test_knob_off_falls_back_to_chunked(self):
        os.environ["SIGHET_INFOMAT"] = "0"
        G_off = self._fast(self.slots)
        np.testing.assert_array_equal(G_off, self._chunked_ref())

    def test_wrong_length_data_index_raises(self):
        with self.assertRaises(ValueError):
            self._fast(np.array([0, 1, 0], dtype=np.int32))

    def test_referenceless_slot_raises(self):
        """Slots outside this comp's slot->ref map (the multi-shard
        global-vs-intra hazard) must die loudly before any scoring."""
        with self.assertRaises(RuntimeError):
            self._fast(np.array([0, 7], dtype=np.int32))


if __name__ == "__main__":
    unittest.main()
