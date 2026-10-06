"""Lookup Fisher ``<dh_a|dh_b>`` (``GBLookupComputations.information_matrix``,
``SIGHET_INFOMAT_ENGINE=lookup``) against the EXACT-template Gram matrix:
central differences of ``fill_global_wdm`` (the chunked template) with
Tobs-scaled steps, contracted with the same invC. That reference is converged
(identical at two step scales); the chunked delegate's own
``information_matrix`` is NOT used as truth -- its default fdot step gives a
NEGATIVE fdot-fdot entry on this fixture.

Same fixture as ``test_sighet_lookup_reference`` (unequal-gain XYZ invC, four
carriers across the Meyer flat top and transition band, SNR 300).
"""

import unittest

import numpy as np

from . import test_sighet_lookup_reference as _ref

# physical slots the in-model proposal scores (fddot is a fill, never sampled)
INDS = np.array([0, 1, 2, 4, 5, 6, 7, 8])


class LookupInformationMatrixTest(unittest.TestCase):
    _comp = _ref.SighetLookupReferenceTest._comp

    @classmethod
    def setUpClass(cls):
        _ref.SighetLookupReferenceTest.setUpClass.__func__(cls)
        wdm, R = cls.wdm, cls.R
        Nfa, Nta = int(wdm.ind_max_f - wdm.ind_min_f + 1), int(wdm.Nt_active)
        iC = cls.holder.linear_psd_arr[0].reshape(R, 3, 3, Nfa, Nta)

        def fill(p):
            out = np.zeros((R, 3, Nfa, Nta))
            cls.ch.fill_global_wdm(p, out.reshape(-1), data_index=np.arange(R, dtype=np.int32),
                                   factors=np.ones(R))
            return out

        T = wdm.Tobs
        E = np.array([1e-3 * cls.ref[0, 0], 1e-4 / T, 1e-4 / T**2, 0, 1e-4, 1e-4, 1e-4, 1e-4,
                      1e-4])
        dh = []
        for i in INDS:
            e = np.zeros(9)
            e[i] = E[i]
            dh.append((fill(cls.ref + e) - fill(cls.ref - e)) / (2 * E[i]))
        dh = np.array(dh)
        cls.truth = np.einsum("arcft,rcdft,brdft->rab", dh, iC, dh)

    def _chunked(self):
        return self.truth

    def _lookup(self, comp=None):
        comp = comp or self._comp()
        di = np.arange(self.R, dtype=np.int32)
        return np.asarray(comp._ref_lookup.information_matrix(
            self.ref, self.holder, inds=INDS, noise_index=di))

    @staticmethod
    def _rel(a, b):
        # per source, relative to the matrix scale in each (a, b) pair:
        # |A_ab - B_ab| / sqrt(B_aa B_bb)
        d = np.sqrt(np.einsum("naa->na", np.abs(b)))
        return np.abs(a - b) / (d[:, :, None] * d[:, None, :])

    def test_matches_exact_template_gram(self):
        # measured 7.6e-3 (lookup template accuracy); 5.06 with the chunked
        # default f0 / fdot steps (the defect this guards)
        ch, lk = self._chunked(), self._lookup()
        rel = self._rel(lk, ch)
        self.assertLess(rel.max(), 2e-2, f"max correlation-normalized diff {rel.max():.2e}")
        # the proposal uses the inverse: compare the marginal widths too. The
        # (A, inc, psi, phi0) block is near-degenerate at 30 d, so 7.6e-3 in the
        # elements becomes up to 20 % (measured) along it -- a proposal-shape
        # error only (M-H corrects); a gross defect is far outside 30 %.
        s_ch = np.sqrt(np.einsum("naa->na", np.linalg.inv(ch)))
        s_lk = np.sqrt(np.einsum("naa->na", np.linalg.inv(lk)))
        np.testing.assert_allclose(s_lk, s_ch, rtol=0.3)

    def test_engine_knob_routes_without_an_in_model_block(self):
        comp = self._comp()
        di = np.arange(self.R, dtype=np.int32)
        with _ref._Env(SIGHET_INFOMAT_ENGINE="lookup"):
            via = np.asarray(comp.information_matrix(
                self.ref, self.holder, inds=INDS, noise_index=di))
        np.testing.assert_allclose(via, self._lookup(comp), rtol=1e-12, atol=0)
        with _ref._Env(SIGHET_INFOMAT_ENGINE="lookup"):
            with self.assertRaises(RuntimeError):
                self._comp(table=False).information_matrix(
                    self.ref, self.holder, inds=INDS, noise_index=di)

    def test_NEGATIVE_CONTROL_a_wrong_noise_row_is_caught(self):
        """The comparison has teeth: scoring against a 2x invC must fail it."""
        ch = self._chunked()
        lk = 2.0 * self._lookup()
        self.assertGreater(self._rel(lk, ch).max(), 2e-2)


if __name__ == "__main__":
    unittest.main()
