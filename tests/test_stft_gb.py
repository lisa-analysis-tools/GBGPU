"""Stage 1 + Stage 2 tests for the STFT/Fresnel galactic-binary likelihood port.

Validates ``gbgpu.gbcomps.STFTGBComputations`` (``gb_stft_{get_ll, fill_global,
swap_ll}`` on the backend, templated ``stft_*_impl<GBTDIonTheFly>`` in LAT's
``lat_stft_kernels.hh``) on the CPU backend.

Stage 1 primary check (self-consistency, exact by construction): the on-the-fly
``get_ll_stft`` (d|h),(h|h) must equal the result of running ``fill_global_stft``
to a template grid and feeding that grid through the *same* STFT-domain inner
product (``STFTDomainWrap.compute_likelihood_terms``, the kernel
``STFTComputationGroup`` uses) -- because ``fill_global`` stores exactly the
per-pixel value ``get_ll`` accumulates. The synthetic data / inverse-noise
arrays are arbitrary: the (d|h),(h|h) agreement is data/invC-independent.

Stage 2 (``get_swap_ll_stft``, the 5 RJMCMC source-swap terms): (a) the
degenerate ``swap(A, A)`` reproduces ``get_ll(A)``'s (d|h),(h|h) in all five
terms (same single-carrier band, same FP order); (b) relabeling add<->remove
swaps the per-track terms and conjugates the cross term ``(h_add|h_remove)``
(Hermitian invC).
"""

import unittest

import numpy as np

from lisatools.domains import get_stft_settings
from lisatools.utils.parallelbase import LISAToolsParallelModule

from gbgpu.testing import stft_information_matrix_reference

try:
    from gbgpu.gbcomps import STFTGBComputations

    HAVE_STFT_GB = True
except (ImportError, ModuleNotFoundError):
    HAVE_STFT_GB = False


class _BackendHolder(LISAToolsParallelModule):
    """Minimal concrete module used to resolve the CPU backend object."""


# GB params (amp, f0, fdot, fddot, phi0, iota, psi, lam, beta) -- the dev_stft.py
# reference source, carrier ~4.23 mHz.
GB_PARAMS = np.array(
    [1e-23, 4.2300812341e-3, 1e-18, 0.0, 0.892342, 1.230980, 3.009081, 4.827342, -0.509234]
)

# A second, distinct source inside the same band (different amp / carrier /
# fdot / angles) -- its side-band overlaps GB_PARAMS so the swap cross term
# (h_add|h_remove) is non-trivial. Used for the add<->remove symmetry test.
GB_PARAMS_B = np.array(
    [2e-23, 4.2800123000e-3, 2e-18, 0.0, 1.100000, 0.700000, 2.200000, 3.500000, 0.300000]
)


class _StftCompsShim:
    """Stand-in for STFTComputationGroup exposing only what STFTGBComputations
    reads: ``cpp_fresnel`` / ``cpp_domain`` / ``d_d``."""

    def __init__(self, cpp_fresnel, cpp_domain, d_d=None):
        self.cpp_fresnel = cpp_fresnel
        self.cpp_domain = cpp_domain
        self.d_d = d_d


@unittest.skipUnless(HAVE_STFT_GB, "requires the GBGPU STFT-GB build")
class _STFTGBFixture(unittest.TestCase):
    """Shared STFT-GB test fixture (no tests of its own): CPU backend, a small
    STFT grid, a synthetic-data / identity-invC ``STFTDomainWrap`` shim, and an
    ``STFTGBComputations`` factory. Subclassed by the Stage-1/2 and Stage-3
    cases so they share one setup."""

    @classmethod
    def setUpClass(cls):
        cls.backend = _BackendHolder(force_backend="cpu").backend
        cls.nch = 3
        cls.tdi_type = cls.backend.TDITypeDict["XYZ"]
        # Small STFT grid: 24 six-hour bins (~6 days), a tight band straddling
        # the GB carrier so the (NT, NF_active) grid stays tiny.
        dt = 10.0
        stft_dt = 6 * 3600.0
        n_stft = 24
        nobs = int(n_stft * stft_dt / dt)
        t = np.arange(nobs) * dt
        cls.settings = get_stft_settings(
            t, stft_dt, min_freq=4.0e-3, max_freq=4.5e-3, force_backend="cpu"
        )
        cls.Tobs = nobs * dt

    def _build_shim(self, window_alpha=0.0, num_data=1):
        """Build cpp_domain (synthetic data + identity invC) + cpp_fresnel."""
        s = self.settings
        NT, NF, nch = s.NT, s.NF_active, self.nch
        rng = np.random.default_rng(1234)
        data = (
            rng.standard_normal((num_data, nch, NT, NF))
            + 1j * rng.standard_normal((num_data, nch, NT, NF))
        ).astype(np.complex128)
        # XYZ inverse-noise: (num_noise, nch, nch, NT, NF); identity in channels.
        invC = np.zeros((num_data, nch, nch, NT, NF), dtype=np.complex128)
        for c in range(nch):
            invC[:, c, c] = 1.0
        data = np.ascontiguousarray(data)
        invC = np.ascontiguousarray(invC)
        domain = self.backend.STFTDomainWrap(
            NT, NF, nch, s.t0, s.min_freq, s.max_freq, s.dt, s.df,
            data.reshape(-1), invC.reshape(-1), num_data, num_data, self.tdi_type,
        )
        fres = self.backend.STFTFresnelWrap(
            NT, NF, nch, s.t0, s.min_freq, s.max_freq, s.dt, s.df,
            window_alpha=window_alpha, use_midpoint=False,
        )
        # Keep data/invC alive (the domain holds raw pointers into them).
        shim = _StftCompsShim(fres, domain)
        shim._keepalive = (data, invC)
        return shim

    def _gb(self, shim, n_side_bins=3, window_factor=1.0, freq_from_tdi_phase=True):
        return STFTGBComputations(
            stft_comps=shim, T=self.Tobs, t_ref=0.0, orbits=None, tdi_config=None,
            force_backend="cpu", n_side_bins=n_side_bins, window_factor=window_factor,
            freq_from_tdi_phase=freq_from_tdi_phase,
        )


class STFTGBStage1Test(_STFTGBFixture):
    def test_backend_exposes_methods(self):
        # GBComputationGroupWrap lives on the *gbgpu* backend (not the lisatools
        # backend used to build the domain/fresnel wraps).
        gb = STFTGBComputations(stft_comps=None, T=1.0, t_ref=0.0, force_backend="cpu")
        w = gb.backend.GBComputationGroupWrap()
        self.assertTrue(hasattr(w, "gb_stft_get_ll"))
        self.assertTrue(hasattr(w, "gb_stft_fill_global"))
        self.assertTrue(hasattr(w, "gb_stft_swap_ll"))

    def test_get_ll_runs_finite(self):
        shim = self._build_shim()
        gb = self._gb(shim)
        gb.get_ll_stft(GB_PARAMS)
        self.assertEqual(gb.d_h_out.shape, (1,))
        self.assertTrue(np.all(np.isfinite(gb.d_h_out)))
        self.assertTrue(np.all(np.isfinite(gb.h_h_out)))
        # h_h must be real-positive with identity invC.
        self.assertGreater(gb.h_h_out[0].real, 0.0)

    def _fill_then_inner_product(self, gb, shim):
        s = self.settings
        templates = np.zeros((1, self.nch, s.NT, s.NF_active), dtype=np.complex128)
        gb.fill_global_stft(
            GB_PARAMS, templates, data_index=np.array([0], dtype=np.int32)
        )
        d_h = np.zeros(1, dtype=np.complex128)
        h_h = np.zeros(1, dtype=np.complex128)
        shim.cpp_domain.compute_likelihood_terms(
            d_h, h_h, templates.reshape(-1),
            np.array([0.0]),            # start_times -> t_idx 0
            np.array([s.min_freq]),     # start_freqs -> f_idx 0 (full active band)
            1,
            np.array([0], dtype=np.int32), np.array([0], dtype=np.int32),
            s.NT, s.NF_active, False,
        )
        return d_h, h_h

    def test_fill_global_matches_get_ll(self):
        """on-the-fly get_ll == fill_global -> domain inner product (rectangular)."""
        for window_alpha in (0.0, 0.5):
            with self.subTest(window_alpha=window_alpha):
                shim = self._build_shim(window_alpha=window_alpha)
                wf = 1.0 if window_alpha == 0.0 else 0.9
                gb = self._gb(shim, n_side_bins=3, window_factor=wf)
                gb.get_ll_stft(GB_PARAMS)
                d_h_b, h_h_b = gb.d_h_out.copy(), gb.h_h_out.copy()
                d_h_a, h_h_a = self._fill_then_inner_product(gb, shim)
                np.testing.assert_allclose(d_h_a, d_h_b, rtol=1e-10, atol=1e-25)
                np.testing.assert_allclose(h_h_a, h_h_b, rtol=1e-10, atol=1e-25)

    def test_freq_from_tdi_phase_ab(self):
        """Both f0/fdot modes run and are finite; the Doppler-corrected result
        differs from the astrophysical-only one (sanity, not magnitude)."""
        shim = self._build_shim()
        gb_corr = self._gb(shim, freq_from_tdi_phase=True)
        gb_astro = self._gb(shim, freq_from_tdi_phase=False)
        gb_corr.get_ll_stft(GB_PARAMS)
        gb_astro.get_ll_stft(GB_PARAMS)
        self.assertTrue(np.all(np.isfinite(gb_corr.d_h_out)))
        self.assertTrue(np.all(np.isfinite(gb_astro.d_h_out)))

    # ---- Stage 2: swap_ll ------------------------------------------------
    def test_swap_degenerate_matches_get_ll(self):
        """swap(A, A): all 5 terms reduce to get_ll(A)'s (d|h),(h|h).

        With identical add/remove params the union band collapses to A's single
        carrier band and add_ip_swap_contrib's add/remove/cross terms evaluate
        the same template, so on the single-thread CPU path the swap terms equal
        the get_ll terms to floating-point order.
        """
        shim = self._build_shim()
        gb = self._gb(shim, n_side_bins=3)
        gb.get_ll_stft(GB_PARAMS)
        d_h, h_h = gb.d_h_out.copy(), gb.h_h_out.copy()

        la, lr, d_h_a, d_h_r, aa, rr, ar = gb.get_swap_ll_stft(GB_PARAMS, GB_PARAMS)
        # (d|h_add) == (d|h_remove) == (d|h)
        np.testing.assert_allclose(d_h_a, d_h, rtol=1e-10, atol=0.0)
        np.testing.assert_allclose(d_h_r, d_h, rtol=1e-10, atol=0.0)
        # (h_add|h_add) == (h_remove|h_remove) == (h_add|h_remove) == (h|h)
        np.testing.assert_allclose(aa, h_h, rtol=1e-10, atol=0.0)
        np.testing.assert_allclose(rr, h_h, rtol=1e-10, atol=0.0)
        np.testing.assert_allclose(ar, h_h, rtol=1e-10, atol=0.0)
        # like_add / like_remove both equal the get_ll(A) likelihood.
        ll = (-0.5 * (h_h - 2.0 * d_h)).real  # d_d == 0 (shim has no d_d)
        np.testing.assert_allclose(la, ll, rtol=1e-10, atol=0.0)
        np.testing.assert_allclose(lr, ll, rtol=1e-10, atol=0.0)

    def test_swap_add_remove_symmetry(self):
        """Relabeling add<->remove swaps the per-track terms and conjugates the
        cross term: with swap(A,B) vs swap(B,A) (same union band both ways),
        d_h_add(B,A)==d_h_remove(A,B), add_add(B,A)==remove_remove(A,B), and
        add_remove(B,A)==conj(add_remove(A,B)) for Hermitian invC."""
        shim = self._build_shim()
        gb = self._gb(shim, n_side_bins=3)
        _, _, dha_ab, dhr_ab, aa_ab, rr_ab, ar_ab = gb.get_swap_ll_stft(
            GB_PARAMS, GB_PARAMS_B
        )
        _, _, dha_ba, dhr_ba, aa_ba, rr_ba, ar_ba = gb.get_swap_ll_stft(
            GB_PARAMS_B, GB_PARAMS
        )
        np.testing.assert_allclose(dha_ba, dhr_ab, rtol=1e-10, atol=0.0)
        np.testing.assert_allclose(dhr_ba, dha_ab, rtol=1e-10, atol=0.0)
        np.testing.assert_allclose(aa_ba, rr_ab, rtol=1e-10, atol=0.0)
        np.testing.assert_allclose(rr_ba, aa_ab, rtol=1e-10, atol=0.0)
        np.testing.assert_allclose(ar_ba, np.conj(ar_ab), rtol=1e-10, atol=0.0)
        # The cross term is genuinely non-trivial (the two sources overlap in
        # band), so the conjugate relation is a real check, not 0 == 0.
        self.assertGreater(np.abs(ar_ab[0]), 0.0)


# Per-parameter central-FD steps for the gradient tests, at sane magnitudes for
# the GB_PARAMS scales. The kernel-vs-NumPy match is independent of the exact
# step (both sides use the identical step + the identical forward evaluation).
GRAD_EPS = np.array([1e-26, 1e-11, 1e-21, 1e-28, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6])


def _numpy_ll_grad(gb, params, eps):
    """Host-side central difference of the STFT log-likelihood, formed from the
    raw (d|h),(h|h) ``get_ll_stft`` stores so it matches the kernel's scalar
    ``q = Re(d|h) - 0.5*Re(h|h)`` bit-for-bit (the -0.5*(d|d) term cancels)."""
    params = np.atleast_2d(np.asarray(params, dtype=np.float64))
    num_bin, nparams = params.shape
    grad = np.zeros((num_bin, nparams))
    for k in range(nparams):
        if eps[k] <= 0.0:
            continue
        pp = params.copy(); pp[:, k] += eps[k]
        gb.get_ll_stft(pp)
        q_p = gb.d_h_out.real - 0.5 * gb.h_h_out.real
        pm = params.copy(); pm[:, k] -= eps[k]
        gb.get_ll_stft(pm)
        q_m = gb.d_h_out.real - 0.5 * gb.h_h_out.real
        grad[:, k] = (q_p - q_m) / (2.0 * eps[k])
    return grad


def _swap_S(gb, pa, pr):
    """The swap scalar the swap gradient differentiates, from the 5 raw terms:
    S = Re(d|h_add) - Re(d|h_remove) - 0.5(h_add|h_add) - 0.5(h_remove|h_remove)
        + Re(h_add|h_remove)."""
    _, _, dha, dhr, aa, rr, ar = gb.get_swap_ll_stft(pa, pr)
    return (dha - dhr - 0.5 * aa - 0.5 * rr + ar).real


def _numpy_swap_grad(gb, pa, pr, eps_add, eps_remove):
    pa = np.atleast_2d(np.asarray(pa, dtype=np.float64))
    pr = np.atleast_2d(np.asarray(pr, dtype=np.float64))
    num_bin, nparams = pa.shape
    grad_add = np.zeros((num_bin, nparams))
    grad_rem = np.zeros((num_bin, nparams))
    for k in range(nparams):
        if eps_add[k] > 0.0:
            pap = pa.copy(); pap[:, k] += eps_add[k]
            pam = pa.copy(); pam[:, k] -= eps_add[k]
            grad_add[:, k] = (_swap_S(gb, pap, pr) - _swap_S(gb, pam, pr)) / (2.0 * eps_add[k])
        if eps_remove[k] > 0.0:
            prp = pr.copy(); prp[:, k] += eps_remove[k]
            prm = pr.copy(); prm[:, k] -= eps_remove[k]
            grad_rem[:, k] = (_swap_S(gb, pa, prp) - _swap_S(gb, pa, prm)) / (2.0 * eps_remove[k])
    return grad_add, grad_rem


class STFTGBStage3Test(_STFTGBFixture):
    """Stage 3: ``get_ll_grad_stft`` / ``get_swap_ll_grad_stft`` reproduce a
    host-side central difference of ``get_ll_stft`` / ``get_swap_ll_stft``. The
    kernels reuse the Stage-1/2 forward evaluation (``stft_eval_block_*``) for
    each perturbed point, so the agreement is to ~machine precision."""

    def test_get_ll_grad_matches_numpy(self):
        shim = self._build_shim()
        gb = self._gb(shim, n_side_bins=3)
        grad = gb.get_ll_grad_stft(GB_PARAMS, param_eps=GRAD_EPS)
        ref = _numpy_ll_grad(gb, GB_PARAMS, GRAD_EPS)
        self.assertEqual(grad.shape, (1, 9))
        self.assertTrue(np.all(np.isfinite(grad)))
        np.testing.assert_allclose(grad, ref, rtol=1e-6, atol=1e-25)
        # Genuinely non-trivial (not all ~0) -- a real check, not 0 == 0.
        self.assertGreater(np.max(np.abs(grad)), 0.0)

    def test_get_ll_grad_freezes_on_nonpositive_eps(self):
        shim = self._build_shim()
        gb = self._gb(shim, n_side_bins=3)
        eps = GRAD_EPS.copy()
        eps[0] = 0.0       # freeze amp
        eps[4] = -1.0      # freeze phi0
        grad = gb.get_ll_grad_stft(GB_PARAMS, param_eps=eps)
        self.assertEqual(grad[0, 0], 0.0)
        self.assertEqual(grad[0, 4], 0.0)
        self.assertNotEqual(grad[0, 1], 0.0)   # f0 still computed

    def test_swap_ll_grad_matches_numpy(self):
        shim = self._build_shim()
        gb = self._gb(shim, n_side_bins=3)
        grad_add, grad_rem = gb.get_swap_ll_grad_stft(
            GB_PARAMS, GB_PARAMS_B,
            param_eps_add=GRAD_EPS, param_eps_remove=GRAD_EPS,
        )
        ref_add, ref_rem = _numpy_swap_grad(gb, GB_PARAMS, GB_PARAMS_B, GRAD_EPS, GRAD_EPS)
        self.assertEqual(grad_add.shape, (1, 9))
        self.assertEqual(grad_rem.shape, (1, 9))
        self.assertTrue(np.all(np.isfinite(grad_add)) and np.all(np.isfinite(grad_rem)))
        np.testing.assert_allclose(grad_add, ref_add, rtol=1e-6, atol=1e-25)
        np.testing.assert_allclose(grad_rem, ref_rem, rtol=1e-6, atol=1e-25)
        self.assertGreater(np.max(np.abs(grad_add)), 0.0)
        self.assertGreater(np.max(np.abs(grad_rem)), 0.0)

    def test_swap_ll_grad_freezes_on_nonpositive_eps(self):
        shim = self._build_shim()
        gb = self._gb(shim, n_side_bins=3)
        eps_add = GRAD_EPS.copy(); eps_add[2] = 0.0     # freeze add fdot
        eps_rem = GRAD_EPS.copy(); eps_rem[7] = -1.0    # freeze remove lam
        grad_add, grad_rem = gb.get_swap_ll_grad_stft(
            GB_PARAMS, GB_PARAMS_B,
            param_eps_add=eps_add, param_eps_remove=eps_rem,
        )
        self.assertEqual(grad_add[0, 2], 0.0)
        self.assertEqual(grad_rem[0, 7], 0.0)


# --- F-stat (Stage 4) Cornish & Crowder '05 basis filters -------------------
# 4 fixed extrinsic param sets; intrinsic (f0, fdot, fddot, lam, beta) copied.
# GB param slots: 0=amp(A), 4=phi0, 5=iota, 6=psi.
_FSTAT_A    = (2.0, 2.0, 2.0, 2.0)
_FSTAT_IOTA = (np.pi / 2, np.pi / 2, np.pi / 2, np.pi / 2)
_FSTAT_PSI  = (0.0, np.pi / 4, 0.0, np.pi / 4)
_FSTAT_PHI0 = (0.0, np.pi, 3 * np.pi / 2, np.pi / 2)


def _fstat_basis(params, fi):
    """The fi-th F-stat basis filter params (extrinsic slots overwritten)."""
    p = np.asarray(params, dtype=float).copy()
    p[0] = _FSTAT_A[fi]
    p[4] = _FSTAT_PHI0[fi]
    p[5] = _FSTAT_IOTA[fi]
    p[6] = _FSTAT_PSI[fi]
    return p


def _fstat_reference(gb, params):
    """Reference N (4) and upper-tri M (10) from the validated get_ll / swap.

    N_i = Re(d|A_i), M_ii = Re(A_i|A_i) from ``get_ll_stft``; M_ij = Re(A_i|A_j)
    from ``get_swap_ll_stft``'s add_remove term. The upper-triangle flatten
    matches the kernel's m_idx == ``numpy.triu_indices(4)`` order.
    """
    basis = [_fstat_basis(params, fi) for fi in range(4)]
    N_ref = np.zeros(4)
    M4 = np.zeros((4, 4))
    for i in range(4):
        gb.get_ll_stft(basis[i])
        N_ref[i] = gb.d_h_out[0].real
        M4[i, i] = gb.h_h_out[0].real
    for i in range(4):
        for j in range(i + 1, 4):
            _, _, _, _, _, _, ar = gb.get_swap_ll_stft(basis[i], basis[j])
            M4[i, j] = ar[0].real
    return N_ref, M4[np.triu_indices(4)]


class STFTGBStage4Test(_STFTGBFixture):
    """F-statistic: N=(d|A_i), M=(A_i|A_j) upper-tri, 2F = N^T M^-1 N."""

    def test_backend_exposes_fstat(self):
        gb = STFTGBComputations(stft_comps=None, T=1.0, t_ref=0.0, force_backend="cpu")
        self.assertTrue(
            hasattr(gb.backend.GBComputationGroupWrap(), "gb_stft_get_fstat_ll"))

    def test_fstat_NM_match_get_ll_and_swap(self):
        """Kernel N, M == the same terms from get_ll_stft / get_swap_ll_stft.

        The F-stat kernel calls the very same stft_eval_block_ll /
        stft_eval_block_swap device helpers, so N/M are byte-identical to the
        validated Stage-1/2 kernels (machine precision)."""
        shim = self._build_shim()
        gb = self._gb(shim, n_side_bins=3)
        N, M = gb.get_fstat_ll_stft(GB_PARAMS)
        self.assertEqual(N.shape, (1, 4))
        self.assertEqual(M.shape, (1, 10))
        N_ref, M_ref = _fstat_reference(gb, GB_PARAMS)
        scale = max(np.abs(N_ref).max(), np.abs(M_ref).max())
        self.assertGreater(scale, 0.0)   # templates carry power -- a real check
        np.testing.assert_allclose(N[0], N_ref, rtol=1e-9, atol=1e-12 * scale)
        np.testing.assert_allclose(M[0], M_ref, rtol=1e-9, atol=1e-12 * scale)

    def test_fstat_2F_well_formed(self):
        """M is symmetric positive-definite; 2F = N^T M^-1 N is finite, >= 0."""
        shim = self._build_shim()
        gb = self._gb(shim, n_side_bins=3)
        N, M = gb.get_fstat_ll_stft(GB_PARAMS)
        M4 = np.zeros((4, 4))
        M4[np.triu_indices(4)] = M[0]
        M4 = M4 + M4.T - np.diag(np.diag(M4))
        self.assertTrue(np.all(np.linalg.eigvalsh(M4) > 0.0))   # 4 independent filters
        two_F = STFTGBComputations.fstat_2F(N, M)
        self.assertEqual(two_F.shape, (1,))
        self.assertTrue(np.isfinite(two_F[0]))
        self.assertGreaterEqual(two_F[0], 0.0)


class _FisherGroupShim(_StftCompsShim):
    """The shim plus what the reference Fisher reads from a computation group."""

    def __init__(self, shim, invC, settings, num_noise, nch):
        super().__init__(shim.cpp_fresnel, shim.cpp_domain)
        self._keepalive = shim._keepalive
        self.invC_arr = invC
        self.settings = settings
        self.num_noise = num_noise
        self.num_channels = nch

    def _owned_cpp_array(self, name, arr, dtype):
        return np.ascontiguousarray(arr, dtype=dtype)


class STFTGBFisherKernelTest(_STFTGBFixture):
    """The Fisher kernel against the retained reference Fisher (F1 to F6).

    Tapered window, midpoint anchor and a random Hermitian positive-definite inverse CSD, so the
    cross-channel contraction and the shared Fresnel kernel are both exercised. The metric is
    |G_kernel - G_ref| / sqrt(G_ii G_jj): off-diagonal entries pass through zero.
    """

    TEST_INDS = [0, 1, 2, 4, 5, 6, 7, 8]
    # ? Measured 2.9e-15 (four-point) and 4.6e-15 (two-point) on this fixture, 2026-09-15.
    BOUND = 1e-13

    def _fisher_case(self, start_bin=0, trim_bins=0, cross_channel=True):
        s, nch, num_noise = self.settings, self.nch, 2
        NT, NF_full = s.NT, s.NF_active
        rng = np.random.default_rng(7)
        if cross_channel:
            raw = (rng.standard_normal((num_noise, NT, NF_full, nch, nch))
                   + 1j * rng.standard_normal((num_noise, NT, NF_full, nch, nch)))
            hpd = raw @ np.conj(np.swapaxes(raw, -1, -2)) + nch * np.eye(nch)
            invC_full = np.transpose(hpd, (0, 3, 4, 1, 2))
        else:
            invC_full = rng.uniform(0.5, 2.0, (num_noise, nch, NT, NF_full)).astype(np.complex128)
        # A window is the full-grid inverse CSD sliced in frequency.
        NF = NF_full - start_bin - trim_bins
        invC = np.ascontiguousarray(invC_full[..., start_bin:start_bin + NF])
        data = np.zeros((num_noise, nch, NT, NF), dtype=np.complex128)
        tdi_type = self.backend.TDITypeDict["XYZ" if cross_channel else "AET"]
        domain = self.backend.STFTDomainWrap(
            NT, NF, nch, s.t0, s.min_freq, s.max_freq, s.dt, s.df,
            data.reshape(-1), invC.reshape(-1), num_noise, num_noise, tdi_type,
        )
        fres = self.backend.STFTFresnelWrap(
            NT, NF, nch, s.t0, s.min_freq, s.max_freq, s.dt, s.df,
            window_alpha=0.5, use_midpoint=True,
        )
        shim = _StftCompsShim(fres, domain)
        shim._keepalive = (data, invC)
        grid = type("Grid", (), {"NT": NT, "NF_active": NF, "df": s.df})()
        group = _FisherGroupShim(shim, invC, grid, num_noise, nch)
        return self._gb(group, n_side_bins=3), group

    def _params(self, gb):
        return np.stack([GB_PARAMS, GB_PARAMS_B]), np.array([0, 1], dtype=np.int32)

    def _boundary_pair(self):
        """A source whose astro carrier sits on a half-bin boundary at one anchor, and the same
        source a quarter bin away.

        On the boundary the f0 and fdot steps move a stencil carrier across the bin. The kernel gives
        all four templates of a derivative the union of their supports, so the two sources must give
        almost the same Fisher; with each template on its own support the boundary one was 1e3 to 2e4
        times larger. The reference keeps the own-support rule, so it is not comparable here.
        """
        s = self.settings
        boundary = GB_PARAMS.copy()
        t_anchor = s.t0 + 12.5 * s.dt
        boundary[1] = s.min_freq + 5.5 * s.df - boundary[2] * t_anchor
        away = boundary.copy()
        away[1] += 0.25 * s.df
        return np.stack([boundary, away]), np.array([0, 0], dtype=np.int32)

    def _assert_parity(self, kernel, reference):
        diagonal = np.abs(np.diagonal(reference, axis1=1, axis2=2))
        self.assertTrue(np.all(diagonal > 0.0))
        norm = np.sqrt(diagonal[:, :, None] * diagonal[:, None, :])
        worst = float(np.max(np.abs(kernel - reference) / norm))
        self.assertLess(worst, self.BOUND, msg=f"kernel left the reference by {worst:.3e}")

    def test_f1_f4_parity_and_symmetry(self):
        gb, group = self._fisher_case()
        params, noise_index = self._params(gb)
        for kwargs in ({}, {"inds": self.TEST_INDS}, {"inds": self.TEST_INDS, "astro_track": False}):
            with self.subTest(**{k: str(v) for k, v in kwargs.items()}):
                kernel = gb.information_matrix(params, group, noise_index=noise_index, **kwargs)
                reference = stft_information_matrix_reference(
                    gb, params, group, noise_index=noise_index, **kwargs)
                self._assert_parity(kernel, reference)
                np.testing.assert_array_equal(kernel, np.swapaxes(kernel, 1, 2))

    def test_f2_easy_central_difference(self):
        gb, group = self._fisher_case()
        params, noise_index = self._params(gb)
        kwargs = dict(inds=self.TEST_INDS, noise_index=noise_index, easy_central_difference=True)
        self._assert_parity(gb.information_matrix(params, group, **kwargs),
                            stft_information_matrix_reference(gb, params, group, **kwargs))

    def test_diagonal_inverse_csd(self):
        gb, group = self._fisher_case(cross_channel=False)
        params, noise_index = self._params(gb)
        kwargs = dict(inds=self.TEST_INDS, noise_index=noise_index)
        self._assert_parity(gb.information_matrix(params, group, **kwargs),
                            stft_information_matrix_reference(gb, params, group, **kwargs))

    def test_f3_subset_is_submatrix(self):
        gb, group = self._fisher_case()
        params, noise_index = self._params(gb)
        subset = [1, 4, 7]
        full = gb.information_matrix(params, group, noise_index=noise_index)
        sub = gb.information_matrix(params, group, inds=subset, noise_index=noise_index)
        np.testing.assert_array_equal(sub, full[:, subset][:, :, subset])

    def test_f5_window_reproduces_full_grid(self):
        gb, group = self._fisher_case()
        gb_w, group_w = self._fisher_case(start_bin=1, trim_bins=1)
        params, noise_index = self._params(gb)
        # Only the first two sources: their supports lie inside [1, NF - 1).
        full = gb.information_matrix(params[:2], group, inds=self.TEST_INDS,
                                     noise_index=noise_index[:2])
        windowed = gb_w.information_matrix(params[:2], group_w, inds=self.TEST_INDS,
                                           noise_index=noise_index[:2],
                                           start_freq_inds=np.array([1, 1], dtype=np.int32))
        np.testing.assert_array_equal(windowed, full)

    def test_f6_nonpositive_step_freezes(self):
        gb, group = self._fisher_case()
        params, noise_index = self._params(gb)
        eps = gb._information_matrix_steps(9, None, None, 1e-6)[0].copy()
        eps[5] = 0.0
        info = gb.information_matrix(params, group, inds=self.TEST_INDS,
                                     noise_index=noise_index, param_eps=eps)
        row = self.TEST_INDS.index(5)
        self.assertEqual(float(np.abs(info[:, row, :]).max()), 0.0)
        self.assertEqual(float(np.abs(info[:, :, row]).max()), 0.0)
        self.assertGreater(float(np.abs(info[:, 0, 0]).min()), 0.0)

    def test_rejects_out_of_range_inds(self):
        gb, group = self._fisher_case()
        params, noise_index = self._params(gb)
        with self.assertRaises(ValueError):
            gb.information_matrix(params, group, inds=[0, 9], noise_index=noise_index)

    def test_carrier_crossing_does_not_inflate(self):
        gb, group = self._fisher_case()
        params, noise_index = self._boundary_pair()
        info = np.asarray(gb.information_matrix(params, group, inds=self.TEST_INDS,
                                                noise_index=noise_index))
        diagonal = np.abs(np.diagonal(info, axis1=1, axis2=2))
        ratio = diagonal[0] / diagonal[1]
        # ? A boundary source keeps slightly less of itself inside the +-n_side_bins cut, so the ratio
        # ? sits just below 1 and rises toward it with n_side_bins; the defect this guards against was
        # ? a factor 1e3 or more on the f0 and fdot entries.
        for k, name in enumerate(("amp", "f0", "fdot", "phi0", "iota", "psi", "lam", "beta")):
            with self.subTest(parameter=name):
                self.assertGreater(ratio[k], 0.5, msg=f"{name} diagonal collapsed at a crossing")
                self.assertLess(ratio[k], 2.0, msg=f"{name} diagonal inflated at a crossing: {ratio[k]:.3e}")


def _time_domain_pixel(f0, fdot0, f, t0, dt, window_alpha, use_midpoint=True, sample=0.5):
    """The integral the Fresnel evaluator represents, summed in longdouble on a fine time grid.

    V(f) = e^{-2 pi i f (t_ref - t0)} int w(t - t0) exp(i[pi fdot0 tau^2 + 2 pi (f0 - f) tau]) dt,
    tau = t - t_ref. Independent of the Fresnel machinery, so it catches an error in the evaluator
    itself rather than only a change in it.
    """
    ld = np.longdouble
    n = int(round(dt / sample)) + 1
    x = np.arange(n, dtype=ld) * ld(sample)
    taper = ld(window_alpha) * ld(dt) / 2
    w = np.ones(n, dtype=ld)
    if taper > 0:
        rise, fall = x < taper, x > ld(dt) - taper
        w[rise] = (1 - np.cos(np.pi * x[rise] / taper)) / 2
        w[fall] = (1 - np.cos(np.pi * (ld(dt) - x[fall]) / taper)) / 2
    tau = x - (ld(dt) / 2 if use_midpoint else ld(0.0))
    phase = np.pi * ld(fdot0) * tau * tau + 2 * np.pi * (ld(f0) - ld(f)) * tau
    integrand = w * (np.cos(phase) + 1j * np.sin(phase))
    integral = (np.sum(integrand) - 0.5 * (integrand[0] + integrand[-1])) * ld(sample)
    front = np.exp(-2j * np.pi * ld(f) * (ld(dt) / 2 if use_midpoint else ld(0.0)))
    return complex(front * integral)


@unittest.skipUnless(HAVE_STFT_GB, "requires the GBGPU STFT-GB build")
class STFTFresnelAccuracyTest(unittest.TestCase):
    """The evaluator against the integral it represents, over the fdot range the population spans.

    Replaces an earlier bit-identity check. The evaluator writes a pixel as
    exp(-i pi fdot0 zeta^2) [C + i s S](v); both factors carry ~pi (f0 - f)^2 / fdot0, up to 1e12 rad,
    and cancel analytically. Forming them apart cost up to 1e-4 of the value at small |fdot0| and far
    more of its f0 derivative, which is what this test pins down.
    """

    DT = 86400.0
    T0 = 30.0 * 86400.0
    ALPHA = 1.0 / (10e-3 * 86400.0)   # production taper: band start 10 mHz
    BOUND = 1e-11

    @classmethod
    def setUpClass(cls):
        cls.backend = _BackendHolder(force_backend="cpu").backend

    def test_pixel_values_match_the_time_domain_integral(self):
        df = 1.0 / self.DT
        fresnel = self.backend.STFTFresnelWrap(
            1, 1, 3, self.T0, 0.0, 1.0, self.DT, df,
            window_alpha=self.ALPHA, use_midpoint=True,
        )
        k0 = int(round(3.0e-3 / df))
        f0 = k0 * df
        for magnitude in (1e-19, 1e-17, 1e-15):
            for sign in (1.0, -1.0):
                for offset in (0.25, 10.25):
                    fdot0 = sign * magnitude
                    f = (k0 - offset) * df
                    with self.subTest(fdot0=fdot0, offset=offset):
                        out = np.zeros(1, dtype=np.complex128)
                        fresnel.compute_fourier_values(
                            out, np.ones(1), np.zeros(1), np.array([f0]), np.array([fdot0]),
                            np.array([self.T0]), np.array([f]), 1.0, 1, 1)
                        truth = _time_domain_pixel(f0, fdot0, f, self.T0, self.DT, self.ALPHA)
                        error = abs(out[0] - truth) / abs(truth)
                        self.assertLess(error, self.BOUND,
                                        msg=f"pixel value off the integral by {error:.3e}")

    def test_phase_kernel_and_moment_match_their_integrals(self):
        """The single-interval pair the evaluator is built on, against a direct quadrature.

        kernel = sqrt(2|fdot0|) int e^{i psi} dtau and moment = the same with a tau weight, both over
        one interval with tau from t_ref. The cases cover the three regimes the evaluator branches on:
        endpoints far from the stationary point, the stationary point inside the interval, and an
        endpoint within 1.6 Fresnel units of it. The moment is what the linear envelope multiplies;
        its own conditioning fails differently from the kernel's, so both are checked.
        """
        df = 1.0 / self.DT
        fresnel = self.backend.STFTFresnelWrap(
            1, 1, 3, self.T0, 0.0, 1.0, self.DT, df,
            window_alpha=self.ALPHA, use_midpoint=True,
        )
        k0 = int(round(3.0e-3 / df))
        f0, t_ref = k0 * df, self.T0 + 0.5 * self.DT
        cases = []
        for magnitude in (1e-19, 1e-17, 1e-15):
            for sign in (1.0, -1.0):
                fdot0 = sign * magnitude
                cases.append(("far", fdot0, (k0 - 10.25) * df))
                cases.append(("stationary inside", fdot0, f0 + fdot0 * 0.5 * self.DT))
                cases.append(("series endpoint", fdot0,
                              f0 + fdot0 * (t_ref - self.T0) + 0.4 * np.sqrt(abs(fdot0) / 2.0)))
        n = len(cases)
        kernel = np.zeros(n, dtype=np.complex128)
        moment = np.zeros(n, dtype=np.complex128)
        fresnel.compute_phase_kernel_moments(
            kernel, moment, np.array([c[2] for c in cases]), np.full(n, t_ref), np.full(n, f0),
            np.array([c[1] for c in cases]), np.full(n, self.T0), np.full(n, self.T0 + self.DT),
            np.full(n, self.T0), n)
        for i, (label, fdot0, f_eff) in enumerate(cases):
            k_ref, m_ref = _quadrature_pair(f_eff, t_ref, f0, fdot0, self.T0, self.T0 + self.DT, self.T0)
            with self.subTest(case=label, fdot0=fdot0):
                self.assertLess(abs(kernel[i] - k_ref) / abs(k_ref), self.BOUND)
                # ? The moment vanishes by symmetry at delta_f = 0, so it is judged against the scale
                # ? it enters the envelope correction with, |kernel| * DT.
                scale = max(abs(m_ref), abs(k_ref) * self.DT)
                self.assertLess(abs(moment[i] - m_ref) / scale, 1e-6)


def _quadrature_pair(f_eff, t_ref, f0, fdot0, t_start, t_end, t_origin, pieces=16384):
    """kernel and moment of one interval by Simpson quadrature in longdouble.

    ! The piece count sets the floor of this test: Simpson error falls as h^4, and 4096 pieces leave
    ! 2e-11 on a pixel ten bins off the tone, which is above the evaluator's own error.
    """
    ld = np.longdouble
    tau = np.linspace(ld(t_start) - ld(t_ref), ld(t_end) - ld(t_ref), 2 * pieces + 1, dtype=ld)
    phase = np.pi * ld(fdot0) * tau * tau + 2 * np.pi * (ld(f0) - ld(f_eff)) * tau
    values = np.cos(phase) + 1j * np.sin(phase)
    weights = np.ones(2 * pieces + 1, dtype=ld)
    weights[1:-1:2], weights[2:-1:2] = 4.0, 2.0
    step = (tau[-1] - tau[0]) / (2 * pieces)
    root = np.sqrt(2.0 * abs(ld(fdot0)))
    front = np.exp(-2j * np.pi * ld(f_eff) * (ld(t_ref) - ld(t_origin)))
    integral = np.sum(weights * values) * step / 3
    moment = np.sum(weights * tau * values) * step / 3
    return complex(front * root * integral), complex(front * root * moment)


if __name__ == "__main__":
    unittest.main()
