"""F-stat basis-filter 4->2 fold: parity of the folded kernel vs the unfolded one.

The chunked-WDM F-stat kernel builds 4 Cornish & Crowder '05 basis filters at
fixed ``(A, iota, psi, phi0) = (2, pi/2, {0, pi/4, 0, pi/4}, {0, pi, 3pi/2,
pi/2})``. Those are 2 polarization directions x 2 phase quadratures: filters
(0, 2) share ``psi = 0`` and (1, 3) share ``psi = pi/4``, so each pair differs
ONLY in ``phi0``. The analytic TDI the kernel builds is an EXACT phasor in
``phi0`` (``get_hp_hc`` forms the complex sample from the ``[0, pi/2]``
quadrature pair as ``-e^{-i Phi}`` with ``Phi = -phi0 + 2 pi (f0 dt + ...)``),
so only TWO waveform generations are needed and the quadrature partners follow
by an exact constant rotation.

Contracts pinned here:

* **Parity** -- folded and unfolded ``(N, M)`` are equal in exact arithmetic
  and differ only by floating-point reassociation, so they are compared at the
  fused-kernel tolerance (``rtol=1e-9`` + ``_dyn_atol``), NEVER for bit
  equality.
* **Signed** -- the comparison is per-element and signed. This is what pins the
  rotation sign, and it is the ONLY thing that can: a conjugated sign is the
  similarity ``N -> D N``, ``M -> D M D`` with ``D = diag(1, 1, -1, -1)``, and
  ``F = N^T M^-1 N`` is INVARIANT under it (``test_sign_error_is_invisible_in_F``
  demonstrates this). A wrong sign would leave F and ln_snr perfect while
  silently corrupting the recovered ``(phi0_max, psi_max)`` of every F-stat
  birth.
* **Default ON** (2026-08-28) -- ``fstat_fold`` defaults to 1. The fold is
  exact, so it is the production path; ``GB_FSTAT_FOLD=0`` restores the
  unfolded 4-generation path bit-for-bit as a reference.
* **End-to-end** -- ``fstat_maximized_extrinsics`` agrees on the folded path,
  ``F`` included.

Backend-agnostic: the fixture allocates through the COMP's own array module,
so this runs on whichever backend ``GBWDMComputations`` resolves. Same small
grid as ``test_phase_max_fused``.
"""

import os
import unittest

import numpy as np

from lisatools.detector import ESAOrbits
from lisatools.utils.utility import get_array_module
from lisatools.domains import WDMSettings
from lisatools.sampling.fstat_proposal import fstat_maximized_extrinsics
from lisatools.utils.constants import YRSID_SI

from gbgpu.gbcomps import GBWDMComputations

#: folded-vs-unfolded tolerance. The fold is EXACT in exact arithmetic (the
#: rotation is applied as sign flips / real-imag swaps, which are themselves
#: exact in floating point), so the only residual is the reassociation of the
#: waveform generation: the unfolded path evaluates cos/sin at ``Phi + phi0``
#: while the folded path evaluates at ``Phi`` and rotates. Same absolute-floor
#: reasoning as test_phase_max_fused: rounding scales with the BATCH's largest
#: accumulation, so small rows need a floor pinned to the batch max.
RTOL = 1e-9
ANGLE_ATOL = 1e-6

#: filter -> (stage, alpha) of the fold, mirrored on the host for the
#: documentation test below. alpha_i = e^{-i phi0_i} over
#: phi0 = {0, pi, 3pi/2, pi/2}.
ALPHA = np.array([1.0 + 0.0j, -1.0 + 0.0j, 0.0 + 1.0j, 0.0 - 1.0j])

_TRIU = [(0, 0), (0, 1), (0, 2), (0, 3), (1, 1),
         (1, 2), (1, 3), (2, 2), (2, 3), (3, 3)]


class _TwoSlotHolder:
    """Minimal wdm_holder: N buffer slots (residual slab + XYZ invC slab).

    The kernel dereferences these as DEVICE pointers on a GPU backend, so the
    contiguity pass must go through the slabs' OWN array module. Hard-coding
    ``np.ascontiguousarray`` here raised ``TypeError: Implicit conversion to a
    NumPy array is not allowed`` out of cupy and took down setUpClass before a
    single test ran.
    """

    def __init__(self, data_slabs, invC_slabs):
        xp = get_array_module(data_slabs)
        self.linear_data_arr = [xp.ascontiguousarray(data_slabs).ravel()]
        self.linear_psd_arr = [xp.ascontiguousarray(invC_slabs).ravel()]

    def __len__(self):
        return 1


def _host(a):
    """Device-or-host array -> numpy, for the host-side assertions.

    ``.get()`` is an exact copy, so the bitwise compares below stay bitwise.
    """
    return a.get() if hasattr(a, "get") else np.asarray(a)


def _dyn_atol(*arrays):
    """Absolute floor scaled to the batch max -- see test_phase_max_fused."""
    m = max(float(np.max(np.abs(np.asarray(a)))) for a in arrays)
    return 1e-9 * max(m, 1e-300)


def _assert_angles_close(a, b, atol=ANGLE_ATOL, msg=""):
    d = np.mod(np.asarray(a) - np.asarray(b) + np.pi, 2 * np.pi) - np.pi
    np.testing.assert_allclose(d, 0.0, atol=atol, err_msg=msg)


def build_fixture():
    """Chunked comp + holder + scoring batch on a small CPU WDM grid.

    Module-level (not just a ``setUpClass``) so the golden-capture script used
    for the bit-identity regression can build the IDENTICAL batch.
    """
    backend = "cpu"
    dt = 10.0
    Nf, Nt = 256, 512
    t_start = int(0.5 * YRSID_SI / dt) * dt
    layer_df = 1.0 / (2.0 * Nf * dt)
    edge = 40

    orbits = ESAOrbits(force_backend=backend)
    wdm_set = WDMSettings(
        Nf, Nt, dt, t0=t_start,
        min_freq=1e-4, max_freq=2e-2,
        min_time=edge * Nf * dt, max_time=(Nt - edge) * Nf * dt,
        force_backend=backend,
    )
    chunked = GBWDMComputations(
        wdm_set, t_ref=t_start,
        Nt_sub=128, n_pad=16, N_sparse=256,
        N_cp_sig=0, N_cp_orbit=0,
        orbits=orbits, tdi_config="2nd generation",
        force_backend=backend, d_d=0.0, tdi_type="XYZ",
    )
    chunked.convert_to_ra_dec = False
    # Allocate through the COMP's own array module, not hard-coded numpy.
    # ``force_backend="cpu"`` above is a request, not a guarantee: on a
    # GPU-only install it can still resolve to a cupy-backed comp, and
    # ``fill_global_wdm`` then asserts ``isinstance(templates,
    # self.xp.ndarray)`` and fails in setUpClass before a single fold test
    # runs. Building the slabs with ``chunked.xp`` makes the fixture work
    # on either backend, which is what lets the parity gate run on the
    # cluster GPU -- the place where it actually matters.
    xp = chunked.xp

    f0_A = (int(3e-3 / layer_df) + 0.37) * layer_df
    f0_C = (int(5e-3 / layer_df) + 0.62) * layer_df
    A = np.array([1e-21, f0_A, 1e-17, 0.0, 1.2, 0.7, 0.4, 2.0, 0.5])
    C = np.array([8e-22, f0_C, 2e-17, 0.0, 0.4, 1.1, 0.9, 4.0, -0.3])

    ilo, ihi = wdm_set.ind_min_f, wdm_set.ind_max_f + 1
    slabs, invCs = [], []
    for p in (A, C):
        h = xp.zeros((3, Nf, Nt))
        chunked.fill_global_wdm(p[None, :], h, convert_to_ra_dec=False)
        h_act = xp.ascontiguousarray(h[:, ilo:ihi, wdm_set.active_slice_t])
        slabs.append(h_act)
        nch, nfa, nta = h_act.shape
        invC = xp.zeros((nch, nch, nfa, nta))
        for c in range(nch):
            invC[c, c] = 1.0
        invCs.append(invC)
    holder = _TwoSlotHolder(xp.stack(slabs), xp.stack(invCs))

    # Scoring batch: the two references + jittered copies spanning generic
    # phases, amplitudes and small f0 offsets (so the fold is exercised at
    # on-peak AND off-peak (N, M) scales, not just the trivially large rows).
    rng = np.random.default_rng(20260828)
    rows = [A, C]
    for _ in range(3):
        for p in (A, C):
            q = p.copy()
            q[0] *= 1.0 + 0.15 * rng.standard_normal()
            q[1] += 0.05 * layer_df * rng.standard_normal()
            q[4] = rng.uniform(0.0, 2 * np.pi)   # phi0
            q[5] = rng.uniform(0.0, np.pi)       # iota
            q[6] = rng.uniform(0.0, np.pi)       # psi
            rows.append(q)
    params = np.stack(rows)
    di = np.arange(params.shape[0], dtype=np.int32) % 2
    return chunked, holder, params, di


class FstatFilterFoldTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.chunked, cls.holder, cls.params, cls.di = build_fixture()

    def _NM(self, fold):
        """(N, M) for one fold setting, pulled to the HOST.

        Every assertion below is host arithmetic (``np.testing``, the D-flip
        premise test, ``fstat_maximized_extrinsics``). Pulling once here keeps
        the device out of the comparison logic entirely; ``.get()`` is exact so
        the bitwise tests remain bitwise.
        """
        N, M = self.chunked.get_fstat_ll_wdm(
            self.params, self.holder,
            data_index=self.di, noise_index=self.di,
            convert_to_ra_dec=False, fstat_fold=fold)
        return _host(N), _host(M)

    # ---- default-off contract ----------------------------------------

    def test_default_is_folded(self):
        """The shipped default is now the FOLDED path (user ruling
        2026-08-28, promoted from opt-in once parity passed on GPU).

        The fold is exact, so there is no reason to default to the slower
        path. ``GB_FSTAT_FOLD=0`` still restores it bit-for-bit -- which is
        what ``test_default_call_equals_explicit_off_bitwise`` and the
        parity tests below exercise.
        """
        self.assertIsNone(os.environ.get("GB_FSTAT_FOLD"),
                          "GB_FSTAT_FOLD leaked into the test environment")
        self.assertEqual(int(GBWDMComputations.fstat_fold), 1)
        self.assertEqual(int(self.chunked.fstat_fold), 1)
        # ... and that default is what an unspecified call sends to C++.
        self.assertEqual(
            self.chunked._fstat_fold_kernel_args(None), (1,))
        self.assertEqual(
            self.chunked._fstat_fold_kernel_args(1), (1,))

    def test_default_call_equals_the_default_path_bitwise(self):
        """``fstat_fold=None`` must resolve to the SHIPPED default path.

        Bit equality is required here (same branch, same arithmetic), so
        an unspecified call is provably the same code as an explicit one.
        The default is now 1 (2026-08-28); ``_fstat_fold_kernel_args`` is
        the single place that resolves None, and this pins that it agrees
        with what the kernel actually runs rather than only with the
        attribute.
        """
        default = int(self.chunked._fstat_fold_kernel_args(None)[0])
        N_d, M_d = self._NM(None)
        N_0, M_0 = self._NM(default)
        np.testing.assert_array_equal(np.asarray(N_d), np.asarray(N_0))
        np.testing.assert_array_equal(np.asarray(M_d), np.asarray(M_0))

    # ---- the parity gate ---------------------------------------------

    def test_fold_matches_unfolded_NM(self):
        """SIGNED per-element parity of (N, M). This pins the rotation sign."""
        N_u, M_u = self._NM(0)
        N_f, M_f = self._NM(1)
        N_u, M_u = np.asarray(N_u), np.asarray(M_u)
        N_f, M_f = np.asarray(N_f), np.asarray(M_f)

        # Guard against a degenerate batch: if the filters that the fold
        # RECONSTRUCTS (2 and 3) carried no weight, the test would pass on a
        # wrong sign too.
        self.assertGreater(
            np.max(np.abs(N_u[:, 2:])), 1e-3 * np.max(np.abs(N_u)),
            "batch does not excite the reconstructed filters -- gate is blind")

        np.testing.assert_allclose(
            N_f, N_u, rtol=RTOL, atol=_dyn_atol(N_u),
            err_msg="folded N != unfolded N (check the rotation sign)")
        np.testing.assert_allclose(
            M_f, M_u, rtol=RTOL, atol=_dyn_atol(M_u),
            err_msg="folded M != unfolded M (check the rotation sign)")

    def test_fold_matches_unfolded_extrinsics(self):
        """End-to-end: ``fstat_maximized_extrinsics``, F above all."""
        A_u, p_u, i_u, s_u, F_u = fstat_maximized_extrinsics(*self._NM(0))
        A_f, p_f, i_f, s_f, F_f = fstat_maximized_extrinsics(*self._NM(1))

        np.testing.assert_allclose(
            F_f, F_u, rtol=RTOL, atol=_dyn_atol(F_u), err_msg="F moved")
        np.testing.assert_allclose(
            A_f, A_u, rtol=1e-7, atol=_dyn_atol(A_u), err_msg="A_max moved")
        # sigma / ln_snr are pure functions of F -- pin the one that matters.
        np.testing.assert_allclose(
            1.0 / np.sqrt(np.maximum(2.0 * np.asarray(F_f), 1.0)),
            1.0 / np.sqrt(np.maximum(2.0 * np.asarray(F_u), 1.0)),
            rtol=1e-9, err_msg="sigma moved")
        _assert_angles_close(p_f, p_u, msg="phi0_max moved")
        _assert_angles_close(i_f, i_u, msg="iota_max moved")
        _assert_angles_close(s_f, s_u, msg="psi_max moved")

    # ---- why the gate has to be on (N, M) and not on F ----------------

    def test_sign_error_is_invisible_in_F(self):
        """A conjugated rotation sign leaves F EXACTLY invariant.

        Pure host arithmetic on the unfolded (N, M): conjugating alpha_2 and
        alpha_3 flips filters 2 and 3, i.e. N -> D N and M -> D M D with
        D = diag(1, 1, -1, -1). Then N^T M^-1 N is unchanged, so no
        F / SNR / ln_snr check can ever catch the sign -- only the signed
        (N, M) compare above can. This test exists so nobody "simplifies"
        the parity gate down to F.
        """
        N_u, M_u = self._NM(0)
        N_u, M_u = np.asarray(N_u), np.asarray(M_u)

        D = np.array([1.0, 1.0, -1.0, -1.0])
        N_bad = N_u * D
        M_bad = np.empty_like(M_u)
        for k, (i, j) in enumerate(_TRIU):
            M_bad[:, k] = M_u[:, k] * D[i] * D[j]

        F_good = fstat_maximized_extrinsics(N_u, M_u)[4]
        F_bad = fstat_maximized_extrinsics(N_bad, M_bad)[4]
        np.testing.assert_allclose(
            F_bad, F_good, rtol=1e-12,
            err_msg="premise broken: the sign flip DOES move F")
        # ... while (N, M) themselves move by O(1) -- which is exactly the
        # signal test_fold_matches_unfolded_NM is sensitive to.
        self.assertGreater(
            np.max(np.abs(N_bad - N_u)), 1e-3 * np.max(np.abs(N_u)))

    def test_alpha_constants_are_exact_units(self):
        """The rotation constants are exact in floating point.

        alpha = e^{-i phi0} over {0, pi, 3pi/2, pi/2} is {1, -1, +i, -i}. The
        kernel hardcodes these as sign flips / real-imag swaps rather than
        cos/sin so the rotation contributes ZERO rounding of its own -- note
        cos(3*pi/2) evaluates to -1.8e-16, not 0.
        """
        phi0 = np.array([0.0, np.pi, 3.0 * np.pi / 2.0, np.pi / 2.0])
        np.testing.assert_allclose(np.exp(-1j * phi0), ALPHA, atol=1e-15)
        self.assertTrue(np.all(np.abs(ALPHA) == 1.0))
        self.assertNotEqual(np.cos(3.0 * np.pi / 2.0), 0.0)

    # ---- orbit spline cache (2026-09-01 port) -------------------------

    def _NM_with_orbit_cache(self, n_cp, armed=True):
        """(N, M) with the comp's ``N_cp_orbit`` / arm switch temporarily set.

        Both are read at CALL time by ``_fstat_orbit_cache_kernel_args``, so a
        set/restore around one call is the whole mechanism -- no rebuild.
        """
        saved = (self.chunked.N_cp_orbit, self.chunked.fstat_orbit_cache)
        try:
            self.chunked.N_cp_orbit = int(n_cp)
            self.chunked.fstat_orbit_cache = 1 if armed else 0
            return self._NM(1)   # folded = the production path
        finally:
            self.chunked.N_cp_orbit, self.chunked.fstat_orbit_cache = saved

    def test_orbit_cache_matches_direct_NM(self):
        """Cached orbit evals reproduce direct lookups on (N, M).

        Unlike the fold this is an APPROXIMATION (per-chunk cubic spline of
        the orbit tables), but get_ll's contract is that at N_cp_orbit >= 32
        the LTT/position residuals sit below float64 precision over typical
        chunk lengths -- so the SAME fold-parity tolerance must hold. Signed
        per-element compare, same reasoning as the fold gate.
        """
        N_d, M_d = self._NM_with_orbit_cache(0)
        N_c, M_c = self._NM_with_orbit_cache(32)
        # Guard against a silently-ignored N_cp_orbit: the spline is an
        # approximation, so SOMEWHERE in the batch it must differ from the
        # direct path in the last bits. Bitwise equality everywhere means the
        # kernel never engaged the cache and this test is blind.
        self.assertTrue(
            np.any(np.asarray(N_c) != np.asarray(N_d))
            or np.any(np.asarray(M_c) != np.asarray(M_d)),
            "cached == direct BITWISE everywhere -- orbit cache never engaged")
        np.testing.assert_allclose(
            np.asarray(N_c), np.asarray(N_d), rtol=RTOL,
            atol=_dyn_atol(np.asarray(N_d)),
            err_msg="orbit-cached N != direct N")
        np.testing.assert_allclose(
            np.asarray(M_c), np.asarray(M_d), rtol=RTOL,
            atol=_dyn_atol(np.asarray(M_d)),
            err_msg="orbit-cached M != direct M")

    def test_orbit_cache_disarm_is_bitwise_off(self):
        """``fstat_orbit_cache = 0`` forces the kernel's direct-lookup path.

        With the arm switch off the python layer passes N_cp_orbit = 0
        regardless of the comp's setting, so the result must be BITWISE equal
        to a plain N_cp_orbit = 0 call -- same code path, not merely close.
        This is the ``GB_FSTAT_ORBIT_CACHE=0`` safety valve.
        """
        N_0, M_0 = self._NM_with_orbit_cache(0)
        N_x, M_x = self._NM_with_orbit_cache(32, armed=False)
        np.testing.assert_array_equal(np.asarray(N_x), np.asarray(N_0))
        np.testing.assert_array_equal(np.asarray(M_x), np.asarray(M_0))


if __name__ == "__main__":
    unittest.main()
