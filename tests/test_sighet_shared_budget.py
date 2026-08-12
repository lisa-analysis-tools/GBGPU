"""CPU-only tests for the sig-het shared-memory budget mirror + clamps.

Covers the Python side of the SIGHET_NT_LAYER shared-memory work
(2026-08-12; the ``GPUassert: invalid argument gb_tdi_on_the_fly.cu:6747``
gate death):

* ``_sighet_fstat_shared_bytes`` -- pinned to reference values computed by
  compiling the C++ ``gb_sighet_fstat_shared_bytes`` /
  ``gb_sighet_v5_region_sizes`` (copied verbatim from
  ``cutils/gb_tdi_on_the_fly.cu``) as host C++. If the kernel carve changes,
  BOTH the C++ and this mirror must change together and these constants must
  be regenerated.
* ``_resolve_nt_layer`` -- the SIGHET_NT_LAYER=-1 auto policy (constant
  ~35 h sparse spacing, divisor snap, device clamp) and the explicit-value
  snap behavior.
* ``_check_fstat_shared_for_device`` -- the setup-time device gate that
  replaces the mid-run GPUassert with an actionable ValueError naming the
  max supported SIGHET_NT_LAYER.

No GPU, no data files, no compiled sig-het setup -- pure knob logic.
"""

import os
import unittest
from unittest import mock

import numpy as np

from gbgpu.gbsignalhetcomputations import (
    GBSignalHetComputations,
    _resolve_nt_layer,
    _sighet_fstat_shared_bytes,
    _SIGHET_STATIC_SHARED_RESERVE,
)

# (n_nodes, n_knots, nch, m_half, N_sparse_t, band_len, n_stages) -> bytes,
# computed from the C++ formula compiled as host C++ (2026-08-12).
CXX_REFERENCE = {
    (64, 128, 3, 2, 58, 32, 2): 40128,     # legacy 3-mo default (nt 60 grid)
    (64, 128, 3, 2, 64, 32, 2): 40704,
    (64, 128, 3, 2, 135, 32, 2): 47760,    # the failed gate's config, mode 0
    (64, 128, 3, 2, 135, 32, 4): 60720,    # ... mode 1
    (64, 128, 3, 2, 525, 32, 2): 85920,    # 23-mo prescription, mode 0
    (64, 128, 3, 2, 525, 32, 4): 136320,   # ... mode 1
    (64, 128, 3, 2, 525, 0, 2): 108960,    # non-banded (v4_band=0)
    (64, 128, 3, 2, 1024, 32, 4): 232960,  # over every device: must be huge
}

A100_OPTIN = 163840
H100_OPTIN = 232448


class TestFstatSharedMirror(unittest.TestCase):
    def test_pinned_cxx_values(self):
        for args, expected in CXX_REFERENCE.items():
            self.assertEqual(_sighet_fstat_shared_bytes(*args), expected,
                             msg=f"mirror drifted from C++ at {args}")

    def test_23mo_prescription_fits_a100(self):
        """SIGHET_NT_LAYER=525 (N_sparse_t=525) must fit an A100 at both
        fstat modes with the static reserve -- the capacity requirement of
        the 23-month run."""
        budget = A100_OPTIN - _SIGHET_STATIC_SHARED_RESERVE
        for stages in (2, 4):
            self.assertLessEqual(
                _sighet_fstat_shared_bytes(64, 128, 3, 2, 525, 32, stages),
                budget)

    def test_monotonic_in_n_sparse_t(self):
        prev = 0
        for n in range(8, 2049, 8):
            b = _sighet_fstat_shared_bytes(64, 128, 3, 2, n, 32, 2)
            self.assertGreaterEqual(b, prev)
            prev = b


class TestResolveNtLayer(unittest.TestCase):
    def test_explicit_divisor_untouched(self):
        self.assertEqual(
            _resolve_nt_layer(525, 1440, 16800, 16752, 2.5, v3_n_nodes=64,
                              v4_knots=128, v4_band=16, m_half=2,
                              device_shared_limit=A100_OPTIN),
            525)

    def test_explicit_nondivisor_snaps(self):
        # 64 does not divide 2160; nearest divisor is 60 (pre-existing
        # behavior, must not change).
        self.assertEqual(
            _resolve_nt_layer(64, 1440, 2160, 2112, 2.5, v3_n_nodes=64,
                              v4_knots=128, v4_band=16, m_half=2,
                              device_shared_limit=None),
            60)

    def test_explicit_never_device_clamped(self):
        # An explicit prescription on a tiny device is NOT silently
        # coarsened (the setup gate raises later instead).
        self.assertEqual(
            _resolve_nt_layer(525, 1440, 16800, 16752, 2.5, v3_n_nodes=64,
                              v4_knots=128, v4_band=16, m_half=2,
                              device_shared_limit=60000),
            525)

    def test_auto_targets_35h_and_divides(self):
        # 23-mo grid: layer_dt = 1440*2.5 = 3600 s -> stride 35 -> 480.
        got = _resolve_nt_layer(-1, 1440, 16800, 16752, 2.5, v3_n_nodes=64,
                                v4_knots=128, v4_band=16, m_half=2,
                                device_shared_limit=None)
        self.assertEqual(got, 480)
        self.assertEqual(16800 % got, 0)
        # 3-mo grid: Nt=2160 -> 61.7 -> snap to divisor 60 (the legacy
        # default's landing spot).
        got3 = _resolve_nt_layer(-1, 1440, 2160, 2112, 2.5, v3_n_nodes=64,
                                 v4_knots=128, v4_band=16, m_half=2,
                                 device_shared_limit=None)
        self.assertEqual(got3, 60)

    def test_auto_spacing_env(self):
        with mock.patch.dict(os.environ, {"SIGHET_NT_AUTO_SPACING_H": "32"}):
            got = _resolve_nt_layer(-1, 1440, 16800, 16752, 2.5,
                                    v3_n_nodes=64, v4_knots=128, v4_band=16,
                                    m_half=2, device_shared_limit=None)
        self.assertEqual(got, 525)   # 32 h stride = the 23-mo prescription

    def test_auto_clamps_to_device(self):
        got = _resolve_nt_layer(-1, 1440, 16800, 16752, 2.5, v3_n_nodes=64,
                                v4_knots=128, v4_band=16, m_half=2,
                                device_shared_limit=60000)
        self.assertEqual(16800 % got, 0)
        self.assertLess(got, 480)
        # and the clamped value's budget actually fits
        nsp = 16752 // (16800 // got)
        self.assertLessEqual(
            _sighet_fstat_shared_bytes(64, 128, 3, 2, nsp, 32, 2),
            60000 - _SIGHET_STATIC_SHARED_RESERVE)

    def test_auto_fits_a100_at_23mo(self):
        got = _resolve_nt_layer(-1, 1440, 16800, 16752, 2.5, v3_n_nodes=64,
                                v4_knots=128, v4_band=16, m_half=2,
                                device_shared_limit=A100_OPTIN)
        self.assertEqual(got, 480)   # no clamp needed: 480 fits an A100

    def test_invalid_raises(self):
        with self.assertRaises(ValueError):
            _resolve_nt_layer(0, 1440, 2160, 2112, 2.5, v3_n_nodes=64,
                              v4_knots=128, v4_band=16, m_half=2,
                              device_shared_limit=None)


def _bare_comp(g, limit):
    """A GBSignalHetComputations shell with a fixed device limit -- no
    backend, no data; exactly what the clamp methods touch."""
    comp = GBSignalHetComputations.__new__(GBSignalHetComputations)
    comp._g = g
    comp._device_shared_limit = lambda: limit
    return comp


class TestSetupTimeGate(unittest.TestCase):
    G_23MO = dict(Nt=16800, Nt_active=16752, N_sparse_t=523, nt_layer=525,
                  m_half=2, v3_n_nodes=64, v4_knots=128, v4_band=16,
                  Tobs=6.048e7)

    def test_cpu_noop(self):
        comp = _bare_comp(dict(self.G_23MO), None)
        comp._check_fstat_shared_for_device(0)   # must not raise

    def test_fits_a100_both_modes(self):
        comp = _bare_comp(dict(self.G_23MO), A100_OPTIN)
        comp._check_fstat_shared_for_device(0)
        comp._check_fstat_shared_for_device(1)

    def test_over_budget_raises_with_max_layer(self):
        comp = _bare_comp(dict(self.G_23MO), 60000)
        with self.assertRaises(ValueError) as ctx:
            comp._check_fstat_shared_for_device(0)
        msg = str(ctx.exception)
        self.assertIn("SIGHET_NT_LAYER=", msg)
        self.assertIn("60000", msg)
        # the recommended layer must be a divisor of Nt whose N_sparse_t
        # fits the stated budget
        rec = int(msg.split("SIGHET_NT_LAYER=")[1].split()[0].rstrip(","))
        self.assertEqual(16800 % rec, 0)
        nsp = 16752 // (16800 // rec)
        self.assertLessEqual(
            _sighet_fstat_shared_bytes(64, 128, 3, 2, nsp, 32, 2),
            60000 - _SIGHET_STATIC_SHARED_RESERVE)

    def test_mode1_tighter_than_mode0(self):
        # a budget that passes mode 0 but fails mode 1
        need0, _ = _bare_comp(dict(self.G_23MO), None)._fstat_kernel_budget(0)
        need1, _ = _bare_comp(dict(self.G_23MO), None)._fstat_kernel_budget(1)
        self.assertGreater(need1, need0)
        limit = need0 + _SIGHET_STATIC_SHARED_RESERVE + 16
        comp = _bare_comp(dict(self.G_23MO), limit)
        comp._check_fstat_shared_for_device(0)
        with self.assertRaises(ValueError):
            comp._check_fstat_shared_for_device(1)


if __name__ == "__main__":
    unittest.main()
