"""Rank-based intra-shard indexing (`GBGPU._intra_split_index`).

The legacy ``% num_per_gpu`` block trick assumed equal contiguous shards;
``np.array_split`` with ``n % ngpus != 0`` makes shards after the first
misindex. The rank-based mapping is exact for uneven contiguous splits and
for arbitrary (e.g. striped) ``data_splits`` assignments, and reduces to
the identity for the single-buffer default.
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import numpy as np

from gbgpu.gbgpu import GBGPU


def _call(index_arr, data_splits, gpu, num_per_gpu=int(1e9)):
    fake_self = SimpleNamespace(xp=np)
    return GBGPU._intra_split_index(
        fake_self, np.asarray(index_arr), data_splits, gpu, num_per_gpu
    )


class IntraSplitIndexTest(unittest.TestCase):
    def test_uneven_contiguous_three_shards(self):
        # array_split(7, 3) -> blocks [0,1,2], [3,4], [5,6]; offsets 0, 3, 5.
        data_splits = np.array([0, 0, 0, 1, 1, 2, 2])
        # shard 2 rows 5, 6 -> intra 0, 1 (the legacy % 3 gave 2, 0).
        np.testing.assert_array_equal(_call([5, 6], data_splits, 2), [0, 1])
        np.testing.assert_array_equal(_call([3, 4], data_splits, 1), [0, 1])
        np.testing.assert_array_equal(_call([2, 0], data_splits, 0), [2, 0])

    def test_legacy_modulo_would_misindex(self):
        data_splits = np.array([0, 0, 0, 1, 1, 2, 2])
        num_per_gpu = 3  # the old callers' len(gpu_splits[0])
        legacy = np.asarray([5, 6]) % num_per_gpu
        exact = _call([5, 6], data_splits, 2)
        self.assertFalse(np.array_equal(legacy, exact))

    def test_striped_assignment(self):
        # striped rows: shard = row % 2 -> shard 1 rows 1,3,5 -> intra 0,1,2
        data_splits = np.arange(6) % 2
        np.testing.assert_array_equal(
            _call([5, 1, 3], data_splits, 1), [2, 0, 1])
        np.testing.assert_array_equal(
            _call([0, 4], data_splits, 0), [0, 2])

    def test_single_buffer_default_is_identity(self):
        # default: every row on one gpu -> rank == global index
        data_splits = np.zeros(5, dtype=int)
        np.testing.assert_array_equal(
            _call([4, 0, 2], data_splits, 0), [4, 0, 2])

    def test_none_data_splits_falls_back_to_modulo(self):
        out = _call([4, 7], None, 0, num_per_gpu=3)
        np.testing.assert_array_equal(out, [1, 1])

    def test_int32_dtype(self):
        data_splits = np.array([0, 0, 1, 1])
        out = _call([2, 3], data_splits, 1)
        self.assertEqual(out.dtype, np.int32)


if __name__ == "__main__":
    unittest.main()
