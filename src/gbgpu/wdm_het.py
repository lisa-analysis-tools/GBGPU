"""Backwards-compat re-export shim.

Phase 3L.7p-followup (2026-06-06, "SOBBH-to-BBHx" architectural fix):
the chunked-heterodyne utility functions originally shipped here moved
to ``lisatools.wdm_het`` so the SOBBH side (which now lives in BBHx)
can share them without introducing a BBHx -> GBGPU import dependency.

GB-side imports continue to work because everything is re-exported
from the new home below. New code should import directly from
``lisatools.wdm_het``.
"""
from __future__ import annotations

from lisatools.wdm_het import *  # noqa: F401,F403
from lisatools.wdm_het import (  # explicit list for `from gbgpu.wdm_het import X`
    USE_RECOMMENDED_TUKEY,
    RECOMMENDED_TUKEY_ALPHA_TD,
    RECOMMENDED_TUKEY_ALPHA_HET_WIDE,
    RECOMMENDED_TUKEY_ALPHA_HET_NARROW,
    recommended_tukey_alpha,
    resolve_tukey_alpha,
    compute_chunk_geometry,
    compute_wdm_window,
    compute_layer_groups,
    compute_swap_layer_groups,
)
