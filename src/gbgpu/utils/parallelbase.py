"""GBGPU dispatch base -- backward-compatibility shim.

The canonical implementation of :class:`GBGPUParallelModule` lives in
:mod:`gbgpu.parallelbase` (Phase 3L.7k). This module re-exports it under
the same name so existing ``from .utils.parallelbase import
GBGPUParallelModule`` imports (e.g. ``gbgpu.gbgpu.GBGPUBase``) keep
working unchanged. New code should import from ``gbgpu.parallelbase``
directly. Mirrors the LAT precedent at
``lisatools.response.parallelbase`` (shim) ->
``lisatools.utils.parallelbase.LISAToolsParallelModule`` (canonical).
"""

from ..parallelbase import GBGPUParallelModule

__all__ = ["GBGPUParallelModule"]
