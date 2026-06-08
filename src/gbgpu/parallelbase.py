"""GBGPU dispatch base.

Phase 3L.7k (2026-06-04): GBGPU-specific subclass of
``ParallelModuleBase`` that prefixes ``gbgpu_`` onto bare backend tags
(``cpu`` / ``cuda13x`` / ``jax``) so consumers like
``gbgpu.gbcomps.GBWDMComputations`` automatically resolve to the
GBGPU-composed backend (LAT response Wraps + GB-specific Wraps in one
object). Parallels :class:`lisatools.response.parallelbase.FastLISAResponseParallelModule`
which prefixes ``lisatools_``.
"""

from gpubackendtools import ParallelModuleBase


class GBGPUParallelModule(ParallelModuleBase):
    """Base class for GBGPU parallel-dispatch modules.

    Subclasses (``GBWDMComputations`` / ``GBFDComputations`` / etc.)
    inherit from this and pass ``force_backend="cpu"`` style strings.
    The string is automatically prefixed with ``gbgpu_`` so the
    resolved backend is ``gbgpu_cpu`` (which carries both LAT and GB
    native symbols after Phase 3L.7k).
    """

    _BACKEND_PREFIX = "gbgpu"

    def __init__(self, force_backend=None):
        if isinstance(force_backend, str) and not force_backend.startswith(
            self._BACKEND_PREFIX + "_"
        ):
            force_backend = (self._BACKEND_PREFIX, force_backend)
        super().__init__(force_backend)

    @staticmethod
    def GPU_RECOMMENDED_WITH_JAX() -> list[str]:
        """Same as GPU_RECOMMENDED() but with the JAX backend appended."""
        return ["cuda13x", "cuda12x", "cuda11x", "cpu", "jax"]
