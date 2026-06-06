from gpubackendtools import ParallelModuleBase


class GBGPUParallelModule(ParallelModuleBase):
    """ParallelModuleBase variant that resolves ``force_backend`` strings
    through the ``gbgpu`` backend family (``gbgpu_cpu`` / ``gbgpu_cuda12x``
    / ``gbgpu_jax`` / ...). Mirrors the LAT pattern in
    ``lisatools.response.parallelbase.FastLISAResponseParallelModule``.

    Subclasses can override ``_BACKEND_PREFIX`` to re-route through a
    different backend family without rewriting ``__init__``. The chunked-
    heterodyne ``SOBBHWDMComputations`` does this to route through
    ``bbhx_<flavor>`` (which carries ``SOBBHComputationGroupWrap``)
    instead of ``gbgpu_<flavor>``.
    """

    _BACKEND_PREFIX = "gbgpu"

    def __init__(self, force_backend=None):
        force_backend_in = (
            (self._BACKEND_PREFIX, force_backend)
            if isinstance(force_backend, str)
            else force_backend
        )
        super().__init__(force_backend_in)
