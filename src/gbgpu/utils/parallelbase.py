from gpubackendtools import ParallelModuleBase


class GBGPUParallelModule(ParallelModuleBase):
    """ParallelModuleBase variant that resolves ``force_backend`` strings
    through the ``gbgpu`` backend family (``gbgpu_cpu`` / ``gbgpu_cuda12x``
    / ``gbgpu_jax`` / ...). Mirrors the LAT pattern in
    ``lisatools.response.parallelbase.FastLISAResponseParallelModule``.
    """

    def __init__(self, force_backend=None):
        force_backend_in = (
            ("gbgpu", force_backend) if isinstance(force_backend, str) else force_backend
        )
        super().__init__(force_backend_in)
