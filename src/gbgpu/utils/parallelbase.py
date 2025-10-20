from typing import Optional, Sequence, TypeVar, Union
import types
from .. import get_first_backend

from gpubackendtools import ParallelModuleBase


class GBGPUParallelModule(ParallelModuleBase):
    def __init__(self, force_backend=None):
        if force_backend is None:
            breakpoint()
            force_backend = get_first_backend(self.CPU_RECOMMENDED)

        force_backend_in = ('gbgpu', force_backend) if isinstance(force_backend, str) else force_backend
        super().__init__(force_backend_in)
