from __future__ import annotations
import dataclasses
import enum
import types
import typing
import abc
from typing import Optional, Sequence, TypeVar, Union
from ..utils.exceptions import *

from gpubackendtools.gpubackendtools import BackendMethods, CpuBackend, Cuda11xBackend, Cuda12xBackend
from gpubackendtools.exceptions import *

@dataclasses.dataclass
class GBGPUBackendMethods(BackendMethods):
    get_ll: typing.Callable[(...), None]
    fill_global: typing.Callable[(...), None]
    sharedmem: object

class GBGPUBackend:
    get_ll: typing.Callable[(...), None]
    fill_global: typing.Callable[(...), None]
    sharedmem: object

    def __init__(self, gbgpu_backend_methods):

        # set direct gbgpu methods
        # pass rest to general backend
        assert isinstance(gbgpu_backend_methods, GBGPUBackendMethods)

        self.get_ll = gbgpu_backend_methods.get_ll
        self.fill_global = gbgpu_backend_methods.fill_global
        self.sharedmem = gbgpu_backend_methods.sharedmem

class GBGPUCpuBackend(CpuBackend, GBGPUBackend):
    """Implementation of the CPU backend"""
    
    _backend_name = "gbgpu_backend_cpu"
    _name = "gbgpu_cpu"
    def __init__(self, *args, **kwargs):
        CpuBackend.__init__(self, *args, **kwargs)
        GBGPUBackend.__init__(self, self.cpu_methods_loader())

    @staticmethod
    def cpu_methods_loader() -> GBGPUBackendMethods:
        try:
            import gbgpu_backend_cpu.utils
            import gbgpu_backend_cpu.sharedmem
            
        except (ModuleNotFoundError, ImportError) as e:
            raise BackendUnavailableException(
                "'cpu' backend could not be imported."
            ) from e

        numpy = GBGPUCpuBackend.check_numpy()

        return GBGPUBackendMethods(
            get_ll=gbgpu_backend_cpu.utils.get_ll,
            fill_global=gbgpu_backend_cpu.utils.fill_global,
            sharedmem=gbgpu_backend_cpu.sharedmem,
            xp=numpy,
        )


class GBGPUCuda11xBackend(Cuda11xBackend, GBGPUBackend):

    """Implementation of CUDA 11.x backend"""
    _backend_name : str = "gbgpu_backend_cuda11x"
    _name = "gbgpu_cuda11x"

    def __init__(self, *args, **kwargs):
        Cuda11xBackend.__init__(self, *args, **kwargs)
        GBGPUBackend.__init__(self, self.cuda11x_module_loader())
        
    @staticmethod
    def cuda11x_module_loader():
        try:
            import gbgpu_backend_cuda11x.utils
            import gbgpu_backend_cuda11x.sharedmem

        except (ModuleNotFoundError, ImportError) as e:
            raise BackendUnavailableException(
                "'cuda11x' backend could not be imported."
            ) from e

        try:
            import cupy
        except (ModuleNotFoundError, ImportError) as e:
            raise MissingDependencies(
                "'cuda11x' backend requires cupy", pip_deps=["cupy-cuda11x"]
            ) from e

        return GBGPUBackendMethods(
            get_ll=gbgpu_backend_cuda11x.utils.get_ll,
            fill_global=gbgpu_backend_cuda11x.utils.fill_global,
            sharedmem=gbgpu_backend_cuda11x.sharedmem,
            xp=cupy,
        )

class GBGPUCuda12xBackend(Cuda12xBackend, GBGPUBackend):
    """Implementation of CUDA 12.x backend"""
    _backend_name : str = "gbgpu_backend_cuda12x"
    _name = "gbgpu_cuda12x"
    
    def __init__(self, *args, **kwargs):
        Cuda12xBackend.__init__(self, *args, **kwargs)
        GBGPUBackend.__init__(self, self.cuda12x_module_loader())
        
    @staticmethod
    def cuda12x_module_loader():
        try:
            import gbgpu_backend_cuda12x.utils
            import gbgpu_backend_cuda12x.sharedmem

        except (ModuleNotFoundError, ImportError) as e:
            raise BackendUnavailableException(
                "'cuda12x' backend could not be imported."
            ) from e

        try:
            import cupy
        except (ModuleNotFoundError, ImportError) as e:
            raise MissingDependencies(
                "'cuda12x' backend requires cupy", pip_deps=["cupy-cuda12x"]
            ) from e

        return GBGPUBackendMethods(
            get_ll=gbgpu_backend_cuda12x.utils.get_ll,
            fill_global=gbgpu_backend_cuda12x.utils.fill_global,
            sharedmem=gbgpu_backend_cuda12x.sharedmem,
            xp=cupy,
        )


"""List of existing backends, per default order of preference."""
# TODO: __all__ ?


