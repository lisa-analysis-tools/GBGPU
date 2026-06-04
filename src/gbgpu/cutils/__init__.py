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
            import gbgpu_backend_cpu.cgbgpu  # Phase GBGPU.pybind.bulk: sole GBGPU backend module
        except (ModuleNotFoundError, ImportError) as e:
            raise BackendUnavailableException(
                "'cpu' backend could not be imported."
            ) from e

        numpy = GBGPUCpuBackend.check_numpy()

        # Single-instance GBGPUComputationWrap holds every migrated wrapper
        # (utils + sharedmem methods both land as members on the same wrap).
        # `sharedmem` exposes the same instance under its prior namespace
        # name so user code (`self.backend.sharedmem.SharedMemoryWaveComp_wrap(...)`)
        # keeps working unchanged.
        _cgbgpu = gbgpu_backend_cpu.cgbgpu.GBGPUComputationWrapCPU()
        return GBGPUBackendMethods(
            get_ll=_cgbgpu.get_ll,
            fill_global=_cgbgpu.fill_global,
            sharedmem=_cgbgpu,
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
            import gbgpu_backend_cuda11x.cgbgpu  # Phase GBGPU.pybind.bulk
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

        _cgbgpu = gbgpu_backend_cuda11x.cgbgpu.GBGPUComputationWrapGPU()
        return GBGPUBackendMethods(
            get_ll=_cgbgpu.get_ll,
            fill_global=_cgbgpu.fill_global,
            sharedmem=_cgbgpu,
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
            import gbgpu_backend_cuda12x.cgbgpu  # Phase GBGPU.pybind.bulk
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

        _cgbgpu = gbgpu_backend_cuda12x.cgbgpu.GBGPUComputationWrapGPU()
        return GBGPUBackendMethods(
            get_ll=_cgbgpu.get_ll,
            fill_global=_cgbgpu.fill_global,
            sharedmem=_cgbgpu,
            xp=cupy,
        )


"""List of existing backends, per default order of preference."""
# TODO: __all__ ?


