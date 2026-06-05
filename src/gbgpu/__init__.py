"""GBGPU."""

# ruff: noqa: E402
try:
    from gbgpu._version import (  # pylint: disable=E0401,E0611
        __version__,
        __version_tuple__,
    )

except ModuleNotFoundError:
    from importlib.metadata import PackageNotFoundError, version  # pragma: no cover

    try:
        __version__ = version(__name__)
        __version_tuple__ = tuple(__version__.split("."))
    except PackageNotFoundError:  # pragma: no cover
        __version__ = "unknown"
        __version_tuple__ = (0, 0, 0, "unknown")
    finally:
        del version, PackageNotFoundError

_is_editable: bool
try:
    from . import _editable

    _is_editable = True
    del _editable
except (ModuleNotFoundError, ImportError):
    _is_editable = False


def get_include() -> str:
    """Absolute path to GBGPU's public C++/CUDA header directory.

    Downstream sprint packages and lisa-on-gpu (during the Phase 3L.7
    transition) add this to their compiler include path so that

        #include "gb_tdi_on_the_fly.hh"    // GBTDIonTheFly + GBComputationGroup

    resolves against the installed wheel.

    Pair with ``gpubackendtools.get_include()`` and
    ``lisatools.get_include()`` -- a downstream typically needs all three:
    GBT for ``gbt_global.h`` + ``cuda_complex.hpp`` + ``InterpolateDevice.hh``,
    LAT for ``Detector.hpp`` + ``lat_chunked_het_kernels.hh`` + base
    classes, GBGPU for the GB source-class-specific machinery
    (``GBTDIonTheFly``, ``GBComputationGroup``, ``gb_run_wave_tdi_wrap``).
    """
    import os.path
    return os.path.join(os.path.dirname(__file__), "cutils")


def get_cmake_module_path() -> str:
    """Absolute path to the directory containing ``GBGPUConfig.cmake``.

    For downstreams that prefer ``find_package(GBGPU CONFIG)`` over the
    Python shell-out form of :func:`get_include`. Forward-compatible:
    ``GBGPUConfig.cmake`` ships at the cutils directory once added;
    consumers can begin using this path before that file exists (CMake
    just won't find the package and they fall through to the shell-out
    form).

    Returns the same directory as :func:`get_include`.
    """
    import os.path
    return os.path.join(os.path.dirname(__file__), "cutils")


from . import cutils, utils

from gpubackendtools import Globals
from .cutils import (
    GBGPUCpuBackend,
    GBGPUCuda11xBackend,
    GBGPUCuda12xBackend,
    GBGPUCuda13xBackend,
)


# Phase 3L.7k (2026-06-04): GBGPU's backends now compose LAT's
# LISAToolsBackend surface with GB-specific Wraps. The previous
# fastlisaresponse_<flavor> backend family (lisa-on-gpu) is being
# retired and consumers can use these directly via
# ``gbgpu.get_backend(...)``.
add_backends = {
    "gbgpu_cpu": GBGPUCpuBackend,
    "gbgpu_cuda11x": GBGPUCuda11xBackend,
    "gbgpu_cuda12x": GBGPUCuda12xBackend,
    "gbgpu_cuda13x": GBGPUCuda13xBackend,
}

Globals().backends_manager.add_backends(add_backends)


from gpubackendtools import get_backend as _get_backend
from gpubackendtools import has_backend as _has_backend
from gpubackendtools import get_first_backend as _get_first_backend
from gpubackendtools.gpubackendtools import Backend


def get_backend(backend: str) -> Backend:
    __doc__ = _get_backend.__doc__
    if "gbgpu_" not in backend:
        return _get_backend("gbgpu_" + backend)
    else:
        return _get_backend(backend)


def has_backend(backend: str) -> Backend:
    __doc__ = _has_backend.__doc__
    if "gbgpu_" not in backend:
        return _has_backend("gbgpu_" + backend)
    else:
        return _has_backend(backend)

        
def get_first_backend(backend: str) -> Backend:
    __doc__ = _get_first_backend.__doc__
    if "gbgpu_" not in backend:
        return _get_first_backend("gbgpu_" + backend)
    else:
        return _get_first_backend(backend)


__all__ = [
    "__version__",
    "__version_tuple__",
    "_is_editable",
    # "amplitude",
    # "cutils",
    # "files",
    # "summation",
    # "trajectory",
    # "utils",
    # "waveform",
    "get_logger",
    "get_config",
    "get_config_setter",
    "get_backend",
    "get_file_manager",
    "has_backend",
]
