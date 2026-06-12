"""GBGPU backend definitions.

Phase 3L.7k (2026-06-04): GBGPU backends now compose the LAT
``LISAToolsBackend`` surface (Orbits / WDM / FD / Spline / response
Wraps) with GBGPU's own GB-specific Wraps
(`GBTDIonTheFlyWrap`, `GBComputationGroupWrap`, `GBGPUComputationWrap`).
Each module loader imports both ``lisatools_backend_<flavor>.pycppdetector``
and ``gbgpu_backend_<flavor>.cgbgpu`` so a single
``gbgpu.get_backend(name)`` call returns a backend object carrying
every native symbol the GB pipeline needs.

This is the GBGPU-side counterpart to the post-Phase-3L.7k LAT backend
extension; the previously-separate ``fastlisaresponse_<flavor>``
backend family (defined in lisa-on-gpu) is being retired in favor of
this composition.
"""

from __future__ import annotations

import dataclasses
import typing

from gpubackendtools.exceptions import *
from gpubackendtools.gpubackendtools import (
    CpuBackend,
    Cuda11xBackend,
    Cuda12xBackend,
    Cuda13xBackend,
)
from lisatools.cutils import (
    LISAToolsBackend,
    LISAToolsBackendMethods,
)

from ..utils.exceptions import *


@dataclasses.dataclass
class GBGPUBackendMethods(LISAToolsBackendMethods):
    """Container of native symbols exposed by a GBGPU backend.

    Extends :class:`lisatools.cutils.LISAToolsBackendMethods` with the
    GB-specific Wraps that the chunked-het + FD pipelines need.

    Attributes:
        GBTDIonTheFlyWrap: ``GBTDIonTheFlyWrap{CPU,GPU}`` -- the GB
            source-class wrap (carved to GBGPU at Phase 3L.7g).
        GBComputationGroupWrap: ``GBComputationGroupWrap{CPU,GPU}`` --
            the GB chunked-het + FD + signal-het computation group
            (carved to GBGPU at Phase 3L.7g).
        get_ll / fill_global / sharedmem: Pass-through accessors onto the
            ``GBGPUComputationWrap`` instance for the pre-existing
            gbgpu_utils + SharedMemoryGBGPU surface. The single wrap
            instance is exposed as ``sharedmem`` so existing call sites
            (``self.backend.sharedmem.SharedMemoryWaveComp_wrap(...)``)
            keep working.
    """

    GBTDIonTheFlyWrap: object
    GBComputationGroupWrap: object
    get_ll: typing.Callable[(...), None]
    fill_global: typing.Callable[(...), None]
    sharedmem: object


class GBGPUBackend(LISAToolsBackend):
    """Mixin attaching GBGPU-specific symbols on top of a LAT backend.

    Inherits :class:`lisatools.cutils.LISAToolsBackend` so concrete
    GBGPU backends expose every LAT native symbol (OrbitsWrap,
    TDIConfigWrap, WDM/FD/Spline wraps, TDITypeDict) plus the
    GB-specific Wraps. Consumer code can do ``self.backend.OrbitsWrap``
    and ``self.backend.GBTDIonTheFlyWrap`` on the same object.
    """

    GBTDIonTheFlyWrap: object
    GBComputationGroupWrap: object
    get_ll: typing.Callable[(...), None]
    fill_global: typing.Callable[(...), None]
    sharedmem: object

    def __init__(self, gbgpu_backend_methods):
        # Populate LAT-side fields first so the order matches LAT's
        # backend layout exactly.
        assert isinstance(gbgpu_backend_methods, GBGPUBackendMethods)
        LISAToolsBackend.__init__(self, gbgpu_backend_methods)

        # GB-specific.
        self.GBTDIonTheFlyWrap = gbgpu_backend_methods.GBTDIonTheFlyWrap
        self.GBComputationGroupWrap = gbgpu_backend_methods.GBComputationGroupWrap
        self.get_ll = gbgpu_backend_methods.get_ll
        self.fill_global = gbgpu_backend_methods.fill_global
        self.sharedmem = gbgpu_backend_methods.sharedmem


# --- module-loader helper ------------------------------------------------

def _lat_methods_from_pycppdetector(_lat_pd, _gbt_interp, *, gpu: bool, xp):
    """Build a LAT methods dict from a `lisatools_backend_*.pycppdetector` module.

    Pulled out so each GBGPU loader (cpu/cuda11x/cuda12x/cuda13x) can
    share the populate code without copy-pasting 14 fields.
    """
    suffix = "GPU" if gpu else "CPU"
    return {
        "OrbitsWrap": getattr(_lat_pd, f"OrbitsWrap{suffix}"),
        "Orbits": getattr(_lat_pd, f"Orbits{suffix}"),
        "check_orbits": _lat_pd.check_orbits,
        "TDSplineTDIWaveformWrap": getattr(_lat_pd, f"TDSplineTDIWaveformWrap{suffix}"),
        "FDSplineTDIWaveformWrap": getattr(_lat_pd, f"FDSplineTDIWaveformWrap{suffix}"),
        "LISAResponseWrap": getattr(_lat_pd, f"LISAResponseWrap{suffix}"),
        "LISAResponse": getattr(_lat_pd, f"LISAResponse{suffix}"),
        "TDIConfigWrap": getattr(_lat_pd, f"TDIConfigWrap{suffix}"),
        "TDIConfig": getattr(_lat_pd, f"TDIConfig{suffix}"),
        # stft_tof merge (2026-06): XYZ sensitivity backend + galactic grid
        # reactivated on the LAT backend table; mirror them here.
        "SensitivityMatrixWrap": getattr(_lat_pd, f"XYZSensitivityMatrixWrap{suffix}"),
        "GalacticGridSetup": _lat_pd.GalacticGridSetup,
        "GalacticGridWrap": getattr(_lat_pd, f"GalacticGridWrap{suffix}"),
        "psd_likelihood": _lat_pd.psd_likelihood,
        "compute_logpdf": _lat_pd.compute_logpdf,
        # GBT is the single registrant for CubicSplineWrap (same pattern
        # as this package consuming LAT's OrbitsWrap).
        "CubicSplineWrap": getattr(_gbt_interp, f"CubicSplineWrap{suffix}"),
        "WDMSettingsWrap": getattr(_lat_pd, f"WDMSettingsWrap{suffix}"),
        "WDMDomainWrap": getattr(_lat_pd, f"WDMDomainWrap{suffix}"),
        "FDDomainWrap": getattr(_lat_pd, f"FDDomainWrap{suffix}"),
        "TDITypeDict": {"XYZ": _lat_pd.TDI_XYZ, "AET": _lat_pd.TDI_AET, "AE": _lat_pd.TDI_AE},
        "xp": xp,
    }


def _gb_methods_from_cgbgpu(_cgbgpu, *, gpu: bool):
    """Build the GB-specific portion of GBGPUBackendMethods."""
    suffix = "GPU" if gpu else "CPU"
    _gbcomp = getattr(_cgbgpu, f"GBGPUComputationWrap{suffix}")()
    return {
        "GBTDIonTheFlyWrap": getattr(_cgbgpu, f"GBTDIonTheFlyWrap{suffix}"),
        "GBComputationGroupWrap": getattr(_cgbgpu, f"GBComputationGroupWrap{suffix}"),
        "get_ll": _gbcomp.get_ll,
        "fill_global": _gbcomp.fill_global,
        "sharedmem": _gbcomp,
    }


# --- concrete backends ---------------------------------------------------


class GBGPUCpuBackend(CpuBackend, GBGPUBackend):
    """CPU backend backed by ``lisatools_backend_cpu.pycppdetector`` +
    ``gbgpu_backend_cpu.cgbgpu``."""

    _backend_name = "gbgpu_backend_cpu"
    _name = "gbgpu_cpu"

    def __init__(self, *args, **kwargs):
        CpuBackend.__init__(self, *args, **kwargs)
        GBGPUBackend.__init__(self, self.cpu_methods_loader())

    @staticmethod
    def cpu_methods_loader() -> GBGPUBackendMethods:
        try:
            import gbt_backend_cpu.interp
            import gbgpu_backend_cpu.cgbgpu
            import lisatools_backend_cpu.pycppdetector
        except (ModuleNotFoundError, ImportError) as e:
            raise BackendUnavailableException("'cpu' backend could not be imported.") from e

        numpy = GBGPUCpuBackend.check_numpy()
        return GBGPUBackendMethods(
            **_lat_methods_from_pycppdetector(
                lisatools_backend_cpu.pycppdetector, gbt_backend_cpu.interp, gpu=False, xp=numpy
            ),
            **_gb_methods_from_cgbgpu(gbgpu_backend_cpu.cgbgpu, gpu=False),
        )


def _make_cuda_loader(flavor):
    """Generate the cuda*x_module_loader function body."""

    def _loader() -> GBGPUBackendMethods:
        gb_mod_name = f"gbgpu_backend_{flavor}"
        lat_mod_name = f"lisatools_backend_{flavor}"
        try:
            import importlib

            gb_mod = importlib.import_module(f"{gb_mod_name}.cgbgpu")
            gbt_mod = importlib.import_module(f"gbt_backend_{flavor}.interp")
            lat_mod = importlib.import_module(f"{lat_mod_name}.pycppdetector")
        except (ModuleNotFoundError, ImportError) as e:
            raise BackendUnavailableException(
                f"'{flavor}' backend could not be imported."
            ) from e

        try:
            import cupy
        except (ModuleNotFoundError, ImportError) as e:
            raise MissingDependencies(
                f"'{flavor}' backend requires cupy",
                pip_deps=[f"cupy-{flavor}"],
            ) from e

        return GBGPUBackendMethods(
            **_lat_methods_from_pycppdetector(lat_mod, gbt_mod, gpu=True, xp=cupy),
            **_gb_methods_from_cgbgpu(gb_mod, gpu=True),
        )

    return _loader


class GBGPUCuda11xBackend(Cuda11xBackend, GBGPUBackend):
    """CUDA 11.x backend."""

    _backend_name: str = "gbgpu_backend_cuda11x"
    _name = "gbgpu_cuda11x"
    cuda11x_module_loader = staticmethod(_make_cuda_loader("cuda11x"))

    def __init__(self, *args, **kwargs):
        Cuda11xBackend.__init__(self, *args, **kwargs)
        GBGPUBackend.__init__(self, self.cuda11x_module_loader())


class GBGPUCuda12xBackend(Cuda12xBackend, GBGPUBackend):
    """CUDA 12.x backend."""

    _backend_name: str = "gbgpu_backend_cuda12x"
    _name = "gbgpu_cuda12x"
    cuda12x_module_loader = staticmethod(_make_cuda_loader("cuda12x"))

    def __init__(self, *args, **kwargs):
        Cuda12xBackend.__init__(self, *args, **kwargs)
        GBGPUBackend.__init__(self, self.cuda12x_module_loader())


class GBGPUCuda13xBackend(Cuda13xBackend, GBGPUBackend):
    """CUDA 13.x backend."""

    _backend_name: str = "gbgpu_backend_cuda13x"
    _name = "gbgpu_cuda13x"
    cuda13x_module_loader = staticmethod(_make_cuda_loader("cuda13x"))

    def __init__(self, *args, **kwargs):
        Cuda13xBackend.__init__(self, *args, **kwargs)
        GBGPUBackend.__init__(self, self.cuda13x_module_loader())
