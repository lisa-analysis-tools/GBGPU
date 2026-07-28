# ======================================
# == GBGPU project base configuration ==
# ======================================

# TODO:
#  - check cmake_minimum_required range
#  - check if project-specific CMake function are needed

# ---- CMake related definitions ----
cmake_minimum_required(VERSION 3.23...3.31)

# ---- Main project definition ----
project(gbgpu VERSION ${SKBUILD_PROJECT_VERSION} LANGUAGES CXX)

# ---- Find required dependencies ----
find_package(Python COMPONENTS Interpreter Development.Module NumPy REQUIRED)

# Phase 3M (2026-06-04): sprint-wide pybind11 -> nanobind migration.
# nanobind ships its CMake config under `$(python -m nanobind --cmake_dir)`;
# resolve here once so cutils/CMakeLists.txt can call `nanobind_add_module`.
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_VARIABLE nanobind_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE)
find_package(nanobind CONFIG REQUIRED)

# This line defines the install directory variables
include(GNUInstallDirs)

# ---- Import project-specific CMake functions ----
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# ---- Define project specific options ----

# GBGPU_WITH_GPU controls the activation of both GPU and CPU backend compilations.
# Its allowed values are:
#
# * AUTO: enable CPU backend, and enable GPU backend if CUDA toolchain is found
#   in environment, otherwise disable it
# * ON: enable CPU backend, and enable GPU backend, fail if CUDA toolchain is
#   not available
# * OFF: enable CPU backend and disable GPU backend
# * ONLY: disable CPU backend and enable GPU backend (used in plugin wheel build
#   process)
# * BARE: disable both CPU and GPU backends
set(
  GBGPU_WITH_GPU "AUTO" CACHE
  STRING "Whether to compile GPU backend"
)
set_property(
  CACHE GBGPU_WITH_GPU PROPERTY
  STRINGS "AUTO" "ON" "OFF" "ONLY" "BARE"
)

# GBGPU_WITH_SHAREDMEM controls whether the LEGACY SharedMemoryGBGPU.cu kernel
# family (SharedMemoryWaveComp / LikeComp / SwapLikeComp / ChiSquaredComp /
# GenerateGlobal / FstatLikeComp / InfoMatComp) is compiled into cgbgpu.
#
# It is the single most expensive part of a GBGPU build: 4k lines of heavily
# templated CUDA, and the ONLY consumer of cuFFTDx -- so turning it OFF also
# drops the configure-time ~nvidia-mathdx tarball download + extract +
# find_package(mathdx) entirely. A GPU build with it OFF needs no network
# access at configure time and compiles substantially faster.
#
# ON (default) preserves current behaviour. OFF is for builds that only
# exercise the modern kernel families (GBTDIonTheFly, chunked-heterodyne,
# signal-heterodyne / sig-het in-model). With it OFF the seven
# ``SharedMemory*_wrap`` methods are still BOUND (identical Python surface)
# but raise ``RuntimeError`` when called. Paths that still need ON:
# GB F-stat RJ births (``GBGPU.get_fstat_ll``), synthetic GB injections and
# the FD-domain global-template subtraction
# (``GBGPU.generate_global_template``), ``GBAETWaveform`` (``run_wave``),
# and ``gathergalaxy``'s ``swap_likelihood_difference``.
option(
  GBGPU_WITH_SHAREDMEM
  "Compile the legacy SharedMemoryGBGPU.cu kernels (requires cuFFTDx/mathdx on GPU)"
  ON
)

# GBGPU_CUDA_ARCH will be passed as the CUDA_ARCHITECTURES property for the CUDA
# backend if it is compiled. See the documentation of CUDA_ARCHITECTURES:
# https://cmake.org/cmake/help/latest/prop_tgt/CUDA_ARCHITECTURES.html
#
# The default is set below (after CUDAToolkit is found) to "80;90" -- Ampere +
# Hopper, the sprint's GPU targets -- rather than "native". "native" needs
# CMake to see a GPU at configure time; on a cluster BUILD/LOGIN node with no
# visible device that detection fails and nvcc silently falls back to its own
# default (sm_52 on CUDA 12.x) -- which is below the sm_60 required for
# `atomicAdd(double*, double)`, so gb_tdi_on_the_fly.cu and LAT's
# lat_chunked_het_kernels.hh fail to compile with "no instance of overloaded
# function atomicAdd matches (double*, double)".
#
# Pass -DGBGPU_CUDA_ARCH=<arch> to override (70=V100, 75=Turing, 80=A100,
# 90=H100); a single arch builds fastest. V100/Turing are NOT in the default.

set(
  GBGPU_MARCH "native"
  CACHE STRING "Value of the -march compiler option if supported by compiler"
)

# ---- Phony target for project specific properties ----
add_library(gbgpu INTERFACE)

# ---- Enable building the CPU version of backends by default ----
set_target_properties(gbgpu PROPERTIES WITH_CPU ON)

# ---- Legacy SharedMemoryGBGPU.cu kernels (see GBGPU_WITH_SHAREDMEM above) ----
set_target_properties(gbgpu PROPERTIES WITH_SHAREDMEM ${GBGPU_WITH_SHAREDMEM})
if(GBGPU_WITH_SHAREDMEM)
  message(STATUS "GBGPU legacy SharedMemoryGBGPU kernels: ON (cuFFTDx/mathdx required for GPU).")
else()
  message(STATUS "GBGPU legacy SharedMemoryGBGPU kernels: OFF -- SharedMemory*_wrap will \
raise RuntimeError; no cuFFTDx/mathdx download.")
endif()

# ---- Test whether the GBGPU_MARCH option is supported by CXX compiler ----
include(CheckCXXCompilerFlag)
set(GBGPU_MARCH_CXX_OPT "-march=${GBGPU_MARCH}")
check_cxx_compiler_flag("${GBGPU_MARCH_CXX_OPT}" CXX_COMPILER_SUPPORTS_GBGPU_MARCH)
if(CXX_COMPILER_SUPPORTS_GBGPU_MARCH)
  set_property(TARGET gbgpu PROPERTY CXX_MARCH
    "${GBGPU_MARCH_CXX_OPT}")
  message(STATUS "The CXX compiler supports option '${GBGPU_MARCH_CXX_OPT}'.")
else()
  message(
    WARNING "The CXX compiler does not support option '${GBGPU_MARCH_CXX_OPT}'. \
      It will be ignored.")
endif()

# ---- Optionnally check if GPU is supported ----
if(GBGPU_WITH_GPU STREQUAL "AUTO")
  include(CheckLanguage)
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    find_package(CUDAToolkit)
  endif()
  if(CMAKE_CUDA_COMPILER AND CUDAToolkit_FOUND)
    message(
      STATUS
      "GBGPU GPU support was set to AUTO and will be turned ON as CUDA and \
CUDA Toolkit are available with CUDA version \
${CUDAToolkit_VERSION_MAJOR}.")
    set_target_properties(gbgpu PROPERTIES WITH_GPU ON)
  else()
    message(
      STATUS
      "GBGPU GPU support was set to AUTO and will be turned OFF as CUDA and \
CUDA Toolkit are not found (CMAKE_CUDA_COMPILER:${CMAKE_CUDA_COMPILER} \
CUDAToolkit_FOUND:${CUDAToolkit_FOUND}).")
    set_target_properties(gbgpu PROPERTIES WITH_GPU OFF)
  endif()
elseif(GBGPU_WITH_GPU STREQUAL "ONLY")
  message(
    STATUS
      "GBGPU GPU support is set to ON and CPU support is set to OFF (use only to \
      build a standalone plugin).")
  set_target_properties(gbgpu PROPERTIES WITH_GPU ON)
  set_target_properties(gbgpu PROPERTIES WITH_CPU OFF)
elseif(GBGPU_WITH_GPU STREQUAL "BARE")
  message(STATUS "GBGPU GPU and CPU support are disabled (use only to build a \
      non-functional pure-python release).")
  set_target_properties(gbgpu PROPERTIES WITH_GPU OFF)
  set_target_properties(gbgpu PROPERTIES WITH_CPU OFF)
else()
  message(STATUS "GBGPU GPU support is set to ${GBGPU_WITH_GPU}.")
  set_target_properties(gbgpu PROPERTIES WITH_GPU ${GBGPU_WITH_GPU})
endif()

# ---- Handle GPU support ----
get_target_property(_GBGPU_WITH_GPU gbgpu WITH_GPU)
if(_GBGPU_WITH_GPU)
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)

  # Default: sm_80 (Ampere) + sm_90 (Hopper) -- the sprint's actual GPU targets.
  # Deliberately NARROWER than LISAanalysistools' "70;75;80;86;90": two archs
  # instead of five roughly halves an already-slow nvcc build, and CUDA binary
  # compatibility is guaranteed forward across MINOR revisions within a major
  # generation, so the sm_80 cubin also covers sm_86 (A40/A6000/RTX30) and
  # sm_89 (Ada/L40). NOT covered: sm_70 (V100) and sm_75 (Turing) -- those need
  # an explicit -DGBGPU_CUDA_ARCH=70 or they fail at RUNTIME with "no kernel
  # image is available for execution on the device". There is no PTX entry, so
  # future majors (sm_100+) also need an explicit value.
  #
  # This is the fix for the "native" default: "native" needs CMake to see a GPU
  # at CONFIGURE time, and on a GPU-less build/login node nvcc falls back to
  # sm_52 -- below the sm_60 floor below.
  set(DEFAULT_ARCHS "80;90")

  set(
    GBGPU_CUDA_ARCH "${DEFAULT_ARCHS}"
    CACHE STRING "CUDA Architectures targeted for GBGPU compilation (see doc of \
            CMAKE_CUDA_ARCHITECTURES)."
  )

  # An explicitly-passed -DGBGPU_CUDA_ARCH= wins (it is already in the cache);
  # guard only against an EMPTY value, which would leave the target property
  # unset and hand the choice to nvcc's sm_52 default.
  if(NOT GBGPU_CUDA_ARCH)
    set(GBGPU_CUDA_ARCH "${DEFAULT_ARCHS}" CACHE STRING
        "CUDA Architectures targeted for GBGPU compilation (see doc of \
            CMAKE_CUDA_ARCHITECTURES)." FORCE)
  endif()

  # sm_60 is the floor for atomicAdd(double*, double), used by the chunked-het
  # fill_global kernels. Fail LOUDLY at configure time rather than with a wall
  # of nvcc overload-resolution notes.
  foreach(_arch IN LISTS GBGPU_CUDA_ARCH)
    string(REGEX MATCH "^[0-9]+" _arch_num "${_arch}")
    if(_arch_num AND _arch_num LESS 60)
      message(FATAL_ERROR
        "GBGPU_CUDA_ARCH contains '${_arch}', below sm_60. The chunked-het "
        "fill_global kernels use atomicAdd(double*, double), which requires "
        "compute capability >= 6.0. Use e.g. -DGBGPU_CUDA_ARCH=80 (A100), "
        "90 (H100), or 70 (V100).")
    endif()
  endforeach()

  # Also seed the GLOBAL default, as LAT does. Every GPU target today gets the
  # CUDA_ARCHITECTURES property explicitly (apply_gpu_backend_common_options +
  # gbgpu_gpu_cgbgpu), but a target added later without it would otherwise
  # silently inherit nvcc's sm_52 and reproduce the atomicAdd(double*) failure.
  set(CMAKE_CUDA_ARCHITECTURES ${GBGPU_CUDA_ARCH})

  message(STATUS "GBGPU CUDA Architectures set to: ${GBGPU_CUDA_ARCH}")
endif()

# ---- Include sources ----
add_subdirectory(src)
