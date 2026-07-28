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
set(
  GBGPU_CUDA_ARCH "native"
  CACHE STRING "CUDA Architecture targetted for GBGPU compilation (see doc of \
          CMAKE_CUDA_ARCHITECTURES)."
)

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
endif()

# ---- Include sources ----
add_subdirectory(src)
