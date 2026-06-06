# =============================
# ==== BACKEND COMPILATION ====
# =============================

# TODO: how can we avoid moving source and header files in this directory ?
#include_directories(BEFORE "include")
#include_directories(BEFORE "src")

# In the project root CMakeLists.txt, we defined a "gbgpu" interface
# target with properties WITH_CPU and WITH_GPU defining whether the CPU and a
# GPU backend need to be compiled. Let's retrieve these information here:
get_target_property(GBGPU_WITH_CPU gbgpu WITH_CPU)
get_target_property(GBGPU_WITH_GPU gbgpu WITH_GPU)

# Locate the gpubackendtools cutils directory so we can find gbt_global.h /
# common helpers that LISAanalysistools' Detector.cu pulls in. Mirrors how
# lisa-on-gpu/src/fastlisaresponse/cutils/CMakeLists.txt does it.
execute_process(
  COMMAND "${Python_EXECUTABLE}" -c "if True:
    import gpubackendtools
    print(gpubackendtools.__file__.split('__init__.py')[0] + 'cutils')"
  OUTPUT_VARIABLE GBT_CUTILS
  OUTPUT_STRIP_TRAILING_WHITESPACE)

# Locate the LISAanalysistools cutils directory (same pattern as lisa-on-gpu).
# Replaces the prior "file(DOWNLOAD ... refs/heads/main ...)" pattern that
# pulled stale snapshots from GitHub; instead we consume LAT's local install
# directly, which guarantees Detector.cu/.hpp + global.hpp are version-
# matched with the lisaanalysistools build-system dependency declared in
# this repo's pyproject.toml.
execute_process(
  COMMAND "${Python_EXECUTABLE}" -c "if True:
    import lisatools
    print(lisatools.__file__.split('__init__.py')[0] + 'cutils')"
  OUTPUT_VARIABLE LISATOOLS_CUTILS
  OUTPUT_STRIP_TRAILING_WHITESPACE)

if(NOT EXISTS "${LISATOOLS_CUTILS}/Detector.cu")
  message(FATAL_ERROR
          "${LISATOOLS_CUTILS}/Detector.cu does not exist -- "
          "lisaanalysistools install is broken or LISATOOLS_CUTILS detection "
          "failed. Ensure 'pip install lisaanalysistools' completed before "
          "building gbgpu.")
endif()

message(STATUS "LISATOOLS_CUTILS location is ${LISATOOLS_CUTILS}.")

# Phase 3L.7g (2026-06-04): cgbgpu now includes GBT's Interpolate.cu
# (fit_cubic_spline_thomas + friends) because the GB chunked-het
# pipeline relies on it. Pull in GBT's cmake_functions.cmake so we can
# call get_lapacke() the same way lisa-on-gpu + GBT itself do.
set(GBT_CMAKE_TOOLS "${GBT_CUTILS}/cmake_functions.cmake")
if(NOT EXISTS "${GBT_CMAKE_TOOLS}")
  message(FATAL_ERROR
    "${GBT_CMAKE_TOOLS} not found -- gpubackendtools install is broken or "
    "GBT_CUTILS detection failed.")
endif()
include(${GBT_CMAKE_TOOLS})
# GBT's get_lapacke() reads GBT_LAPACKE_DETECT_WITH + GBT_LAPACKE_FETCH.
# Both must be defined or the `if (NOT ${VAR} ...)` test inside the helper
# fails to parse. Default both to AUTO so we pick up a system LAPACKE if
# present, falling back to the CPM-fetched copy otherwise -- same defaults
# lisa-on-gpu uses in its root CMakeLists.
if(NOT DEFINED GBT_LAPACKE_DETECT_WITH)
  set(GBT_LAPACKE_DETECT_WITH "AUTO")
endif()
if(NOT DEFINED GBT_LAPACKE_FETCH)
  set(GBT_LAPACKE_FETCH "AUTO")
endif()
if(NOT DEFINED GBT_LAPACKE_EXTRA_LIBS)
  set(GBT_LAPACKE_EXTRA_LIBS "")
endif()
if(GBGPU_WITH_CPU OR GBGPU_WITH_GPU)
  enable_language(Fortran)
  get_lapacke()
endif()

# Adapter to let inplace editable install work: the compiled backend will be
# placed into the source-tree in the 'src' directory:
# TODO: is this needed ?
if(SKBUILD_STATE STREQUAL "editable")
  set(BACKEND_BASE_OUTPUT_DIRECTORY "${gbgpu_BINARY_DIR}/src")
else()
  set(BACKEND_BASE_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")
endif()

# apply_gpu_backend_common_options
# --------------------------------
#
# This method applies some common directive to GPU backend targets. It:
#
# * Expects a single "libname" argument
# * Expects the target to be named "gbgpu_gpu_${libname}"
# * Defines the LIBRARY_OUTPUT_DIRECTORY property
# * Defines the OUTPUT_NAME property
# * Installs the target in the GPU backend directory (e.g.
#   'gbgpu_backend_cuda12x')
# * Ensures the target includes the NumPy header directory
# * Disable NumPy deprecated API
# * Ensures the target links against CUDA libraries (cuBLAS, cuSPARSE, ...)
# * Defines the CUDA_ARCHITECTURE property
#
# Usage example: apply_gpu_backend_common_options(gbgpu_utils_wrap)
function(apply_gpu_backend_common_options libname)
  set(target_name "gbgpu_gpu_${libname}")
  set(backend_name "gbgpu_backend_cuda${CUDAToolkit_VERSION_MAJOR}x")
  set_property(
    TARGET ${target_name}
    PROPERTY LIBRARY_OUTPUT_DIRECTORY
             "${BACKEND_BASE_OUTPUT_DIRECTORY}/${backend_name}")
  set_property(TARGET ${target_name} PROPERTY OUTPUT_NAME ${libname})

  install(TARGETS ${target_name} DESTINATION ${backend_name})

  target_include_directories(${target_name} PRIVATE ${Python_NumPy_INCLUDE_DIR})
  target_compile_definitions(${target_name}
                             PRIVATE NPY_NO_DEPRECATED_API=NPY_1_9_API_VERSION)
  # __CUDA_COMPILATION__ is the sprint-wide CPU/GPU class-name alias toggle
  # (see lat_tdi_on_the_fly.hh / gb_tdi_on_the_fly.hh / binding_gbgpu.hpp).
  # Without it, .cxx TUs in the GPU build (binding_gbgpu.cxx etc.) compile
  # against the CPU-suffixed class names (GBTDIonTheFlyCPU,
  # LISATDIonTheFlyCPU, ...) and the .so fails to load with
  # ``undefined symbol: _ZTV16GBTDIonTheFlyCPU`` -- the CPU vtable is
  # never emitted in the GPU build because the .cu sources DO get
  # __CUDACC__ from nvcc and alias to *GPU instead. PUBLIC so the define
  # propagates to every TU compiled as part of this target. Mirrors what
  # GBT's canonical apply_gpu_backend_common_options does in
  # cmake_functions.cmake (the local override below was missing this).
  target_compile_definitions(${target_name} PUBLIC __CUDA_COMPILATION__)
  target_link_libraries(${target_name} PUBLIC CUDA::cudart CUDA::cublas
                                              CUDA::cusparse)
  set_property(TARGET ${target_name} PROPERTY CUDA_ARCHITECTURES
                                              ${GBGPU_CUDA_ARCH})
endfunction()

function(apply_cpu_backend_common_options libname)
  set(target_name "gbgpu_cpu_${libname}")
  set_property(
    TARGET ${target_name}
    PROPERTY LIBRARY_OUTPUT_DIRECTORY
             "${BACKEND_BASE_OUTPUT_DIRECTORY}/gbgpu_backend_cpu")
  set_property(TARGET ${target_name} PROPERTY OUTPUT_NAME ${libname})

  install(TARGETS ${target_name} DESTINATION gbgpu_backend_cpu)

  get_target_property(GBGPU_CXX_MARCH_OPT gbgpu CXX_MARCH)
  if(GBGPU_CXX_MARCH_OPT)
    target_compile_options(${target_name} PRIVATE "${GBGPU_CXX_MARCH_OPT}")
  endif()

  target_include_directories(${target_name} PRIVATE ${Python_NumPy_INCLUDE_DIR})
  target_compile_definitions(${target_name}
                             PRIVATE NPY_NO_DEPRECATED_API=NPY_1_9_API_VERSION)
endfunction()

# * * * * * * * * * * * * * * * * * * * *
# * * Definition of compiled backends * *
# * * * * * * * * * * * * * * * * * * * *

# * * * * * * * * * * * * * * * * * * * *
# * * Definition of compiled backends * *
# * * * * * * * * * * * * * * * * * * * *

# ----------------
# --- utils ---
# ----------------

# 0. Copy lisaanalysistools C/CUDA files from the installed LAT package.
# Was previously a hardcoded GitHub URL download from refs/heads/main, which
# (a) required network access at configure time, (b) silently pulled stale
# snapshots that didn't match the version-pinned lisaanalysistools build
# dep, and (c) broke when LAT's main branch advanced past the API GBGPU
# expects. The new pattern reads from ${LISATOOLS_CUTILS} (resolved above
# via 'import lisatools'), so the consumed sources are always the ones the
# user actually installed.
file(COPY "${LISATOOLS_CUTILS}/Detector.cu"  DESTINATION "${CMAKE_CURRENT_SOURCE_DIR}")
file(COPY "${LISATOOLS_CUTILS}/Detector.hpp" DESTINATION "${CMAKE_CURRENT_SOURCE_DIR}")
file(COPY "${LISATOOLS_CUTILS}/global.hpp"   DESTINATION "${CMAKE_CURRENT_SOURCE_DIR}")
# Phase GBGPU.pybind.bulk (2026-06-04): the per-module Cython binding
# pipeline (gbgpu_utils_wrap.pyx -> utils, sharedmemgbgpu.pyx ->
# sharedmem) is retired. Their 9 free functions now live as methods on
# GBGPUComputationWrap inside the cgbgpu pybind11 module. The underlying
# gbgpu_utils.cu / SharedMemoryGBGPU.cu sources fold into cgbgpu
# directly, mirroring LAT's binding_flr.cxx + Detector.cxx layout (one
# .so per backend wheel).

# CPU build: copy .cu sources to .cxx for the host C++ compiler.
if(GBGPU_WITH_CPU)
  add_custom_command(
    OUTPUT "gbgpu_utils.cxx"
    COMMENT "Copy gbgpu_utils.cu to gbgpu_utils.cxx (for cgbgpu CPU)"
    COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/gbgpu_utils.cu"
            "${CMAKE_CURRENT_BINARY_DIR}/gbgpu_utils.cxx"
    DEPENDS "gbgpu_utils.cu"
    VERBATIM)

  add_custom_command(
    OUTPUT "Detector.cxx"
    COMMENT "Copy Detector.cu to Detector.cxx (for cgbgpu CPU)"
    COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/Detector.cu"
            "${CMAKE_CURRENT_BINARY_DIR}/Detector.cxx"
    DEPENDS "Detector.cu"
    VERBATIM)

  add_custom_command(
    OUTPUT "SharedMemoryGBGPU.cxx"
    COMMENT "Copy SharedMemoryGBGPU.cu to SharedMemoryGBGPU.cxx (for cgbgpu CPU)"
    COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/SharedMemoryGBGPU.cu"
            "${CMAKE_CURRENT_BINARY_DIR}/SharedMemoryGBGPU.cxx"
    DEPENDS "SharedMemoryGBGPU.cu"
    VERBATIM)

  # Phase 3L.7d (2026-06-04): the GBTDIonTheFly method bodies + the
  # gb_run_wave_tdi kernel/wrap live in gb_tdi_on_the_fly.cu. Copy to
  # .cxx for the host C++ compiler (CPU backend).
  add_custom_command(
    OUTPUT "gb_tdi_on_the_fly.cxx"
    COMMENT "Copy gb_tdi_on_the_fly.cu to gb_tdi_on_the_fly.cxx (for cgbgpu CPU)"
    COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/gb_tdi_on_the_fly.cu"
            "${CMAKE_CURRENT_BINARY_DIR}/gb_tdi_on_the_fly.cxx"
    DEPENDS "gb_tdi_on_the_fly.cu"
    VERBATIM)

  # Phase 3L.7d (2026-06-04): GBTDIonTheFly inherits from LISATDIonTheFly,
  # whose method bodies live in LAT's lat_tdi_on_the_fly.cu (Phase 3L.5).
  # Copy-compile that source into cgbgpu so the virtual-class typeinfo
  # is present in the cgbgpu .so (mirrors how lisa-on-gpu's tdionthefly
  # module already copy-compiles lat_tdi_on_the_fly.cu since Phase 3L.5).
  add_custom_command(
    OUTPUT "lat_tdi_on_the_fly.cxx"
    COMMENT "Copy LAT's lat_tdi_on_the_fly.cu to lat_tdi_on_the_fly.cxx (for cgbgpu CPU)"
    COMMAND ${CMAKE_COMMAND} -E copy "${LISATOOLS_CUTILS}/lat_tdi_on_the_fly.cu"
            "${CMAKE_CURRENT_BINARY_DIR}/lat_tdi_on_the_fly.cxx"
    DEPENDS "${LISATOOLS_CUTILS}/lat_tdi_on_the_fly.cu"
    VERBATIM)

  # Phase 3L.7g (2026-06-04): gb_tdi_on_the_fly.cu uses
  # fit_cubic_spline_thomas() (GBT's LAPACKE tridiagonal cubic-spline
  # solver) for orbit-spline + signal-spline construction in the
  # chunked-het pipeline. Compile GBT's Interpolate.cu into cgbgpu so the
  # symbol is resolved without depending on lisa-on-gpu's tdionthefly
  # module (which previously held it). Same copy-compile pattern as
  # lisa-on-gpu's CMakeLists.txt uses for its own static archive.
  add_custom_command(
    OUTPUT "Interpolate.cxx"
    COMMENT "Copy GBT's Interpolate.cu to Interpolate.cxx (for cgbgpu CPU)"
    COMMAND ${CMAKE_COMMAND} -E copy "${GBT_CUTILS}/Interpolate.cu"
            "${CMAKE_CURRENT_BINARY_DIR}/Interpolate.cxx"
    DEPENDS "${GBT_CUTILS}/Interpolate.cu"
    VERBATIM)
endif()


# III. Declare the GPU backend
if(GBGPU_WITH_GPU)
  # gbgpu_gpu_utils + gbgpu_gpu_sharedmem (Cython) retired -- functions
  # migrated to cgbgpu (see below). The cufftdx + nvidia-mathdx download
  # logic that lived here is preserved for SharedMemoryGBGPU.cu, which
  # we still compile into cgbgpu.

  file(DOWNLOAD
      "https://developer.nvidia.com/downloads/compute/cuFFTDx/redist/cuFFTDx/cuda12/nvidia-mathdx-25.06.1-cuda12.tar.gz" # The URL of the file to download
      "./src/gbgpu/cutils/nvidia-mathdx-25.06.1-cuda12.tar.gz" # The local path to save the file
      INACTIVITY_TIMEOUT 10 # Optional: Terminate after 10 seconds of inactivity
      # TIMEOUT 30 # Optional: Terminate after 30 seconds total
      STATUS download_status # Optional: Store download status in a variable
      # LOG download_log # Optional: Store a log of the operation in a variable
      # SHOW_PROGRESS # Optional: Display progress information
  )

  execute_process(COMMAND tar -xzf "./src/gbgpu/cutils/nvidia-mathdx-25.06.1-cuda12.tar.gz" -C "./src/gbgpu/cutils/")
  # Define variables for clarity (optional)
  # set(ARCHIVE_FILE "./src/gbgpu/cutils/nvidia-mathdx-25.06.1-cuda12.tar.gz")
  # set(EXTRACT_DIR "./src/gbgpu/cutils/") # extract to cutils directory

  # # Create the extraction directory if it doesn't exist
  # file(MAKE_DIRECTORY "${EXTRACT_DIR}")

  # # Execute the tar command to extract the archive
  # execute_process(
  #     COMMAND tar xzf "${ARCHIVE_FILE}"
  #     WORKING_DIRECTORY "${EXTRACT_DIR}"
  #     RESULT_VARIABLE TAR_RESULT
  #     OUTPUT_VARIABLE TAR_OUTPUT
  #     ERROR_VARIABLE TAR_ERROR
  # )

  # # Optional: Check for errors
  # if(NOT TAR_RESULT EQUAL 0)
  #     message(FATAL_ERROR "Failed to extract archive: ${TAR_ERROR}")
  # endif()
  message(STATUS "CMAKE_CURRENT_SOURCE_DIR: ${CMAKE_CURRENT_SOURCE_DIR}")
  list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/nvidia-mathdx-25.06.1/nvidia/mathdx/25.06/lib/cmake")
  set(CMAKE_PREFIX_PATH "${CMAKE_CURRENT_SOURCE_DIR}/nvidia-mathdx-25.06.1/nvidia/mathdx/25.06/lib/cmake/" "${CMAKE_PREFIX_PATH}")
  set(mathdx_cufftdx_DISABLE_CUTLASS TRUE)
  #set(cufftdx_ROOT "./src/gbgpu/cutils/nvidia-mathdx-25.06.1/nvidia/mathdx/25.06/")
  #find_package(cufftdx REQUIRED CONFIG)
  set(mathdx_DIR "${CMAKE_CURRENT_SOURCE_DIR}/nvidia-mathdx-25.06.1/nvidia/mathdx/25.06/lib/cmake/mathdx")
  find_package(mathdx REQUIRED COMPONENTS cufftdx CONFIG PATHS "${CMAKE_CURRENT_SOURCE_DIR}/nvidia-mathdx-25.06.1/nvidia/mathdx/25.06/")
endif()


# ----------------
# --- cgbgpu (pybind11 module skeleton) ---
# ----------------
#
# Phase GBGPU.pybind (2026-06-03): mirror of the BBHx pybind skeleton
# (BBHx had its skeleton on origin/pybind; GBGPU had no prior pybind
# work so this is fresh against the post-Phase-3L LAT setup). The
# pybind11 module is reserved for GB-specific wrappers (gbgpu_utils +
# SharedMemoryGBGPU as they migrate from the existing Cython modules)
# and the GBTDIonTheFly + GBComputationGroup + gb_wdm_het_* chunked-
# heterodyne kernels scheduled for Phase 3L.7. The skeleton commit
# ships an empty GBGPUComputationWrap; subsequent commits populate it.
# The single-registrant rule is enforced via
# static_assert(!LISATOOLS_IS_WRAPPER_OWNER) in binding_gbgpu.cxx --
# GBGPU must NOT register OrbitsWrap, TDIConfigWrap, LISAResponseWrap,
# WDM*/FD*/Spline* (those are LAT's).

if(GBGPU_WITH_CPU)
  # cgbgpu CPU build: one .so per backend wheel containing the pybind11
  # binding + gbgpu_utils + SharedMemoryGBGPU + LAT's Detector +
  # (Phase 3L.7d) GBTDIonTheFly bodies + LAT's LISATDIonTheFly typeinfo.
  nanobind_add_module(gbgpu_cpu_cgbgpu
    binding_gbgpu.cxx
    Detector.cxx
    gbgpu_utils.cxx
    SharedMemoryGBGPU.cxx
    gb_tdi_on_the_fly.cxx
    lat_tdi_on_the_fly.cxx
    Interpolate.cxx)
  apply_cpu_backend_common_options(cgbgpu)
  target_include_directories(gbgpu_cpu_cgbgpu PRIVATE
    "${LISATOOLS_CUTILS}"        # ReturnPointerBase + lisatools_header_abi.hpp + LAT Detector.hpp
    "${GBT_CUTILS}"              # gbt_global.h, pybind11_cuda_array_interface.hpp
    "${GBT_LAPACKE_INCLUDE_DIR}" # lapacke.h (for Interpolate.cu's tridiagonal solver)
  )
  target_link_libraries(gbgpu_cpu_cgbgpu PRIVATE
    ${GBT_LAPACKE_LIBS} ${GBT_LAPACKE_EXTRA_LIBS})
  # NB: cuda_complex.hpp deliberately omitted -- pulled transitively via
  # ${GBT_CUTILS}/gbt_global.h (GPUBackendTools owns the canonical copy).
  target_sources(gbgpu_cpu_cgbgpu PUBLIC FILE_SET HEADERS FILES
                                       binding_gbgpu.hpp gbgpu_utils.hh
                                       SharedMemoryGBGPU.hpp
                                       global.hpp Detector.hpp
                                       global.h
                                       common.hpp fp16_common.hpp)
  install(TARGETS gbgpu_cpu_cgbgpu
          LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}/python${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR}/site-packages/gbgpu)
endif()

if(GBGPU_WITH_GPU)
  # cgbgpu GPU build: same layout, .cu sources compiled directly by
  # nvcc, plus the mathdx::cufftdx link for SharedMemoryGBGPU.cu. Phase
  # 3L.7d adds gb_tdi_on_the_fly.cu (GBTDIonTheFly method bodies +
  # gb_run_wave_tdi kernel/wrap) and pulls in LAT's lat_tdi_on_the_fly.cu
  # so the LISATDIonTheFly virtual-class typeinfo lives in cgbgpu's
  # .so (same pattern as lisa-on-gpu's tdionthefly module since
  # Phase 3L.5).
  nanobind_add_module(gbgpu_gpu_cgbgpu
    binding_gbgpu.cxx
    "${LISATOOLS_CUTILS}/Detector.cu"
    "${LISATOOLS_CUTILS}/lat_tdi_on_the_fly.cu"
    "${GBT_CUTILS}/Interpolate.cu"
    gbgpu_utils.cu
    SharedMemoryGBGPU.cu
    gb_tdi_on_the_fly.cu)
  apply_gpu_backend_common_options(cgbgpu)
  target_include_directories(gbgpu_gpu_cgbgpu PRIVATE
    "${LISATOOLS_CUTILS}"
    "${GBT_CUTILS}"
    "${GBT_LAPACKE_INCLUDE_DIR}" # lapacke.h (for Interpolate.cu's tridiagonal solver)
  )
  set_property(TARGET gbgpu_gpu_cgbgpu PROPERTY CUDA_ARCHITECTURES ${GBGPU_CUDA_ARCH})
  set_property(TARGET gbgpu_gpu_cgbgpu PROPERTY CUDA_SEPARABLE_COMPILATION ON)
  set_property(TARGET gbgpu_gpu_cgbgpu PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
  set_property(TARGET gbgpu_gpu_cgbgpu PROPERTY POSITION_INDEPENDENT_CODE ON)
  target_link_libraries(gbgpu_gpu_cgbgpu PUBLIC CUDA::cudart CUDA::cublas
      CUDA::cusparse mathdx::cufftdx
      ${GBT_LAPACKE_LIBS} ${GBT_LAPACKE_EXTRA_LIBS})
  # NB: cuda_complex.hpp deliberately omitted -- pulled transitively via
  # ${GBT_CUTILS}/gbt_global.h (GPUBackendTools owns the canonical copy).
  target_sources(gbgpu_gpu_cgbgpu PUBLIC FILE_SET HEADERS FILES
                                       binding_gbgpu.hpp gbgpu_utils.hh
                                       SharedMemoryGBGPU.hpp
                                       global.hpp Detector.hpp
                                       global.h
                                       common.hpp fp16_common.hpp)
  install(TARGETS gbgpu_gpu_cgbgpu
          LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}/python${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR}/site-packages/gbgpu)
endif()


# Declare the CPU backend
# We compile the files in a module called gbgpu_utils_cpu
# if(GBGPU_WITH_CPU)

#   set(module_name "gbgpu_cpu_utils")
#   set(cythonized_file "gbgpu_utils_wrap_cpu.cxx")

#   # Process gbgpu_utils_wrap.pyx into a C++ file
#   add_custom_command(
#     OUTPUT "${cythonized_file}"
#     COMMENT "Cythonize gbgpu_utils_wrap.pyx into ${cythonized_file}"
#     COMMAND
#       Python::Interpreter -m cython "${CMAKE_CURRENT_SOURCE_DIR}/gbgpu_utils_wrap.pyx"
#       -3 -+
#       --output-file "${CMAKE_CURRENT_BINARY_DIR}/${cythonized_file}"
#       --module-name "${module_name}"
#       -I "${CMAKE_CURRENT_SOURCE_DIR}"
#     DEPENDS "gbgpu_utils_wrap.pyx"
#     VERBATIM)

#   # Copy CUDA file into a C++
#   # TODO: use cmake -E create_hardlink ?
#   add_custom_command(
#     OUTPUT "gbgpu_utils.cxx"
#     COMMENT "Copy gbgpu_utils.cu to gbgpu_utils.cxx"
#     COMMAND
#       ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/gbgpu_utils.cu"
#       "${CMAKE_CURRENT_BINARY_DIR}/gbgpu_utils.cxx"
#     DEPENDS "gbgpu_utils.cu"
#     VERBATIM)

#   # Declare the python compiled module
#   python_add_library(
#     "${module_name}"
#     MODULE WITH_SOABI "${cythonized_file}" gbgpu_utils.cxx)

#   # Compilation options
#   target_include_directories(
#     "${module_name}"
#     PRIVATE ${Python_NumPy_INCLUDE_DIR})

#   target_compile_definitions(
#     "${module_name}"
#     PRIVATE NPY_NO_DEPRECATED_API=NPY_1_9_API_VERSION)

#   target_sources(
#     "${module_name}" PUBLIC FILE_SET HEADERS
#     FILES cuda_complex.hpp global.h gbgpu_utils.hh)

#   get_target_property(GBGPU_CXX_MARCH_OPT gbgpu CXX_MARCH)
#   if(GBGPU_CXX_MARCH_OPT)
#     target_compile_options(
#       "${module_name}"
#       PRIVATE "${GBGPU_CXX_MARCH_OPT}")
#   endif()

#   # Installation options
#   install(TARGETS "${module_name}" DESTINATION gbgpu_backend_cpu)

# endif()

# # III. Declare the GPU backend
# # We compile the files in a module called gbgpu_utils_gpu
# if(GBGPU_WITH_GPU)

#   set(module_name "gbgpu_utils")
#   set(cythonized_file "gbgpu_utils_wrap_gpu.cxx")
#   set(backend_name "gbgpu_backend_cuda${CUDAToolkit_VERSION_MAJOR}x")

#   # Process gbgpu_utils_wrap.pyx into a C++ file
#   add_custom_command(
#     OUTPUT "${cythonized_file}"
#     COMMENT "Cythonize gbgpu_utils_wrap.pyx into ${cythonized_file}"
#     COMMAND
#     Python::Interpreter -m cython "${CMAKE_CURRENT_SOURCE_DIR}/gbgpu_utils_wrap.pyx"
#     -3 -+
#     --output-file "${CMAKE_CURRENT_BINARY_DIR}/${cythonized_file}"
#     --module-name "${module_name}"
#     -I "${CMAKE_CURRENT_SOURCE_DIR}"
#     DEPENDS "gbgpu_utils_wrap.pyx"
#     VERBATIM)

#   # Declare the python compiled module
#   python_add_library(
#     "${module_name}"
#     MODULE WITH_SOABI "${cythonized_file}" gbgpu_utils.cu)

#   # Compilation options
#   target_include_directories(
#     "${module_name}"
#     PRIVATE ${Python_NumPy_INCLUDE_DIR})

#   target_compile_definitions(
#     "${module_name}"
#     PRIVATE NPY_NO_DEPRECATED_API=NPY_1_9_API_VERSION)

#   target_sources(
#     "${module_name}" PUBLIC FILE_SET HEADERS
#     FILES cuda_complex.hpp global.h gbgpu_utils.hh)

#   # Explicitly require dynamic linking of cudart
#   set_property(
#     TARGET "${module_name}"
#     PROPERTY CUDA_RUNTIME_LIBRARY "Shared")

#   set_property(
#     TARGET "${module_name}"
#     PROPERTY CUDA_ARCHITECTURES ${GBGPU_CUDA_ARCH})

#   # Installation options
#   install(TARGETS "${module_name}" DESTINATION ${backend_name})


