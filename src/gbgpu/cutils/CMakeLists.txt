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
# common helpers that the downloaded LISAanalysistools Detector.cu pulls in.
# Mirrors how lisa-on-gpu/src/fastlisaresponse/cutils/CMakeLists.txt does it.
execute_process(
  COMMAND "${Python_EXECUTABLE}" -c "if True:
    import gpubackendtools
    print(gpubackendtools.__file__.split('__init__.py')[0] + 'cutils')"
  OUTPUT_VARIABLE GBT_CUTILS
  OUTPUT_STRIP_TRAILING_WHITESPACE)

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

# 0. Download lisa analysis tools c files.
file(
  DOWNLOAD 
  https://raw.githubusercontent.com/mikekatz04/LISAanalysistools/refs/heads/main/src/lisatools/cutils/Detector.cu 
  "./src/gbgpu/cutils/Detector.cu"
)
file(
  DOWNLOAD 
  https://raw.githubusercontent.com/mikekatz04/LISAanalysistools/refs/heads/main/src/lisatools/cutils/Detector.hpp 
  "./src/gbgpu/cutils/Detector.hpp"
)
file(
  DOWNLOAD 
  https://raw.githubusercontent.com/mikekatz04/LISAanalysistools/refs/heads/main/src/lisatools/cutils/global.hpp 
  "./src/gbgpu/cutils/global.hpp"
)
# I. Process gbgpu_utils_wrap.pyx into a C++ file
add_custom_command(
  OUTPUT "utils.cxx"
  COMMENT "Cythonize gbgpu_utils_wrap.pyx into utils.cxx"
  COMMAND
    Python::Interpreter -m cython "${CMAKE_CURRENT_SOURCE_DIR}/gbgpu_utils_wrap.pyx"
    --output-file "${CMAKE_CURRENT_BINARY_DIR}/utils.cxx" -3 -+ --module-name
    "utils" -I "${CMAKE_CURRENT_SOURCE_DIR}"
  DEPENDS "gbgpu_utils_wrap.pyx"
  VERBATIM)

add_custom_command(
  OUTPUT "sharedmemgbgpu.cxx"
  COMMENT "Cythonize sharedmemgbgpu.pyx into sharedmemgbgpu.cxx"
  COMMAND
    Python::Interpreter -m cython "${CMAKE_CURRENT_SOURCE_DIR}/sharedmemgbgpu.pyx"
    --output-file "${CMAKE_CURRENT_BINARY_DIR}/sharedmemgbgpu.cxx" -3 -+ --module-name
    "sharedmem" -I "${CMAKE_CURRENT_SOURCE_DIR}"
  DEPENDS "sharedmemgbgpu.pyx"
  VERBATIM)

# II. Declare the CPU backend
if(GBGPU_WITH_CPU)
  add_custom_command(
    OUTPUT "gbgpu_utils.cxx"
    COMMENT "Copy gbgpu_utils.cu to gbgpu_utils.cxx"
    COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/gbgpu_utils.cu"
            "${CMAKE_CURRENT_BINARY_DIR}/gbgpu_utils.cxx"
    DEPENDS "gbgpu_utils.cu"
    VERBATIM)

  add_custom_command(
    OUTPUT "Detector.cxx"
    COMMENT "Copy Detector.cu to Detector.cxx"
    COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/Detector.cu"
            "${CMAKE_CURRENT_BINARY_DIR}/Detector.cxx"
    DEPENDS "Detector.cu"
    VERBATIM)

  add_custom_command(
    OUTPUT "SharedMemoryGBGPU.cxx"
    COMMENT "Copy SharedMemoryGBGPU.cu to SharedMemoryGBGPU.cxx"
    COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/SharedMemoryGBGPU.cu"
            "${CMAKE_CURRENT_BINARY_DIR}/SharedMemoryGBGPU.cxx"
    DEPENDS "SharedMemoryGBGPU.cu"
    VERBATIM)


  python_add_library(gbgpu_cpu_utils MODULE WITH_SOABI Detector.cxx utils.cxx gbgpu_utils.cxx)
  apply_cpu_backend_common_options(utils)
  # Detector.cu (downloaded from LISAanalysistools) includes gbt_global.h from
  # gpubackendtools; make that path visible to the CPU build.
  target_include_directories(gbgpu_cpu_utils PUBLIC ${GBT_CUTILS})

  target_sources(gbgpu_cpu_utils PUBLIC FILE_SET HEADERS FILES
  global.hpp Detector.hpp cuda_complex.hpp global.h gbgpu_utils.hh)


  python_add_library(gbgpu_cpu_sharedmem MODULE WITH_SOABI Detector.cxx SharedMemoryGBGPU.cxx sharedmemgbgpu.cxx)
  apply_cpu_backend_common_options(sharedmem)
  target_include_directories(gbgpu_cpu_sharedmem PUBLIC ${GBT_CUTILS})
  target_sources(gbgpu_cpu_sharedmem PUBLIC FILE_SET HEADERS FILES
  global.hpp Detector.hpp cuda_complex.hpp global.h SharedMemoryGBGPU.hpp
  common.hpp fp16_common.hpp common.hpp)
  install(TARGETS gbgpu_cpu_sharedmem gbgpu_cpu_utils
          LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}/python${PYTHON_VERSION_MAJOR}.${PYTHON_VERSION_MINOR}/site-packages/gbgpu)


endif()


# III. Declare the GPU backend
if(GBGPU_WITH_GPU)
  python_add_library(gbgpu_gpu_utils MODULE WITH_SOABI utils.cxx gbgpu_utils.cu)
  apply_gpu_backend_common_options(utils)
  target_sources(gbgpu_gpu_utils PUBLIC FILE_SET HEADERS FILES
                                         cuda_complex.hpp global.h gbgpu_utils.hh)

  
  # Add an include directory for a specific target
  # Find a library
  python_add_library(gbgpu_gpu_sharedmem MODULE WITH_SOABI Detector.cu SharedMemoryGBGPU.cu sharedmemgbgpu.cxx)
  apply_gpu_backend_common_options(sharedmem)
  message(STATUS "CUDA ARCHS HERE: ${GBGPU_CUDA_ARCH}")
  set_property(TARGET gbgpu_gpu_sharedmem PROPERTY CUDA_ARCHITECTURES ${GBGPU_CUDA_ARCH})
  set_property(TARGET gbgpu_gpu_sharedmem PROPERTY CUDA_SEPARABLE_COMPILATION ON)
  set_property(TARGET gbgpu_gpu_sharedmem PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS ON)
  set_property(TARGET gbgpu_gpu_sharedmem PROPERTY POSITION_INDEPENDENT_CODE ON)

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
  target_sources(gbgpu_gpu_sharedmem PUBLIC FILE_SET HEADERS FILES
  global.hpp Detector.hpp cuda_complex.hpp global.h SharedMemoryGBGPU.hpp
  common.hpp fp16_common.hpp common.hpp)
  target_link_libraries(gbgpu_gpu_sharedmem PUBLIC CUDA::cudart CUDA::cublas
      CUDA::cusparse mathdx::cufftdx)
  message(STATUS "CMAKE_INSTALL_LIBDIR HERE: ${CMAKE_INSTALL_LIBDIR}")
  install(TARGETS gbgpu_gpu_sharedmem
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


