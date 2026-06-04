#ifndef __BINDING_GBGPU_HPP__
#define __BINDING_GBGPU_HPP__

// GBGPU pybind11 module surface.
//
// Sibling of BBHx's binding_bbhx.{hpp,cxx} -- BBHx had its skeleton on
// origin/pybind (commits 7c383e3 / 0543508 / bd9e127); GBGPU had no
// prior pybind work, so this file is built fresh against the post-
// Phase-3L LAT setup.
//
// LAT is the sole pybind11 registrant of the shared wrapper family
// (OrbitsWrap_responselisa, TDIConfigWrap, LISAResponseWrap, the
// LISATDIonTheFly base, and FD/WDM/Spline Wraps). This TU consumes
// those via `#include "binding_flr.hpp"`. The single-registrant rule
// is enforced via `static_assert(!LISATOOLS_IS_WRAPPER_OWNER, ...)` in
// binding_gbgpu.cxx -- GBGPU never re-registers any LAT-owned class.
//
// GBGPU's cgbgpu module is reserved for GB-specific wrappers
// (SharedMemoryGBGPU + gbgpu_utils as they migrate from the existing
// Cython modules) and -- once Phase 3L.7 lands -- GBTDIonTheFly +
// GBComputationGroup + the gb_wdm_het_* chunked-heterodyne kernels
// + the gb_signal_het_* polyphase signal-heterodyne kernels from the
// in-flight v2 work.

// GBGPU-specific waveform/utils headers are deliberately NOT included
// by the skeleton -- BBHxComputationWrap-equivalent GBGPUComputationWrap
// has no method bodies yet. As each Cython module migrates here, add
// the corresponding GBGPU header (gbgpu_utils.hh, SharedMemoryGBGPU.hpp,
// ...) below; for CubicSpline use GBT's `InterpolateDevice.hh` rather
// than any local Interpolate.hh (mirrors the BBHx note for the same
// reason: include-guard collisions can hide the CubicSpline definition
// binding_flr.hpp needs).

// LAT-canonical pybind11 base + array typedefs + Orbits + TDIConfig wrappers.
// binding_flr.hpp provides ReturnPointerBase and array_type<T>; consuming TUs
// MUST leave LISATOOLS_IS_WRAPPER_OWNER at its default (0) to satisfy the
// per-TU static_assert (GBGPU never re-registers OrbitsWrap_responselisa et al).
#include "lisatools_header_abi.hpp"
#include "binding_flr.hpp"

#include <string>
#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace py = pybind11;

#if defined(__CUDA_COMPILATION__) || defined(__CUDACC__)
#define GBGPUComputationWrap GBGPUComputationWrapGPU
#else
#define GBGPUComputationWrap GBGPUComputationWrapCPU
#endif

// Skeleton wrapper. Phase GBGPU.pybind ships the empty class + the
// pybind11 module entrypoint; subsequent commits populate
// GBGPUComputationWrap with method wrappers as each Cython module
// (gbgpu_utils_wrap.pyx, sharedmemgbgpu.pyx) is migrated to pybind11.
// Phase 3L.7 lands GBTDIonTheFly + GBComputationGroup + gb_wdm_het_*
// kernels here too.
//
// Inherits from ReturnPointerBase so future method wrappers can use
// return_pointer_and_check_length / return_pointer with the LAT-side
// implementations.
class GBGPUComputationWrap : public ReturnPointerBase {
  public:
    GBGPUComputationWrap() = default;
    ~GBGPUComputationWrap() = default;

    // Method wrappers are intentionally omitted in the skeleton commit.
    // When migrating a Cython module (e.g. gbgpu_utils_wrap.pyx) here,
    // add a class member that adapts the array_type<T> -> raw pointer
    // for each existing free function. Phase 3L.7 carve-out adds
    // GB-specific TDI/likelihood/chunked-het method wrappers.
};

// Module entry called from PYBIND11_MODULE(cgbgpu, m) in binding_gbgpu.cxx.
void gbgpu_part(py::module &m);

#endif // __BINDING_GBGPU_HPP__
