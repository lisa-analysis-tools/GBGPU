// GBGPU pybind11 module entry.
//
// Sibling of BBHx's binding_bbhx.cxx -- built fresh against the
// post-Phase-3L LAT setup (GBGPU had no prior pybind work to pull
// forward; BBHx's came from origin/pybind).

#include "binding_gbgpu.hpp"

// Single-registrant rule: this TU must not be marked as the wrapper owner.
// LAT's binding.cxx sets the toggle to 1; every downstream binding TU
// leaves it at 0 and asserts via this:
static_assert(!LISATOOLS_IS_WRAPPER_OWNER,
    "Single-registrant rule: only LISAanalysistools may register "
    "OrbitsWrap / TDIConfigWrap / LISAResponseWrap / WDM*/FD*/Spline* "
    "with pybind11. GBGPU's cgbgpu module is for GB-specific wrappers "
    "only (plus the GBTDIonTheFly + GBComputationGroup carve-out at "
    "Phase 3L.7). See plan it-is-time-to-delegated-peach.md "
    "and the existing pattern in "
    "lisa-on-gpu/src/fastlisaresponse/cutils/binding_tof.cxx.");

void gbgpu_part(py::module &m) {
#if defined(__CUDA_COMPILATION__) || defined(__CUDACC__)
    py::class_<GBGPUComputationWrap>(m, "GBGPUComputationWrapGPU")
#else
    py::class_<GBGPUComputationWrap>(m, "GBGPUComputationWrapCPU")
#endif
        .def(py::init<>())
        // gbgpu_utils.hh (migrated from gbgpu_utils_wrap.pyx)
        .def("get_ll", &GBGPUComputationWrap::get_ll,
             "Compute <d|h> and <h|h> per-binary.")
        .def("fill_global", &GBGPUComputationWrap::fill_global,
             "Sum per-binary templates into global A/E data buffers.")
        .def("swap_ll_diff", &GBGPUComputationWrap::swap_ll_diff,
             "Swap-likelihood per-binary differences.")
        // SharedMemoryGBGPU.hpp (migrated from sharedmemgbgpu.pyx)
        .def("SharedMemoryWaveComp_wrap",
             &GBGPUComputationWrap::SharedMemoryWaveComp_wrap,
             "Shared-memory GB waveform generation.")
        .def("SharedMemoryLikeComp_wrap",
             &GBGPUComputationWrap::SharedMemoryLikeComp_wrap,
             "Shared-memory GB likelihood evaluation.")
        .def("SharedMemorySwapLikeComp_wrap",
             &GBGPUComputationWrap::SharedMemorySwapLikeComp_wrap,
             "Shared-memory GB swap-likelihood evaluation.")
        .def("SharedMemoryChiSquaredComp_wrap",
             &GBGPUComputationWrap::SharedMemoryChiSquaredComp_wrap,
             "Shared-memory GB chi-squared evaluation.")
        .def("SharedMemoryGenerateGlobal_wrap",
             &GBGPUComputationWrap::SharedMemoryGenerateGlobal_wrap,
             "Shared-memory GB global-template generation.")
        .def("SharedMemoryFstatLikeComp_wrap",
             &GBGPUComputationWrap::SharedMemoryFstatLikeComp_wrap,
             "Shared-memory GB F-statistic likelihood evaluation.")
        ;
}


PYBIND11_MODULE(cgbgpu, m) {
    m.doc() = "GBGPU pybind11 backend (skeleton). "
              "GB-specific waveform + utils wrappers will migrate into "
              "this module from their respective Cython submodules; "
              "GBTDIonTheFly + GBComputationGroup + gb_wdm_het_* lands "
              "here at Phase 3L.7.";
    gbgpu_part(m);
}
