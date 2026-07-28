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
// (OrbitsWrap, TDIConfigWrap, LISAResponseWrap, the
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

// GBGPU-specific waveform/utils headers, included so the
// GBGPUComputationWrap method bodies can call out to the underlying free
// functions.
#include "gbgpu_utils.hh"        // fill_global_wrap, get_ll_wrap, swap_ll_diff_wrap
#ifndef GBGPU_NO_SHAREDMEM
#include "SharedMemoryGBGPU.hpp" // SharedMemoryWaveComp, SharedMemoryLikeComp,
                                 // SharedMemorySwapLikeComp,
                                 // SharedMemoryChiSquaredComp,
                                 // SharedMemoryGenerateGlobal,
                                 // SharedMemoryFstatLikeComp
#endif

// GBGPU_NO_SHAREDMEM is defined by the build when -DGBGPU_WITH_SHAREDMEM=OFF
// (see the project-root CMakeLists.txt). SharedMemoryGBGPU.cu is then not
// compiled at all -- which also removes the build's only cuFFTDx/mathdx
// dependency. The seven ``SharedMemory*_wrap`` methods below stay BOUND so the
// Python surface is byte-identical; each raises RuntimeError instead of
// calling into the (absent) kernel. Keep the message actionable: the caller
// needs a rebuild, not a code change.
#ifdef GBGPU_NO_SHAREDMEM
#define GBGPU_SHAREDMEM_UNAVAILABLE(fn)                                       \
    throw std::runtime_error(                                                 \
        "[" fn "] this GBGPU backend was built with "                         \
        "-DGBGPU_WITH_SHAREDMEM=OFF, so the legacy SharedMemoryGBGPU kernels " \
        "are not present. Rebuild GBGPU without that flag (the default) to "  \
        "use the legacy GB path (get_fstat_ll / generate_global_template / "  \
        "run_wave / swap_likelihood_difference / information_matrix). The "   \
        "GBTDIonTheFly, chunked-heterodyne and signal-heterodyne kernels are " \
        "unaffected by this flag.")
#endif

// Phase 3L.7g (2026-06-04): GB TDIonTheFly + GBComputationGroup
// arrived from lisa-on-gpu. The class declarations + kernel host wrappers
// live in gb_tdi_on_the_fly.hh (also in this cutils dir).
#include "gb_tdi_on_the_fly.hh"

// LAT-canonical pybind11 base + array typedefs + Orbits + TDIConfig wrappers.
// binding_flr.hpp provides ReturnPointerBase and array_type<T>; consuming TUs
// MUST leave LISATOOLS_IS_WRAPPER_OWNER at its default (0) to satisfy the
// per-TU static_assert (GBGPU never re-registers OrbitsWrap et al).
#include "lisatools_header_abi.hpp"
#include "binding_flr.hpp"
// Phase 3L.7g (2026-06-04): pybind11 Wrap base + dependent Wraps for the
// new GBTDIonTheFlyWrap + GBComputationGroupWrap classes. WDMSettingsWrap +
// FDDomainWrap + LISATDIonTheFlyWrap live in LAT (moved at Phase 3L.1-3L.6).
#include "binding_wdm_settings.hpp"   // WDMSettingsWrap
#include "binding_fd_domain.hpp"      // FDDomainWrap
#include "binding_lat_spline_tdi.hpp" // LISATDIonTheFlyWrap (parent of GBTDIonTheFlyWrap)

#include <string>
#include <stdexcept>
#include <iostream>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>

namespace nb = nanobind;

#if defined(__CUDA_COMPILATION__) || defined(__CUDACC__)
#define GBGPUComputationWrap GBGPUComputationWrapGPU
// Phase 3L.7g aliases: shipped from lisa-on-gpu's binding_tof.hpp.
#define GBTDIonTheFlyWrap GBTDIonTheFlyWrapGPU
#define GBComputationGroupWrap GBComputationGroupWrapGPU
#else
#define GBGPUComputationWrap GBGPUComputationWrapCPU
#define GBTDIonTheFlyWrap GBTDIonTheFlyWrapCPU
#define GBComputationGroupWrap GBComputationGroupWrapCPU
#endif

// Unified GBGPU pybind11 wrapper. Inherits from ReturnPointerBase so
// every method body can use return_pointer / return_pointer_and_check_length
// (LAT-canonical implementations) to adapt array_type<T> -> raw pointer.
//
// Methods migrate in here one Cython module at a time. Already migrated
// (Phase GBGPU.pybind.bulk):
// - gbgpu_utils_wrap.pyx -> get_ll, fill_global, swap_ll_diff
// - sharedmemgbgpu.pyx   -> SharedMemoryWaveComp_wrap, SharedMemoryLikeComp_wrap,
//                            SharedMemorySwapLikeComp_wrap,
//                            SharedMemoryChiSquaredComp_wrap,
//                            SharedMemoryGenerateGlobal_wrap,
//                            SharedMemoryFstatLikeComp_wrap
//
// Phase 3L.7 lands GBTDIonTheFly + GBComputationGroup + gb_wdm_het_*
// chunked-heterodyne kernels here too.
class GBGPUComputationWrap : public ReturnPointerBase {
  public:
    GBGPUComputationWrap() = default;
    ~GBGPUComputationWrap() = default;

    // ---- gbgpu_utils.hh wrappers (migrated from gbgpu_utils_wrap.pyx) ----
    //
    // NOTE: the original Cython binding had no length checks (`wrapper()`
    // just extracted size_t pointers); the migrated wrappers below preserve
    // that behavior with `return_pointer` (no `_and_check_length` variant).
    // Tightening to checked-length variants is a follow-up.

    void get_ll(
        array_type<std::complex<double>> d_h,
        array_type<std::complex<double>> h_h,
        array_type<std::complex<double>> A_template,
        array_type<std::complex<double>> E_template,
        array_type<std::complex<double>> A_data,
        array_type<std::complex<double>> E_data,
        array_type<double> A_psd, array_type<double> E_psd, double df,
        array_type<int> start_ind, int M, int num_bin,
        array_type<int> data_index, array_type<int> noise_index,
        int data_length)
    {
        get_ll_wrap(
            (cmplx*) return_pointer(d_h,          "d_h"),
            (cmplx*) return_pointer(h_h,          "h_h"),
            (cmplx*) return_pointer(A_template,   "A_template"),
            (cmplx*) return_pointer(E_template,   "E_template"),
            (cmplx*) return_pointer(A_data,       "A_data"),
            (cmplx*) return_pointer(E_data,       "E_data"),
            return_pointer(A_psd,        "A_psd"),
            return_pointer(E_psd,        "E_psd"),
            df,
            return_pointer(start_ind,    "start_ind"),
            M, num_bin,
            return_pointer(data_index,   "data_index"),
            return_pointer(noise_index,  "noise_index"),
            data_length);
    }

    void fill_global(
        array_type<std::complex<double>> A_glob,
        array_type<std::complex<double>> E_glob,
        array_type<std::complex<double>> A_template,
        array_type<std::complex<double>> E_template,
        array_type<int> start_ind, int M, int num_bin,
        array_type<int> group_index, int data_length)
    {
        fill_global_wrap(
            (cmplx*) return_pointer(A_glob,     "A_glob"),
            (cmplx*) return_pointer(E_glob,     "E_glob"),
            (cmplx*) return_pointer(A_template, "A_template"),
            (cmplx*) return_pointer(E_template, "E_template"),
            return_pointer(start_ind,   "start_ind"),
            M, num_bin,
            return_pointer(group_index, "group_index"),
            data_length);
    }

    void swap_ll_diff(
        array_type<std::complex<double>> d_h_remove,
        array_type<std::complex<double>> d_h_add,
        array_type<std::complex<double>> add_remove,
        array_type<std::complex<double>> remove_remove,
        array_type<std::complex<double>> add_add,
        array_type<std::complex<double>> A_remove,
        array_type<std::complex<double>> E_remove,
        array_type<int> start_ind_all_remove,
        array_type<std::complex<double>> A_add,
        array_type<std::complex<double>> E_add,
        array_type<int> start_ind_all_add,
        array_type<std::complex<double>> A_data,
        array_type<std::complex<double>> E_data,
        array_type<double> A_psd, array_type<double> E_psd,
        double df, int M, int num_bin,
        array_type<int> data_index, array_type<int> noise_index,
        int data_length)
    {
        swap_ll_diff_wrap(
            (cmplx*) return_pointer(d_h_remove,    "d_h_remove"),
            (cmplx*) return_pointer(d_h_add,       "d_h_add"),
            (cmplx*) return_pointer(add_remove,    "add_remove"),
            (cmplx*) return_pointer(remove_remove, "remove_remove"),
            (cmplx*) return_pointer(add_add,       "add_add"),
            (cmplx*) return_pointer(A_remove,      "A_remove"),
            (cmplx*) return_pointer(E_remove,      "E_remove"),
            return_pointer(start_ind_all_remove, "start_ind_all_remove"),
            (cmplx*) return_pointer(A_add,         "A_add"),
            (cmplx*) return_pointer(E_add,         "E_add"),
            return_pointer(start_ind_all_add,    "start_ind_all_add"),
            (cmplx*) return_pointer(A_data,        "A_data"),
            (cmplx*) return_pointer(E_data,        "E_data"),
            return_pointer(A_psd, "A_psd"), return_pointer(E_psd, "E_psd"),
            df, M, num_bin,
            return_pointer(data_index,  "data_index"),
            return_pointer(noise_index, "noise_index"),
            data_length);
    }

    // ---- SharedMemoryGBGPU.hpp wrappers (migrated from sharedmemgbgpu.pyx) ----

    void SharedMemoryWaveComp_wrap(
        array_type<std::complex<double>> tdi_out,
        array_type<int> start_inds_out,
        array_type<double> amp, array_type<double> f0,
        array_type<double> fdot0, array_type<double> fddot0,
        array_type<double> phi0, array_type<double> iota,
        array_type<double> psi, array_type<double> lam,
        array_type<double> theta,
        double T, double dt, int N, int num_bin_all, int tdi_channel_setup,
        array_type<double> Ps, double L_arm, bool tdi2,
        int window_type, double window_alpha)
    {
#ifdef GBGPU_NO_SHAREDMEM
        GBGPU_SHAREDMEM_UNAVAILABLE("SharedMemoryWaveComp_wrap");
#else
        SharedMemoryWaveComp(
            (cmplx*) return_pointer(tdi_out,        "tdi_out"),
            return_pointer(start_inds_out, "start_inds_out"),
            return_pointer(amp,    "amp"),
            return_pointer(f0,     "f0"),
            return_pointer(fdot0,  "fdot0"),
            return_pointer(fddot0, "fddot0"),
            return_pointer(phi0,   "phi0"),
            return_pointer(iota,   "iota"),
            return_pointer(psi,    "psi"),
            return_pointer(lam,    "lam"),
            return_pointer(theta,  "theta"),
            T, dt, N, num_bin_all, tdi_channel_setup,
            return_pointer(Ps, "Ps"), L_arm, tdi2,
            window_type, window_alpha);
#endif
    }

    void SharedMemoryLikeComp_wrap(
        array_type<std::complex<double>> d_h,
        array_type<std::complex<double>> h_h,
        array_type<std::complex<double>> data,
        array_type<std::complex<double>> noise,
        array_type<int> data_index, array_type<int> noise_index,
        array_type<double> amp, array_type<double> f0,
        array_type<double> fdot0, array_type<double> fddot0,
        array_type<double> phi0, array_type<double> iota,
        array_type<double> psi, array_type<double> lam,
        array_type<double> theta,
        double T, double dt, int N, int num_bin_all,
        array_type<int> start_freq_inds,
        int data_length, int tdi_channel_setup,
        int device, bool do_synchronize,
        int num_data, int num_noise,
        array_type<double> Ps, double L_arm, bool tdi2,
        int window_type, double window_alpha)
    {
#ifdef GBGPU_NO_SHAREDMEM
        GBGPU_SHAREDMEM_UNAVAILABLE("SharedMemoryLikeComp_wrap");
#else
        SharedMemoryLikeComp(
            (cmplx*) return_pointer(d_h,  "d_h"),
            (cmplx*) return_pointer(h_h,  "h_h"),
            (cmplx*) return_pointer(data, "data"),
            (cmplx*) return_pointer(noise,        "noise"),
            return_pointer(data_index,   "data_index"),
            return_pointer(noise_index,  "noise_index"),
            return_pointer(amp,    "amp"),
            return_pointer(f0,     "f0"),
            return_pointer(fdot0,  "fdot0"),
            return_pointer(fddot0, "fddot0"),
            return_pointer(phi0,   "phi0"),
            return_pointer(iota,   "iota"),
            return_pointer(psi,    "psi"),
            return_pointer(lam,    "lam"),
            return_pointer(theta,  "theta"),
            T, dt, N, num_bin_all,
            return_pointer(start_freq_inds, "start_freq_inds"),
            data_length, tdi_channel_setup,
            device, do_synchronize,
            num_data, num_noise,
            return_pointer(Ps, "Ps"), L_arm, tdi2,
            window_type, window_alpha);
#endif
    }

    void SharedMemorySwapLikeComp_wrap(
        array_type<std::complex<double>> d_h_remove,
        array_type<std::complex<double>> d_h_add,
        array_type<std::complex<double>> remove_remove,
        array_type<std::complex<double>> add_add,
        array_type<std::complex<double>> add_remove,
        array_type<std::complex<double>> data,
        array_type<std::complex<double>> noise,
        array_type<int> data_index, array_type<int> noise_index,
        array_type<double> amp_add, array_type<double> f0_add,
        array_type<double> fdot0_add, array_type<double> fddot0_add,
        array_type<double> phi0_add, array_type<double> iota_add,
        array_type<double> psi_add, array_type<double> lam_add,
        array_type<double> theta_add,
        array_type<double> amp_remove, array_type<double> f0_remove,
        array_type<double> fdot0_remove, array_type<double> fddot0_remove,
        array_type<double> phi0_remove, array_type<double> iota_remove,
        array_type<double> psi_remove, array_type<double> lam_remove,
        array_type<double> theta_remove,
        double T, double dt, int N, int num_bin_all,
        array_type<int> start_freq_inds,
        int data_length, int tdi_channel_setup,
        int device, bool do_synchronize,
        int num_data, int num_noise,
        array_type<double> Ps, double L_arm, bool tdi2,
        int window_type, double window_alpha)
    {
#ifdef GBGPU_NO_SHAREDMEM
        GBGPU_SHAREDMEM_UNAVAILABLE("SharedMemorySwapLikeComp_wrap");
#else
        SharedMemorySwapLikeComp(
            (cmplx*) return_pointer(d_h_remove,    "d_h_remove"),
            (cmplx*) return_pointer(d_h_add,       "d_h_add"),
            (cmplx*) return_pointer(remove_remove, "remove_remove"),
            (cmplx*) return_pointer(add_add,       "add_add"),
            (cmplx*) return_pointer(add_remove,    "add_remove"),
            (cmplx*) return_pointer(data,          "data"),
            (cmplx*) return_pointer(noise,        "noise"),
            return_pointer(data_index,   "data_index"),
            return_pointer(noise_index,  "noise_index"),
            return_pointer(amp_add,    "amp_add"),
            return_pointer(f0_add,     "f0_add"),
            return_pointer(fdot0_add,  "fdot0_add"),
            return_pointer(fddot0_add, "fddot0_add"),
            return_pointer(phi0_add,   "phi0_add"),
            return_pointer(iota_add,   "iota_add"),
            return_pointer(psi_add,    "psi_add"),
            return_pointer(lam_add,    "lam_add"),
            return_pointer(theta_add,  "theta_add"),
            return_pointer(amp_remove,    "amp_remove"),
            return_pointer(f0_remove,     "f0_remove"),
            return_pointer(fdot0_remove,  "fdot0_remove"),
            return_pointer(fddot0_remove, "fddot0_remove"),
            return_pointer(phi0_remove,   "phi0_remove"),
            return_pointer(iota_remove,   "iota_remove"),
            return_pointer(psi_remove,    "psi_remove"),
            return_pointer(lam_remove,    "lam_remove"),
            return_pointer(theta_remove,  "theta_remove"),
            T, dt, N, num_bin_all,
            return_pointer(start_freq_inds, "start_freq_inds"),
            data_length, tdi_channel_setup,
            device, do_synchronize,
            num_data, num_noise,
            return_pointer(Ps, "Ps"), L_arm, tdi2,
            window_type, window_alpha);
#endif
    }

    void SharedMemoryChiSquaredComp_wrap(
        array_type<std::complex<double>> h1_h1,
        array_type<std::complex<double>> h2_h2,
        array_type<std::complex<double>> h1_h2,
        array_type<std::complex<double>> noise,
        array_type<int> noise_index,
        array_type<double> amp, array_type<double> f0,
        array_type<double> fdot0, array_type<double> fddot0,
        array_type<double> phi0, array_type<double> iota,
        array_type<double> psi, array_type<double> lam,
        array_type<double> theta,
        double T, double dt, int N, int num_bin_all,
        array_type<int> start_freq_inds,
        int data_length, int tdi_channel_setup,
        int device, bool do_synchronize,
        int num_data, int num_noise,
        array_type<double> Ps, double L_arm, bool tdi2,
        int window_type, double window_alpha)
    {
#ifdef GBGPU_NO_SHAREDMEM
        GBGPU_SHAREDMEM_UNAVAILABLE("SharedMemoryChiSquaredComp_wrap");
#else
        SharedMemoryChiSquaredComp(
            (cmplx*) return_pointer(h1_h1, "h1_h1"),
            (cmplx*) return_pointer(h2_h2, "h2_h2"),
            (cmplx*) return_pointer(h1_h2, "h1_h2"),
            (cmplx*) return_pointer(noise,        "noise"),
            return_pointer(noise_index,  "noise_index"),
            return_pointer(amp,    "amp"),
            return_pointer(f0,     "f0"),
            return_pointer(fdot0,  "fdot0"),
            return_pointer(fddot0, "fddot0"),
            return_pointer(phi0,   "phi0"),
            return_pointer(iota,   "iota"),
            return_pointer(psi,    "psi"),
            return_pointer(lam,    "lam"),
            return_pointer(theta,  "theta"),
            T, dt, N, num_bin_all,
            return_pointer(start_freq_inds, "start_freq_inds"),
            data_length, tdi_channel_setup,
            device, do_synchronize,
            num_data, num_noise,
            return_pointer(Ps, "Ps"), L_arm, tdi2,
            window_type, window_alpha);
#endif
    }

    void SharedMemoryGenerateGlobal_wrap(
        array_type<std::complex<double>> data,
        array_type<int> data_index,
        array_type<double> factors,
        array_type<double> amp, array_type<double> f0,
        array_type<double> fdot0, array_type<double> fddot0,
        array_type<double> phi0, array_type<double> iota,
        array_type<double> psi, array_type<double> lam,
        array_type<double> theta,
        double T, double dt, int N, int num_bin_all,
        array_type<int> start_freq_inds,
        int data_length, int tdi_channel_setup,
        int device, bool do_synchronize,
        array_type<double> Ps, double L_arm, bool tdi2,
        int window_type, double window_alpha)
    {
#ifdef GBGPU_NO_SHAREDMEM
        GBGPU_SHAREDMEM_UNAVAILABLE("SharedMemoryGenerateGlobal_wrap");
#else
        SharedMemoryGenerateGlobal(
            (cmplx*) return_pointer(data, "data"),
            return_pointer(data_index, "data_index"),
            return_pointer(factors,    "factors"),
            return_pointer(amp,    "amp"),
            return_pointer(f0,     "f0"),
            return_pointer(fdot0,  "fdot0"),
            return_pointer(fddot0, "fddot0"),
            return_pointer(phi0,   "phi0"),
            return_pointer(iota,   "iota"),
            return_pointer(psi,    "psi"),
            return_pointer(lam,    "lam"),
            return_pointer(theta,  "theta"),
            T, dt, N, num_bin_all,
            return_pointer(start_freq_inds, "start_freq_inds"),
            data_length, tdi_channel_setup,
            device, do_synchronize,
            return_pointer(Ps, "Ps"), L_arm, tdi2,
            window_type, window_alpha);
#endif
    }

    void SharedMemoryFstatLikeComp_wrap(
        array_type<std::complex<double>> M_mat,
        array_type<std::complex<double>> N_arr,
        array_type<std::complex<double>> data,
        array_type<std::complex<double>> noise,
        array_type<int> data_index, array_type<int> noise_index,
        array_type<double> f0, array_type<double> fdot0,
        array_type<double> fddot0,
        array_type<double> lam, array_type<double> theta,
        double T, double dt, int N, int num_bin_all,
        array_type<int> start_freq_inds,
        int data_length, int tdi_channel_setup,
        int device, bool do_synchronize,
        int num_data, int num_noise,
        array_type<double> Ps, double L_arm, bool tdi2,
        int window_type, double window_alpha)
    {
#ifdef GBGPU_NO_SHAREDMEM
        GBGPU_SHAREDMEM_UNAVAILABLE("SharedMemoryFstatLikeComp_wrap");
#else
        SharedMemoryFstatLikeComp(
            (cmplx*) return_pointer(M_mat, "M_mat"),
            (cmplx*) return_pointer(N_arr, "N_arr"),
            (cmplx*) return_pointer(data,  "data"),
            (cmplx*) return_pointer(noise,        "noise"),
            return_pointer(data_index,   "data_index"),
            return_pointer(noise_index,  "noise_index"),
            return_pointer(f0,     "f0"),
            return_pointer(fdot0,  "fdot0"),
            return_pointer(fddot0, "fddot0"),
            return_pointer(lam,    "lam"),
            return_pointer(theta,  "theta"),
            T, dt, N, num_bin_all,
            return_pointer(start_freq_inds, "start_freq_inds"),
            data_length, tdi_channel_setup,
            device, do_synchronize,
            num_data, num_noise,
            return_pointer(Ps, "Ps"), L_arm, tdi2,
            window_type, window_alpha);
#endif
    }

    // mojito (2026-06 merge): Fisher / information-matrix kernel. Re-expressed
    // as nanobind from the mojito branch's sharedmemgbgpu.pyx
    // SharedMemoryInfoMatComp_wrap (the .pyx was retired at the nanobind
    // migration). Mirrors GBGPU.information_matrix's tuple_in ordering.
    void SharedMemoryInfoMatComp_wrap(
        array_type<double> info_mat,
        array_type<std::complex<double>> d_dh_workspace,
        array_type<std::complex<double>> noise,
        array_type<int> noise_index,
        array_type<int> inds,
        array_type<double> amp, array_type<double> f0,
        array_type<double> fdot0, array_type<double> fddot0,
        array_type<double> phi0, array_type<double> iota,
        array_type<double> psi, array_type<double> lam,
        array_type<double> theta,
        array_type<double> eps_scaled,
        double eps_orig, double T, double dt, int N,
        int num_bin_all, int num_derivs,
        array_type<int> start_freq_inds,
        int data_length, int tdi_channel_setup,
        int device, bool do_synchronize,
        int num_noise,
        array_type<double> Ps, double L_arm, bool tdi2,
        bool easy_central_difference,
        int window_type, double window_alpha)
    {
#ifdef GBGPU_NO_SHAREDMEM
        GBGPU_SHAREDMEM_UNAVAILABLE("SharedMemoryInfoMatComp_wrap");
#else
        SharedMemoryInfoMatComp(
            return_pointer(info_mat,                "info_mat"),
            (cmplx*) return_pointer(d_dh_workspace, "d_dh_workspace"),
            (cmplx*) return_pointer(noise,          "noise"),
            return_pointer(noise_index,  "noise_index"),
            return_pointer(inds,         "inds"),
            return_pointer(amp,    "amp"),
            return_pointer(f0,     "f0"),
            return_pointer(fdot0,  "fdot0"),
            return_pointer(fddot0, "fddot0"),
            return_pointer(phi0,   "phi0"),
            return_pointer(iota,   "iota"),
            return_pointer(psi,    "psi"),
            return_pointer(lam,    "lam"),
            return_pointer(theta,  "theta"),
            return_pointer(eps_scaled, "eps_scaled"),
            eps_orig, T, dt, N, num_bin_all, num_derivs,
            return_pointer(start_freq_inds, "start_freq_inds"),
            data_length, tdi_channel_setup,
            device, do_synchronize,
            num_noise,
            return_pointer(Ps, "Ps"), L_arm, tdi2,
            easy_central_difference,
            window_type, window_alpha);
#endif
    }
};

// ============================================================================
// Phase 3L.7g (2026-06-04): GB carve-out from lisa-on-gpu's binding_tof.hpp.
//
// `class GBTDIonTheFlyWrap` -- pybind11 wrapper around `GBTDIonTheFly`
//   (declared in gb_tdi_on_the_fly.hh). Inherits from LAT-owned
//   `LISATDIonTheFlyWrap` (binding_lat_spline_tdi.hpp). Methods:
//   - `run_wave_tdi_wrap` : time-domain TDI per binary (dense rfft path).
//   - `run_fd_wave_tdi_wrap` : sparse heterodyne FD TDI (v2 polyphase entry).
//   - `get_buffer_size` / `get_fd_buffer_size`.
//
// `class GBComputationGroupWrap` -- pybind11 wrapper around
//   `GBComputationGroup` (declared in gb_tdi_on_the_fly.hh). Hosts the
//   FD chunked-likelihood, WDM-heterodyne chunked-likelihood, and signal-
//   heterodyne (v2 polyphase) wraps.
//
// SOBBHTDIonTheFlyWrap + SOBBHComputationGroupWrap stay in lisa-on-gpu
// until Phase 3L.8 moves them to BBHx.
// ============================================================================

class GBTDIonTheFlyWrap : public LISATDIonTheFlyWrap {
  public:
    GBTDIonTheFly *waveform;
    double T;
    double t_ref;

    GBTDIonTheFlyWrap(OrbitsWrap *orbits_, TDIConfigWrap *tdi_config_, double T_, double t_ref_): LISATDIonTheFlyWrap(orbits_, tdi_config_)
    {
        T = T_;
        t_ref = t_ref_;
        waveform = new GBTDIonTheFly(orbits_->orbits, tdi_config_->tdi_config, T_, t_ref_);
    };
    ~GBTDIonTheFlyWrap(){
        delete waveform;
    };

    void run_wave_tdi_wrap(
        array_type<std::complex<double>>tdi_channels_arr,
        array_type<double>tdi_amp, array_type<double>tdi_phase, array_type<double>phi_ref,
        array_type<double>params, array_type<double>t_arr, int N, int num_bin, int n_params, int nchannels
    );

    int get_buffer_size(int N){return waveform->get_gb_buffer_size(N);};

    // Heterodyne FD GB on a sparse time grid. Builds the slow positive-freq
    // signal in shared memory, FFTs it, and returns (num_bin, nchannels,
    // N_sparse) complex doubles plus the per-source dense-bin index k_f0 and
    // snapped carrier frequency f0_grid.
    // tukey_alpha: scipy.signal.windows.tukey alpha applied to the slow
    // signal before the in-place sparse FFT. Pass 0.05 (matching the dense
    // rfft path) when chaining into the v2 polyphase signal-het consumer;
    // 0.0 = rectangular reproduces pre-2026-06-03 behavior.
    void run_fd_wave_tdi_wrap(
        array_type<std::complex<double>> X_het,
        array_type<int>    k_f0_out,
        array_type<double> f0_grid_out,
        array_type<double> params,
        double t_start, double Tobs,
        int N_sparse, int num_bin, int n_params, int nchannels,
        double tukey_alpha);

    int get_fd_buffer_size(int N_sparse, int nchannels){
        return waveform->get_gb_fd_buffer_size(N_sparse, nchannels);
    }
};


class GBComputationGroupWrap: public GBComputationGroup, public ReturnPointerBase {
  public:

    // ---- FD analogs ---------------------------------------------------
    void gb_fd_fill_global(
        array_type<std::complex<double>> template_fill,
        OrbitsWrap* orbits_wrap, TDIConfigWrap *tdi_config_wrap,
        FDDomainWrap *fd_wrap,
        array_type<double> params_all, array_type<int> data_index_all,
        array_type<double> factors_all,
        array_type<int> template_start_inds,
        int num_bin, int nparams, double T, double t_start, double t_ref,
        int N_sparse, int nchannels, double tukey_alpha, double edge_frac);

    void gb_fd_get_ll(
        array_type<double> d_h_out, array_type<double> h_h_out,
        OrbitsWrap* orbits_wrap, TDIConfigWrap *tdi_config_wrap,
        FDDomainWrap *fd_wrap,
        array_type<double> params_all,
        array_type<int> data_index_all, array_type<int> noise_index_all,
        int num_bin, int nparams, double T, double t_start, double t_ref,
        int N_sparse, int nchannels, int tdi_type, double tukey_alpha,
        double edge_frac);

    void gb_fd_swap_ll(
        array_type<double> d_h_add_out, array_type<double> d_h_remove_out,
        array_type<double> add_add_out, array_type<double> remove_remove_out,
        array_type<double> add_remove_out,
        OrbitsWrap* orbits_wrap, TDIConfigWrap *tdi_config_wrap,
        FDDomainWrap *fd_wrap,
        array_type<double> params_add_all, array_type<double> params_remove_all,
        array_type<int> data_index_all, array_type<int> noise_index_all,
        int num_bin, int nparams, double T, double t_start, double t_ref,
        int N_sparse, int nchannels, int tdi_type, double tukey_alpha,
        double edge_frac);

    // Chain-rule parameter gradients of gb_fd_get_ll / gb_fd_swap_ll.
    // param_eps[k] is the per-parameter central-FD step (length nparams);
    // pass eps_k <= 0 to freeze parameter k.
    void gb_fd_get_ll_grad(
        array_type<double> grad_out,
        OrbitsWrap* orbits_wrap, TDIConfigWrap *tdi_config_wrap,
        FDDomainWrap *fd_wrap,
        array_type<double> params_all,
        array_type<int> data_index_all, array_type<int> noise_index_all,
        array_type<double> param_eps,
        int num_bin, int nparams, double T, double t_start, double t_ref,
        int N_sparse, int nchannels, int tdi_type);

    void gb_fd_swap_ll_grad(
        array_type<double> grad_add_out, array_type<double> grad_remove_out,
        OrbitsWrap* orbits_wrap, TDIConfigWrap *tdi_config_wrap,
        FDDomainWrap *fd_wrap,
        array_type<double> params_add_all, array_type<double> params_remove_all,
        array_type<int> data_index_all, array_type<int> noise_index_all,
        array_type<double> param_eps_add, array_type<double> param_eps_remove,
        int num_bin, int nparams, double T, double t_start, double t_ref,
        int N_sparse, int nchannels, int tdi_type);

    // ---- Chunked-heterodyne path (no lookup table) ----------------------
    // Geometry arrays (chunk_t_starts, chunk_keep_lo, chunk_keep_hi,
    // chunk_n_global_offset, wdm_window) are precomputed on the Python side
    // by ``gb_wdm_het.compute_chunk_geometry`` / ``compute_wdm_window``.
    // ``grid_dim`` is the CUDA launch grid size (chosen via
    // ``chunked_het_grid_dim``); pass anything > 0 on CPU (ignored).
    void gb_wdm_het_fill_global(
        array_type<double> template_fill,
        OrbitsWrap *orbits_wrap, TDIConfigWrap *tdi_config_wrap,
        WDMSettingsWrap *wdm_settings_wrap,
        array_type<double> params_all, array_type<double> factors_all,
        array_type<int> data_index,
        array_type<double> chunk_t_starts,
        array_type<int> chunk_keep_lo, array_type<int> chunk_keep_hi,
        array_type<int> chunk_n_global_offset,
        array_type<double> wdm_window,
        int n_chunks, int num_bin, int nparams,
        int Nt_sub, int log2_Nt_sub,
        int N_sparse, int log2_N_sparse,
        int nchannels, int n_rfft_chunk,
        double T_chunk, double dt, double T, double t_ref,
        double tukey_alpha, int grid_dim, int N_cp_sig, int N_cp_orbit,
        int m_band_half_width, bool active_band,
        // task-b per-band slab (chunked_het.py always passes: 0 + empty = off).
        int Nf_slab, array_type<int> slab_min_f);

    void gb_wdm_het_get_ll(
        array_type<double> d_h_out, array_type<double> h_h_out,
        OrbitsWrap *orbits_wrap, TDIConfigWrap *tdi_config_wrap,
        WDMSettingsWrap *wdm_settings_wrap,
        array_type<double> params_all,
        array_type<int> data_index_all, array_type<int> noise_index_all,
        array_type<double> chunk_t_starts,
        array_type<int> chunk_keep_lo, array_type<int> chunk_keep_hi,
        array_type<int> chunk_n_global_offset,
        array_type<double> wdm_window,
        array_type<double> data_d, array_type<double> invC,
        int n_chunks, int num_bin, int nparams,
        int Nt_sub, int log2_Nt_sub,
        int N_sparse, int log2_N_sparse,
        int nchannels, int n_rfft_chunk,
        double T_chunk, double dt, double T, double t_ref, int tdi_type,
        double tukey_alpha, int grid_dim, int N_cp_sig, int N_cp_orbit,
        array_type<int> binary_perm, array_type<int> group_starts, array_type<int> group_ends,
        array_type<int> group_m_lo, array_type<int> group_m_hi, int n_groups,
        int m_band_half_width,
        // task-b per-band slab (chunked_het.py always passes: 0 + empty = off).
        int Nf_slab, array_type<int> slab_min_f);

    void gb_wdm_het_swap_ll(
        array_type<double> d_h_add_out, array_type<double> d_h_remove_out,
        array_type<double> add_add_out, array_type<double> remove_remove_out,
        array_type<double> add_remove_out,
        OrbitsWrap *orbits_wrap, TDIConfigWrap *tdi_config_wrap,
        WDMSettingsWrap *wdm_settings_wrap,
        array_type<double> params_add_all, array_type<double> params_remove_all,
        array_type<int> data_index_all, array_type<int> noise_index_all,
        array_type<double> chunk_t_starts,
        array_type<int> chunk_keep_lo, array_type<int> chunk_keep_hi,
        array_type<int> chunk_n_global_offset,
        array_type<double> wdm_window,
        array_type<double> data_d, array_type<double> invC,
        int n_chunks, int num_bin, int nparams,
        int Nt_sub, int log2_Nt_sub,
        int N_sparse, int log2_N_sparse,
        int nchannels, int n_rfft_chunk,
        double T_chunk, double dt, double T, double t_ref, int tdi_type,
        double tukey_alpha, int grid_dim, int N_cp_sig, int N_cp_orbit,
        array_type<int> binary_perm, array_type<int> group_starts, array_type<int> group_ends,
        array_type<int> group_m_lo, array_type<int> group_m_hi, int n_groups,
        array_type<int> pair_m_lo_b, array_type<int> pair_m_hi_b,
        int m_band_half_width,
        // task-b per-band slab (chunked_het.py always passes: 0 + empty = off).
        int Nf_slab, array_type<int> slab_min_f);

    void gb_wdm_het_get_fstat_ll(
        array_type<double> N_arr_re_out, array_type<double> N_arr_im_out,
        array_type<double> M_mat_re_out, array_type<double> M_mat_im_out,
        OrbitsWrap *orbits_wrap, TDIConfigWrap *tdi_config_wrap,
        WDMSettingsWrap *wdm_settings_wrap,
        array_type<double> params_all,
        array_type<int> data_index_all, array_type<int> noise_index_all,
        array_type<double> chunk_t_starts,
        array_type<int> chunk_keep_lo, array_type<int> chunk_keep_hi,
        array_type<int> chunk_n_global_offset,
        array_type<double> wdm_window,
        array_type<double> data_d, array_type<double> invC,
        int n_chunks, int num_bin, int nparams,
        int Nt_sub, int log2_Nt_sub,
        int N_sparse, int log2_N_sparse,
        int nchannels, int n_rfft_chunk,
        double T_chunk, double dt, double T, double t_ref, int tdi_type,
        double tukey_alpha, int grid_dim, int m_band_half_width,
        // task-b per-band slab (chunked_het.py always passes: 0 + empty = off).
        int Nf_slab, array_type<int> slab_min_f);

    // Signal-heterodyne (v2 polyphase) -- Stage 1 (CPU-only):
    // takes precomputed rfft(Tukey * td_cand) as input. Production will move
    // FD generation into the kernel via a sparse-spline absolute-FD source
    // (Stage 2 -- GBAbsoluteFD; ~256-1024 knots/year, no per-source global FD).
    void gb_signal_het_get_ll(
        array_type<double> d_h_out, array_type<double> h_h_out,
        array_type<std::complex<double>> fd_rfft_all,
        array_type<std::complex<double>> c0_sparse_all,
        array_type<std::complex<double>> A0_all,
        array_type<std::complex<double>> A1_all,
        array_type<std::complex<double>> B0_all,
        array_type<std::complex<double>> B1_all,
        array_type<double> wdm_window,
        array_type<int> n_sparse_local_arr,
        array_type<double> params_cand_all,
        array_type<double> params_ref_all,
        array_type<int> data_index_all,
        int num_bin, int num_data,
        int nparams, int f0_idx, int fdot_idx,
        int Nf, int Nt, int Nf_active, int Nt_active,
        int Nt_layer, int N_sparse_t, int stride,
        int ind_min_t, int ind_min_f,
        int m_active_half_width,
        double layer_df, double dt,
        int nchannels, int tdi_type,
        int n_rfft, double max_r);

    // Stage 2a -- sparse-FD entry: consumes X_het (length N_sparse_fd per
    // binary per channel) + per-binary k_f0 instead of the dense rfft.
    // The polyphase fold iterates only the N_sparse_fd bins (the source's
    // intrinsic spectral support around f0).
    void gb_signal_het_get_ll_sparse(
        array_type<double> d_h_out, array_type<double> h_h_out,
        array_type<std::complex<double>> X_het_all,
        array_type<int> k_f0_all,
        array_type<std::complex<double>> c0_sparse_all,
        array_type<std::complex<double>> A0_all,
        array_type<std::complex<double>> A1_all,
        array_type<std::complex<double>> B0_all,
        array_type<std::complex<double>> B1_all,
        array_type<double> wdm_window,
        array_type<int> n_sparse_local_arr,
        array_type<double> params_cand_all,
        array_type<double> params_ref_all,
        array_type<int> data_index_all,
        int num_bin, int num_data,
        int nparams, int f0_idx, int fdot_idx,
        int Nf, int Nt, int Nf_active, int Nt_active,
        int Nt_layer, int N_sparse_t, int stride,
        int ind_min_t, int ind_min_f,
        int m_active_half_width,
        double layer_df, double dt,
        int nchannels, int tdi_type,
        int N_sparse_fd, double max_r);

    // Stage 2b -- in-kernel sparse-FD signal-het. Fuses gb_run_fd_wave_tdi
    // with polyphase + bin-fold using a transient buffer for X_het (no
    // per-source FD storage in global memory). Takes a GBTDIonTheFlyWrap*
    // and unboxes to the underlying GBTDIonTheFly pointer internally.
    // tukey_alpha: Python pushes the same alpha used on the dense
    // rfft(Tukey*td) side so the sparse-FD heterodyne windowing matches.
    void gb_signal_het_get_ll_in_kernel(
        GBTDIonTheFlyWrap *tdi_wrap,
        array_type<double> d_h_out, array_type<double> h_h_out,
        array_type<std::complex<double>> c0_sparse_all,
        array_type<std::complex<double>> A0_all,
        array_type<std::complex<double>> A1_all,
        array_type<std::complex<double>> B0_all,
        array_type<std::complex<double>> B1_all,
        array_type<std::complex<double>> B0nc_all,
        array_type<std::complex<double>> B1nc_all,
        array_type<double> wdm_window,
        array_type<int> n_sparse_local_arr,
        array_type<double> params_cand_all,
        array_type<double> params_ref_all,
        array_type<int> data_index_all,
        int num_bin, int num_data,
        int nparams, int f0_idx, int fdot_idx,
        int Nf, int Nt, int Nf_active, int Nt_active,
        int Nt_layer, int N_sparse_t, int stride,
        int ind_min_t, int ind_min_f,
        int m_active_half_width,
        double layer_df, double dt,
        double T_obs, double t_start,
        int nchannels, int tdi_type,
        int N_sparse_fd, double tukey_alpha, double max_r, int project_real);

    // Signal-het fill_global. Reuses Stage 2b's FD + polyphase machinery to
    // build r at sparse n, then linear-interpolates r to the dense WDM
    // time grid, re-rotates the carrier, multiplies by the stored full
    // c0_dense_complex on the active band, takes Re, and scatters into the
    // (num_data, nchannels, Nf, Nt) template_fill buffer. factors_all
    // scales each binary's contribution; data_index_all routes each binary
    // to a template slab. tukey_alpha is required, no default -- pass the
    // value used to window the dense rfft on the analysis side.
    void gb_signal_het_fill_global_in_kernel(
        GBTDIonTheFlyWrap *tdi_wrap,
        array_type<double> template_fill,
        array_type<std::complex<double>> c0_sparse_all,
        array_type<std::complex<double>> c0_dense_complex_all,
        array_type<double> wdm_window,
        array_type<int> n_sparse_local_arr,
        array_type<double> params_cand_all,
        array_type<double> params_ref_all,
        array_type<double> factors_all,
        array_type<int> data_index_all,
        int num_bin, int num_data,
        int nparams, int f0_idx, int fdot_idx,
        int Nf, int Nt, int Nf_active, int Nt_active,
        int Nt_layer, int N_sparse_t, int stride,
        int ind_min_t, int ind_min_f,
        int m_active_half_width,
        double layer_df, double dt,
        double T_obs, double t_start,
        int nchannels,
        int N_sparse_fd, double tukey_alpha, double max_r);

    // Reference producer: emit the reference WDM c0 (sparse + dense complex) from
    // the REFERENCE params via the backend FD-gen + polyphase -- replaces the
    // Python polyphase. c0_sparse_out (num_data,nch,Nf_active,N_sparse_t);
    // c0_dense_out (num_data,nch,Nf_active,Nt_active).
    void gb_signal_het_make_reference(
        GBTDIonTheFlyWrap *tdi_wrap,
        array_type<std::complex<double>> c0_sparse_out,
        array_type<std::complex<double>> c0_dense_out,
        array_type<double> wdm_window,
        array_type<int> n_sparse_local_arr,
        array_type<double> params_ref_all,
        int num_data,
        int nparams, int f0_idx, int fdot_idx,
        int Nf, int Nt, int Nf_active, int Nt_active,
        int Nt_layer, int N_sparse_t, int stride,
        int ind_min_t, int ind_min_f,
        double layer_df, double dt,
        double T_obs, double t_start,
        int nchannels,
        int N_sparse_fd, double tukey_alpha);

    // Signal-het central-difference gradient of logL = d_h - 0.5*h_h. Per
    // binary, performs 1 central + 2*nparams perturbed get_ll_in_kernel
    // evaluations. grad_out is (num_bin, nparams); d_h_central /
    // h_h_central are (num_bin,) and report the central evaluation so the
    // caller gets logL alongside the gradient in one pass. param_eps[k] <=
    // 0 freezes dimension k.
    void gb_signal_het_get_ll_grad_in_kernel(
        GBTDIonTheFlyWrap *tdi_wrap,
        array_type<double> grad_out,
        array_type<double> d_h_central, array_type<double> h_h_central,
        array_type<std::complex<double>> c0_sparse_all,
        array_type<std::complex<double>> A0_all,
        array_type<std::complex<double>> A1_all,
        array_type<std::complex<double>> B0_all,
        array_type<std::complex<double>> B1_all,
        array_type<double> wdm_window,
        array_type<int> n_sparse_local_arr,
        array_type<double> params_cand_all,
        array_type<double> params_ref_all,
        array_type<int> data_index_all,
        array_type<double> param_eps,
        int num_bin, int num_data,
        int nparams, int f0_idx, int fdot_idx,
        int Nf, int Nt, int Nf_active, int Nt_active,
        int Nt_layer, int N_sparse_t, int stride,
        int ind_min_t, int ind_min_f,
        int m_active_half_width,
        double layer_df, double dt,
        double T_obs, double t_start,
        int nchannels, int tdi_type,
        int N_sparse_fd, double tukey_alpha, double max_r);
};


// Module entry called from NB_MODULE(cgbgpu, m) in binding_gbgpu.cxx.
void gbgpu_part(nb::module_ &m);

#endif // __BINDING_GBGPU_HPP__
