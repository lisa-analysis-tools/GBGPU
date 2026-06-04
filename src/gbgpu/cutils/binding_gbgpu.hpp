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

// GBGPU-specific waveform/utils headers, included so the
// GBGPUComputationWrap method bodies can call out to the underlying free
// functions.
#include "gbgpu_utils.hh"        // fill_global_wrap, get_ll_wrap, swap_ll_diff_wrap
#include "SharedMemoryGBGPU.hpp" // SharedMemoryWaveComp, SharedMemoryLikeComp,
                                 // SharedMemorySwapLikeComp,
                                 // SharedMemoryChiSquaredComp,
                                 // SharedMemoryGenerateGlobal,
                                 // SharedMemoryFstatLikeComp

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
        double T, double dt, int N, int num_bin_all, int tdi_channel_setup)
    {
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
            T, dt, N, num_bin_all, tdi_channel_setup);
    }

    void SharedMemoryLikeComp_wrap(
        array_type<std::complex<double>> d_h,
        array_type<std::complex<double>> h_h,
        array_type<std::complex<double>> data,
        array_type<double> noise,
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
        int num_data, int num_noise)
    {
        SharedMemoryLikeComp(
            (cmplx*) return_pointer(d_h,  "d_h"),
            (cmplx*) return_pointer(h_h,  "h_h"),
            (cmplx*) return_pointer(data, "data"),
            return_pointer(noise,        "noise"),
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
            num_data, num_noise);
    }

    void SharedMemorySwapLikeComp_wrap(
        array_type<std::complex<double>> d_h_remove,
        array_type<std::complex<double>> d_h_add,
        array_type<std::complex<double>> remove_remove,
        array_type<std::complex<double>> add_add,
        array_type<std::complex<double>> add_remove,
        array_type<std::complex<double>> data,
        array_type<double> noise,
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
        int num_data, int num_noise)
    {
        SharedMemorySwapLikeComp(
            (cmplx*) return_pointer(d_h_remove,    "d_h_remove"),
            (cmplx*) return_pointer(d_h_add,       "d_h_add"),
            (cmplx*) return_pointer(remove_remove, "remove_remove"),
            (cmplx*) return_pointer(add_add,       "add_add"),
            (cmplx*) return_pointer(add_remove,    "add_remove"),
            (cmplx*) return_pointer(data,          "data"),
            return_pointer(noise,        "noise"),
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
            num_data, num_noise);
    }

    void SharedMemoryChiSquaredComp_wrap(
        array_type<std::complex<double>> h1_h1,
        array_type<std::complex<double>> h2_h2,
        array_type<std::complex<double>> h1_h2,
        array_type<double> noise,
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
        int num_data, int num_noise)
    {
        SharedMemoryChiSquaredComp(
            (cmplx*) return_pointer(h1_h1, "h1_h1"),
            (cmplx*) return_pointer(h2_h2, "h2_h2"),
            (cmplx*) return_pointer(h1_h2, "h1_h2"),
            return_pointer(noise,        "noise"),
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
            num_data, num_noise);
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
        int device, bool do_synchronize)
    {
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
            device, do_synchronize);
    }

    void SharedMemoryFstatLikeComp_wrap(
        array_type<std::complex<double>> M_mat,
        array_type<std::complex<double>> N_arr,
        array_type<std::complex<double>> data,
        array_type<double> noise,
        array_type<int> data_index, array_type<int> noise_index,
        array_type<double> f0, array_type<double> fdot0,
        array_type<double> fddot0,
        array_type<double> lam, array_type<double> theta,
        double T, double dt, int N, int num_bin_all,
        array_type<int> start_freq_inds,
        int data_length, int tdi_channel_setup,
        int device, bool do_synchronize,
        int num_data, int num_noise)
    {
        SharedMemoryFstatLikeComp(
            (cmplx*) return_pointer(M_mat, "M_mat"),
            (cmplx*) return_pointer(N_arr, "N_arr"),
            (cmplx*) return_pointer(data,  "data"),
            return_pointer(noise,        "noise"),
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
            num_data, num_noise);
    }
};

// Module entry called from PYBIND11_MODULE(cgbgpu, m) in binding_gbgpu.cxx.
void gbgpu_part(py::module &m);

#endif // __BINDING_GBGPU_HPP__
