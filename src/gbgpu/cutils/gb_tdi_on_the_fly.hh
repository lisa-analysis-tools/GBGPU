#ifndef __GB_TDI_ON_THE_FLY_HH__
#define __GB_TDI_ON_THE_FLY_HH__

// GB-specific TDI-on-the-fly machinery, carved out of lisa-on-gpu's
// `TDIonTheFly.{hh,cu}` + `binding_tof.{hpp,cxx}` at Phase 3L.7 (2026-06-04).
//
// What lives here:
// - `class GBTDIonTheFly : public LISATDIonTheFly` — GB UCB physics
//   (ucb_amplitude / ucb_phase / ucb_f / ucb_fdot, get_amp/phase/f/fdot
//   virtual overrides, dtor, buffer-size helpers).
// - GB-specific kernel + host-wrapper functions:
//   - `gb_run_wave_tdi_kernel` + `gb_run_wave_tdi_wrap` (time-domain TDI)
//   - `gb_run_fd_wave_tdi_kernel` + `gb_run_fd_wave_tdi_wrap`
//     (heterodyne sparse-FD TDI)
//   - `fast_wdm_inner_heterodyne_kernel` (block-per-chunk launcher that
//     instantiates `fast_wdm_inner_heterodyne_direct<GBTDIonTheFly>`
//     from LAT)
// - `class GBComputationGroup` — Python-facing computation surface;
//   `gb_fd_*_wrap`, `gb_wdm_het_*_wrap`, `gb_signal_het_*_wrap` methods
//   that instantiate LAT's templated `wdm_het_*_impl<GBTDIonTheFly>` /
//   `signal_het_*_impl<GBTDIonTheFly>` launchers against the GB source
//   class.
//
// What stays in lisa-on-gpu (until Phase 3L.8 carves it to BBHx):
// - `SOBBHTDIonTheFly` + `SOBBHComputationGroup` mirrors.
//
// What lives in LAT (post-Phase-3L.7a):
// - `LISATDIonTheFly` base + `OrbitsSplineCache` (Phase 3L.5)
// - Generic FD/WDM domains: `FDDomain`, `WDMSettings`, `WDMDomain` (Phase 3L.1/2/4)
// - All source-agnostic chunked-het machinery: helpers + 4 templated
//   `wdm_het_*_kernel` bodies + 4 inline `wdm_het_*_impl<SourceT>`
//   launchers (Phase 3L.7a, in `lat_chunked_het_kernels.hh`)
//
// CPU/GPU class-name aliasing follows the sprint-wide rule — both the
// class and its `*Wrap` must be aliased so per-backend plugin wheels
// emit distinct C++ type names that pybind11 can register
// independently.

// Pull in LAT's headers (where most of the chunked-het + base-class
// machinery lives post-Phase-3L).
#include "Detector.hpp"               // Orbits + Vec
#include "LISAResponse.hh"            // TDIConfig
#include "Interpolate.hh"             // NLINKS + CubicSpline helpers
#include "fd_domain.hh"               // FDDomain
#include "wdm_settings.hh"            // WDMSettings
#include "wdm_domain.hh"              // WDMDomain
#include "lat_tdi_on_the_fly.hh"      // LISATDIonTheFly base + OrbitsSplineCache
#include "lat_chunked_het_kernels.hh" // wdm_het_*_impl<SourceT> + helpers
#include "gbt_global.h"               // cmplx + CUDA_DEVICE etc.


// CPU/GPU class-name aliasing -- one rule, both layers.
//
// (a) The C++ classes themselves (GBTDIonTheFly + GBComputationGroup):
//     both must be aliased so per-backend plugin wheels emit distinct
//     C++ type names (e.g. `class GBTDIonTheFlyGPU` vs
//     `class GBTDIonTheFlyCPU`). This is what allows the cuda12x +
//     cpu plugin wheels to coexist in the same Python interpreter
//     without pybind11 type-registry collisions.
// (b) The pybind11 wrappers (GBTDIonTheFlyWrap + GBComputationGroupWrap)
//     get their own aliasing in `binding_gbgpu.hpp` -- see the macro
//     block there.
// Phase 3L.7c (2026-06-04): only GBTDIonTheFly migrates in this slice.
// GBComputationGroup aliasing + declaration stay in lisa-on-gpu's
// TDIonTheFly.hh until a subsequent slice carves the full class out.
#if defined(__CUDA_COMPILATION__) || defined(__CUDACC__)
#define GBTDIonTheFly      GBTDIonTheFlyGPU
#else
#define GBTDIonTheFly      GBTDIonTheFlyCPU
#endif


// ============================================================================
// GBTDIonTheFly
// ----------------------------------------------------------------------------
// LISATDIonTheFly subclass for galactic binaries (GBs). Stores the
// observation window (T, t_ref) plus the GB-parameter indices into the
// shared 9-element params array
// (amp, f0, fdot, fddot, phi0, iota, psi, lam, beta).
//
// Provides the UCB amplitude/phase/frequency/frequency-derivative
// closed-form expressions, which the base-class `get_tdi*` paths call
// at every sparse time point.
// ============================================================================
class GBTDIonTheFly : public LISATDIonTheFly {
  public:
    double T;
    double t_ref;
    int    amplitude_index;
    int    f0_index;
    int    fdot0_index;
    int    fddot0_index;
    int    phi0_index;

    CUDA_CALLABLE_MEMBER
    GBTDIonTheFly(Orbits *orbits_, TDIConfig *tdi_config_, double T_, double t_ref_)
        : LISATDIonTheFly(orbits_, tdi_config_, 5, 6, 7, 8)
    {
        T = T_;
        t_ref = t_ref_;
        amplitude_index = 0;
        f0_index        = 1;
        fdot0_index     = 2;
        fddot0_index    = 3;
        phi0_index      = 4;
    }

    CUDA_CALLABLE_MEMBER
    ~GBTDIonTheFly();

    // UCB closed-form physics (called by the base-class get_tdi pipeline).
    CUDA_DEVICE double ucb_amplitude(double t, double *params);
    CUDA_DEVICE double ucb_phase    (double t, double *params);
    CUDA_DEVICE double ucb_fdot     (double t, double *params);
    CUDA_DEVICE double ucb_f        (double t, double *params);

    // Per-binary virtual overrides used by the chunked-het +
    // signal-het kernels.
    CUDA_DEVICE double get_amp  (double t, double *params, int bin_i);
    CUDA_DEVICE double get_phase(double t, double *params, int bin_i);
    CUDA_DEVICE double get_f    (double t, double *params, int bin_i);
    CUDA_DEVICE double get_fdot (double t, double *params, int bin_i);

    // Shared-memory budget helpers (time-domain + heterodyne FD paths).
    CUDA_CALLABLE_MEMBER int get_gb_buffer_size   (int N);
    CUDA_CALLABLE_MEMBER int get_gb_fd_buffer_size(int N, int nchannels);
};


// ----------------------------------------------------------------------------
// Top-level launcher wrappers (host-side; create the GB source, set up
// shared memory, dispatch the kernel).
// ----------------------------------------------------------------------------

// Time-domain TDI for `num_bin` GB binaries on the sparse `t_arr` grid.
void gb_run_wave_tdi_wrap(
    GBTDIonTheFly *tdi_on_fly,
    cmplx *tdi_channels_arr,
    double *tdi_amp, double *tdi_phase, double *phi_ref,
    double *params, double *t_arr,
    int N, int num_bin, int n_params, int nchannels);

// Heterodyned frequency-domain GB TDI -- builds the slow positive-
// frequency complex signal on a sparse time grid, FFTs it, and writes
// the heterodyne band into X_het around the f0 carrier. See
// `lisa-on-gpu/cutils/TDIonTheFly.hh` Phase 2 commentary for the full
// algorithm description.
void gb_run_fd_wave_tdi_wrap(
    GBTDIonTheFly *tdi_on_fly,
    cmplx *X_het, int *k_f0_out, double *f0_grid_out,
    double *params, double t_start, double Tobs,
    int N_sparse, int num_bin, int n_params, int nchannels,
    double tukey_alpha);


// GBComputationGroup class declaration stays in lisa-on-gpu's
// TDIonTheFly.hh for now (Phase 3L.7c slice migrates only
// GBTDIonTheFly). A subsequent slice will carve GBComputationGroup
// + its methods (gb_fd_*, gb_wdm_het_*, gb_signal_het_*) here.


#endif // __GB_TDI_ON_THE_FLY_HH__
