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
#include <vector>

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
// CPU/GPU class-name aliasing (both the C++ class and any per-class
// pybind11 Wrap get separate aliasing in binding_gbgpu.hpp).
#if defined(__CUDA_COMPILATION__) || defined(__CUDACC__)
#define GBTDIonTheFly      GBTDIonTheFlyGPU
#define GBComputationGroup GBComputationGroupGPU
#else
#define GBTDIonTheFly      GBTDIonTheFlyCPU
#define GBComputationGroup GBComputationGroupCPU
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


// ----------------------------------------------------------------------------
// gbfd_* FD helpers (defined in gb_tdi_on_the_fly.cu).
// ----------------------------------------------------------------------------
// Forward decls so downstream TUs (lisa-on-gpu's TDIonTheFly.cu
// kernels that still use these helpers until Phase 3L.7f.2+ carves
// them out too) find the symbols at compile time. The short helpers
// (gbfd_log2_int / gbfd_bit_reverse / gbfd_dense_bin) are
// header-inline so they remain visible in every TU.
// ----------------------------------------------------------------------------

CUDA_DEVICE
inline int gbfd_log2_int(int n)
{
    int r = 0;
    while ((n >>= 1) != 0) ++r;
    return r;
}

CUDA_DEVICE
inline int gbfd_bit_reverse(int x, int log2n)
{
    int r = 0;
    for (int i = 0; i < log2n; ++i)
    {
        r = (r << 1) | (x & 1);
        x >>= 1;
    }
    return r;
}

// Helper: dense rfft bin index for sparse FFT bin m (FFT order) when
// the heterodyne carrier was snapped to dense bin kf0. Inlined math
// to np.fft.fftfreq(N, d=1/N): m_signed = (m < N/2) ? m : m - N.
CUDA_DEVICE
inline int gbfd_dense_bin(int m, int N, int kf0)
{
    int m_signed = (m < (N >> 1)) ? m : (m - N);
    return kf0 + m_signed;
}

// Larger helpers live in gb_tdi_on_the_fly.cu (compiled in-place by
// each consuming wheel via copy-compile).
CUDA_DEVICE
void gbfd_radix2_fft_inplace(cmplx *a, int N, int log2N);

CUDA_DEVICE
void gbfd_build_one_source(GBTDIonTheFly *tof, void *shared_mem,
                           double *params_in, double t_start, double Tobs,
                           int N, int nchannels, int n_params, int bin_i,
                           int log2N,
                           cmplx **tdi_chan_out,
                           int *kf0_out, double *f0g_out, double *dts_out,
                           double tukey_alpha);

CUDA_DEVICE
void gbfd_run_one_source(GBTDIonTheFly *tof, void *shared_mem,
                         cmplx *X_het, int *k_f0_out, double *f0_grid_out,
                         double *params_in, double t_start, double Tobs,
                         int N, int nchannels, int n_params, int bin_i,
                         int log2N, double tukey_alpha);


// Phase 3L.7f.1 (2026-06-04): `class GBComputationGroup` declaration
// migrates here from lisa-on-gpu's TDIonTheFly.hh:404-790. Method
// signatures only -- the wrap method bodies + the gb_fd_*_kernel /
// gb_wdm_het_*_kernel / gb_signal_het_*_kernel launchers + the
// associated gb_signal_het_* device helpers still live in
// lisa-on-gpu's TDIonTheFly.cu. Subsequent Phase 3L.7f sub-slices
// (3L.7f.2 onward) move them here in chunks by family.


// =============================================================================
// GBComputationGroup
// -----------------------------------------------------------------------------
// Python-facing GB computation surface. Method families:
//
// - `gb_fd_*_wrap` (gb_fd_fill_global, gb_fd_get_ll, gb_fd_swap_ll +
//   their gradient variants) -- heterodyne FD inner-product /
//   template-fill / RJMCMC swap accumulators.
//
// - `gb_wdm_het_*_wrap` (gb_wdm_het_fill_global, gb_wdm_het_get_ll,
//   gb_wdm_het_swap_ll, gb_wdm_het_get_fstat_ll) -- chunked-WDM-
//   heterodyne paths that call `wdm_het_*_impl<GBTDIonTheFly>(...)`
//   from LAT's `lat_chunked_het_kernels.hh` (Phase 3L.7a).
//
// - `gb_signal_het_*_wrap` -- V2 polyphase signal-heterodyne paths.
//
// - `gb_wdm_spline_eval_inputs_wrap` -- spline-path diagnostic
//   (gb_wdm_spline_* kernels disabled at Phase 3L; eval_inputs stays
//   active for direct-vs-spline-input comparison).
// =============================================================================



class GBComputationGroup{
  public:

    // Spline-path mirror of gb_wdm_fill_global_wrap. Replaces per-WDM-pixel
    // fast_wdm_inner calls with cubic-spline interpolation of get_tdi outputs
    // on a coarse uniform time grid of spacing `coarse_dt` (seconds). Output
    // template_fill is bit-compatible with the direct path's output up to
    // cubic-spline interpolation error.
    // gb_wdm_spline_fill_global_wrap disabled at Phase 3L (2026-06-02) -- WaveletLookupTable retiring.

    // Chunked-heterodyne family. Replaces the per-pixel WaveletLookupTable path
    // with a per-chunk dense-rfft + WDM xform built on the slow signal
    // (heterodyne to f0_grid). Geometry (chunk_t_starts / keep_lo / keep_hi /
    // n_global_offset / wdm_window) is precomputed on the host -- see
    // ``gb_wdm_het.compute_chunk_geometry`` / ``compute_wdm_window``.
    //
    // ``grid_dim`` selects the launch grid (number of CUDA blocks). The Python
    // helper ``chunked_het_grid_dim()`` picks an A100/H100-optimal value; pass
    // anything > 0 on CPU (ignored). Workspaces are allocated and freed inside
    // this wrapper.
    void gb_wdm_het_fill_global_wrap(
        double *template_fill,
        Orbits *orbits, TDIConfig *tdi_config,
        WDMSettings *wdm_settings,
        double *params_all, double *factors_all,
        int *data_index_all,
        double *chunk_t_starts, int *chunk_keep_lo, int *chunk_keep_hi,
        int *chunk_n_global_offset,
        double *wdm_window,
        int n_chunks, int num_bin, int nparams,
        int Nt_sub, int log2_Nt_sub,
        int N_sparse, int log2_N_sparse,
        int nchannels, int n_rfft_chunk,
        double T_chunk, double dt, double T, double t_ref,
        double tukey_alpha,
        int grid_dim, int N_cp_sig, int N_cp_orbit,
        int m_band_half_width, bool active_band = false,
        int Nf_slab = 0, int *slab_min_f = nullptr);  // task-b per-band slab

    void gb_wdm_het_get_ll_wrap(
        double *d_h_out, double *h_h_out,
        Orbits *orbits, TDIConfig *tdi_config,
        WDMSettings *wdm_settings,
        double *params_all,
        int *data_index_all, int *noise_index_all,
        double *chunk_t_starts, int *chunk_keep_lo, int *chunk_keep_hi,
        int *chunk_n_global_offset,
        double *wdm_window,
        double *data_d, double *invC,
        int n_chunks, int num_bin, int nparams,
        int Nt_sub, int log2_Nt_sub,
        int N_sparse, int log2_N_sparse,
        int nchannels, int n_rfft_chunk,
        double T_chunk, double dt, double T, double t_ref, int tdi_type,
        double tukey_alpha,
        int grid_dim, int N_cp_sig, int N_cp_orbit,
        int *binary_perm, int *group_starts, int *group_ends,
        int *group_m_lo, int *group_m_hi, int n_groups,
        int m_band_half_width,
        int Nf_slab = 0, int *slab_min_f = nullptr);  // task-b per-band slab

    void gb_wdm_het_swap_ll_wrap(
        double *d_h_add_out, double *d_h_remove_out,
        double *add_add_out, double *remove_remove_out, double *add_remove_out,
        Orbits *orbits, TDIConfig *tdi_config,
        WDMSettings *wdm_settings,
        double *params_add_all, double *params_remove_all,
        int *data_index_all, int *noise_index_all,
        double *chunk_t_starts, int *chunk_keep_lo, int *chunk_keep_hi,
        int *chunk_n_global_offset,
        double *wdm_window,
        double *data_d, double *invC,
        int n_chunks, int num_bin, int nparams,
        int Nt_sub, int log2_Nt_sub,
        int N_sparse, int log2_N_sparse,
        int nchannels, int n_rfft_chunk,
        double T_chunk, double dt, double T, double t_ref, int tdi_type,
        double tukey_alpha,
        int grid_dim, int N_cp_sig, int N_cp_orbit,
        int *binary_perm, int *group_starts, int *group_ends,
        int *group_m_lo, int *group_m_hi, int n_groups,
        int *pair_m_lo_b, int *pair_m_hi_b,
        int m_band_half_width,
        int Nf_slab = 0, int *slab_min_f = nullptr);  // task-b per-band slab

    // F-stat (chunked-heterodyne). Builds the 4 Cornish & Crowder '05 basis
    // filters per binary and writes:
    //   N_arr_re/im_out :  (num_bin, 4)  -- <d|A_i> (im always 0 for real WDM)
    //   M_mat_re/im_out :  (num_bin, 10) -- upper-triangle <A_i|A_j> (i<=j)
    // Python computes F = N^T M^{-1} N / 2 from these.
    void gb_wdm_het_get_fstat_ll_wrap(
        double *N_arr_re_out, double *N_arr_im_out,
        double *M_mat_re_out, double *M_mat_im_out,
        Orbits *orbits, TDIConfig *tdi_config,
        WDMSettings *wdm_settings,
        double *params_all,
        int *data_index_all, int *noise_index_all,
        double *chunk_t_starts, int *chunk_keep_lo, int *chunk_keep_hi,
        int *chunk_n_global_offset,
        double *wdm_window,
        double *data_d, double *invC,
        int n_chunks, int num_bin, int nparams,
        int Nt_sub, int log2_Nt_sub,
        int N_sparse, int log2_N_sparse,
        int nchannels, int n_rfft_chunk,
        double T_chunk, double dt, double T, double t_ref, int tdi_type,
        double tukey_alpha,
        int grid_dim, int m_band_half_width,
        int Nf_slab = 0, int *slab_min_f = nullptr);  // task-b per-band slab

    // Spline-path mirrors. `coarse_dt` (seconds) sets the coarse-grid spacing
    // for the cubic-spline window builder (smaller -> more accurate / more
    // get_tdi work). Python computes coarse_dt from a user knob
    // `coarse_pts_per_year` (typical 256).
    // gb_wdm_spline_get_ll_wrap, gb_wdm_spline_swap_ll_wrap disabled at Phase 3L (2026-06-02).

    // Chain-rule parameter gradients of the two likelihood kernels.
    //
    //   grad_out             (num_bin, nparams)  -- dL/dtheta for get_ll
    //   grad_{add,remove}_out (num_bin, nparams) -- d(ll_diff)/d(theta_{add,remove}) for swap_ll
    //
    // The per-parameter central-difference step size is supplied via
    // ``param_eps`` (length nparams).  Passing eps <= 0 freezes that
    // parameter (gradient stays zero).

    // Spline-path mirror of gb_wdm_get_ll_grad_wrap. Three spline slots in
    // shared memory (base, plus, minus); per-parameter inner loop rebuilds
    // the plus/minus slots from `params + eps_k e_k` and `params - eps_k e_k`
    // while the base slot is reused. `eps_k <= 0` freezes parameter k as
    // with the direct path. Shared-memory footprint is constant in nparams.
    // gb_wdm_spline_get_ll_grad_wrap disabled at Phase 3L (2026-06-02).


    // Spline analog of gb_wdm_eval_inputs_wrap. Builds ONE coarse-grid spline
    // window of WDM_SPLINE_L points starting at t_window_start with spacing
    // coarse_dt, then evaluates the splines at every tn in tn_arr. Outputs
    // are in the same convention as gb_wdm_eval_inputs_wrap.
    void gb_wdm_spline_eval_inputs_wrap(
        Orbits *orbits, TDIConfig *tdi_config,
        double *params_all, double *tn_arr,
        int num_bin, int nparams, int num_t, int nchannels,
        double T, double t_ref,
        double t_window_start, double coarse_dt,
        double *amp_out, double *phi_out, double *f_out, double *fdot_out,
        double *phase_ref_out);

    // ------------------------------------------------------------------
    // FD analogs of the WDM methods above.
    //
    // gb_fd_fill_global_wrap: add (or subtract -- via factors_all[bin_i] in {+1,-1})
    //    each GB's heterodyne FD piece into a global rfft-grid template buffer
    //    of shape (num_data, num_channel, n_rfft) addressed by data_index_all.
    //
    // gb_fd_get_ll_wrap: compute (d|h) and (h|h) per binary using the
    //    standard lisatools FD inner product
    //         (a|b) = 4 Re sum_{c1,c2} sum_k conj(a_c1[k]) b_c2[k] invC[c1,c2][k] df
    //    with cross-channel invC for tdi_type=TDI_XYZ and diagonal invC for
    //    tdi_type=TDI_AET / TDI_AE.
    //
    // gb_fd_swap_ll_wrap: same accumulators as the WDM swap, restricted to the
    //    union of the add- and remove-source sparse supports.
    //
    // All three share the same per-source heterodyne FD pass that
    // GBFDTDIonTheFly already uses.  N_sparse must be a power of two.
    // template_start_inds: per-TEMPLATE-row absolute start bin (length =
    // number of template rows, indexed by data_index). The template buffer
    // is caller-provided and independent of fd's data rows, so it carries
    // its own window offsets (pass zeros for the legacy full-grid layout).
    void gb_fd_fill_global_wrap(cmplx *template_fill,
        Orbits* orbits, TDIConfig *tdi_config, FDDomain *fd,
        double *params_all, int *data_index_all, double *factors_all,
        int *template_start_inds,
        int num_bin, int nparams, double T, double t_start, double t_ref,
        int N_sparse, int nchannels, double tukey_alpha, double edge_frac);

    void gb_fd_get_ll_wrap(double *d_h_out, double *h_h_out,
        Orbits* orbits, TDIConfig *tdi_config, FDDomain *fd,
        double *params_all, int *data_index_all, int *noise_index_all,
        int num_bin, int nparams, double T, double t_start, double t_ref,
        int N_sparse, int nchannels, int tdi_type, double tukey_alpha,
        double edge_frac);

    void gb_fd_swap_ll_wrap(
        double *d_h_add_out, double *d_h_remove_out,
        double *add_add_out, double *remove_remove_out, double *add_remove_out,
        Orbits* orbits, TDIConfig *tdi_config, FDDomain *fd,
        double *params_add_all, double *params_remove_all,
        int *data_index_all, int *noise_index_all,
        int num_bin, int nparams, double T, double t_start, double t_ref,
        int N_sparse, int nchannels, int tdi_type, double tukey_alpha,
        double edge_frac);

    // Chain-rule parameter gradients of the two FD likelihood kernels.
    // Same convention as the WDM counterparts:
    //   grad_out             (num_bin, nparams)  -- dL/dtheta for get_ll
    //   grad_{add,remove}_out (num_bin, nparams) -- d(ll_diff)/d(theta_{add,remove})
    // Per-parameter central-FD step is supplied via ``param_eps_*`` arrays
    // (length nparams).  Passing eps_k <= 0 freezes parameter k (grad stays 0).
    // Both routines are CPU-fully-wired; the GPU branch prints a TODO and
    // returns (matching gb_fd_swap_ll_wrap's status).
    void gb_fd_get_ll_grad_wrap(double *grad_out,
        Orbits* orbits, TDIConfig *tdi_config, FDDomain *fd,
        double *params_all, int *data_index_all, int *noise_index_all,
        double *param_eps,
        int num_bin, int nparams, double T, double t_start, double t_ref,
        int N_sparse, int nchannels, int tdi_type);

    void gb_fd_swap_ll_grad_wrap(
        double *grad_add_out, double *grad_remove_out,
        Orbits* orbits, TDIConfig *tdi_config, FDDomain *fd,
        double *params_add_all, double *params_remove_all,
        int *data_index_all, int *noise_index_all,
        double *param_eps_add, double *param_eps_remove,
        int num_bin, int nparams, double T, double t_start, double t_ref,
        int N_sparse, int nchannels, int tdi_type);

    // ------------------------------------------------------------------
    // Signal-heterodyne (v2 polyphase) family.
    //
    // First port of the v2 polyphase signal-het Python prototype at
    // ``LISAanalysistools/scripts/gb_chunked_het/gb_signal_het_wdm_v2.py``.
    // Takes the candidate's precomputed ``rfft(Tukey * td)`` and the
    // reference's precomputed ``c0_sparse / A0 / A1 / B0 / B1`` (bin-folded
    // at construction) and returns per-binary ``<d|h>``, ``<h|h>`` from
    // the bin-folded inner-product accumulator (v1-style sparse path:
    //    <d|h> = sum_{c',m,b} A0[c',m,b] * r[c',m,b] + A1[c',m,b] * dr/dn[c',m,b]
    // where r and dr/dn are evaluated at sparse bin centres without
    // carrier de-rotation; matches the Python prototype to FP precision
    // at DF0=0 and ~1% relative residual at DF0/layer_df=0.05).
    //
    // Active m-band = m_floor +/- m_active_half_width (m_floor =
    // floor(f0/layer_df), default half-width = 2 => 5 layers).
    //
    // Per-channel-index conventions (XYZ tdi_type=0; AE/AET tdi_type=1
    // uses diagonal B0/B1 of shape (num_data, nch, Nf_active, Nt_layer)
    // instead of (num_data, nch, nch, Nf_active, Nt_layer)).
    void gb_signal_het_get_ll_wrap(
        double *d_h_out,
        double *h_h_out,
        cmplx  *fd_rfft_all,
        cmplx  *c0_sparse_all,
        cmplx  *A0_all,
        cmplx  *A1_all,
        cmplx  *B0_all,
        cmplx  *B1_all,
        double *wdm_window,
        int    *n_sparse_local_arr,
        double *params_cand_all,
        double *params_ref_all,
        int    *data_index_all,
        int     num_bin, int num_data,
        int     nparams, int f0_idx, int fdot_idx,
        int     Nf, int Nt, int Nf_active, int Nt_active,
        int     Nt_layer, int N_sparse_t, int stride,
        int     ind_min_t, int ind_min_f,
        int     m_active_half_width,
        double  layer_df, double dt,
        int     nchannels, int tdi_type,
        int     n_rfft, double max_r);

    // Stage 2a: signal_het_get_ll consuming the SPARSE carrier-removed FD
    // (the output of GBTDIonTheFly::run_fd_wave_tdi -- length N_sparse_fd per
    // (binary, channel), centred at the per-binary k_f0 absolute-FD bin).
    // Polyphase fold iterates only over the N_sparse_fd nonzero bins,
    // implicit zero everywhere else. Eliminates per-source dense FD storage.
    //
    // X_het_all[bin, ch, i] is the absolute FD value at bin
    //     k_abs = k_f0_all[bin] + (i - N_sparse_fd/2)
    // i.e. the dense rfft restricted to a window of N_sparse_fd bins around
    // f0. In production (Stage 2b) this array is filled in-kernel from the
    // source-class heterodyned sparse rfft; here it is an input for
    // validation against the dense-FD path.
    void gb_signal_het_get_ll_sparse_wrap(
        double *d_h_out,
        double *h_h_out,
        cmplx  *X_het_all,
        int    *k_f0_all,
        cmplx  *c0_sparse_all,
        cmplx  *A0_all,
        cmplx  *A1_all,
        cmplx  *B0_all,
        cmplx  *B1_all,
        cmplx  *B0nc_all,
        cmplx  *B1nc_all,
        double *wdm_window,
        int    *n_sparse_local_arr,
        double *params_cand_all,
        double *params_ref_all,
        int    *data_index_all,
        int     num_bin, int num_data,
        int     nparams, int f0_idx, int fdot_idx,
        int     Nf, int Nt, int Nf_active, int Nt_active,
        int     Nt_layer, int N_sparse_t, int stride,
        int     ind_min_t, int ind_min_f,
        int     m_active_half_width,
        double  layer_df, double dt,
        int     nchannels, int tdi_type,
        int     N_sparse_fd, double max_r, int project_real);

    // Reference producer (V2 sig-het) -- EMIT the reference WDM ``c0`` from the
    // backend (replaces the Python polyphase ``_compute_sparse_complex_wdm``).
    // Runs ``gb_run_fd_wave_tdi`` on the REFERENCE params + the SAME polyphase as
    // ``gb_signal_het_get_ll_*``, but over ALL ``Nf_active`` layers and at BOTH the
    // sparse grid (``c0_sparse_out``, what get_ll consumes) and full ``Nt``
    // resolution (``c0_dense_out``, what the bin-fold / fill_global need). One call
    // per reference set; ``data_index`` selects them downstream. CPU-only (GPU TODO,
    // like the sibling sig-het wraps).
    void gb_signal_het_make_reference_wrap(
        GBTDIonTheFly *tdi_on_fly,
        cmplx  *c0_sparse_out,
        cmplx  *c0_dense_out,
        double *wdm_window,
        int    *n_sparse_local_arr,
        double *params_ref_all,
        int     num_data,
        int     nparams, int f0_idx, int fdot_idx,
        int     Nf, int Nt, int Nf_active, int Nt_active,
        int     Nt_layer, int N_sparse_t, int stride,
        int     ind_min_t, int ind_min_f,
        double  layer_df, double dt,
        double  T_obs, double t_start,
        int     nchannels,
        int     N_sparse_fd, double tukey_alpha);

    // Stage 2b -- in-kernel sparse-FD signal-het. Fuses the existing
    // ``gb_run_fd_wave_tdi`` (sparse heterodyned rfft) with the polyphase +
    // bin-fold pipeline. X_het is allocated in a transient per-call buffer
    // (heap on CPU; would be per-block shared memory on GPU at Stage 3).
    // No per-source FD storage in global memory.
    // tukey_alpha is set by the caller to match the alpha applied to the
    // dense rfft(Tukey * td) on the analysis side; 0.05 is the
    // recommended/default value pushed in from Python.
    void gb_signal_het_get_ll_in_kernel_wrap(
        GBTDIonTheFly *tdi_on_fly,
        double *d_h_out, double *h_h_out,
        cmplx  *c0_sparse_all,
        cmplx  *A0_all, cmplx *A1_all,
        cmplx  *B0_all, cmplx *B1_all,
        cmplx  *B0nc_all, cmplx *B1nc_all,
        double *wdm_window,
        int    *n_sparse_local_arr,
        double *params_cand_all,
        double *params_ref_all,
        int    *data_index_all,
        int     num_bin, int num_data,
        int     nparams, int f0_idx, int fdot_idx,
        int     Nf, int Nt, int Nf_active, int Nt_active,
        int     Nt_layer, int N_sparse_t, int stride,
        int     ind_min_t, int ind_min_f,
        int     m_active_half_width,
        double  layer_df, double dt,
        double  T_obs, double t_start,
        int     nchannels, int tdi_type,
        int     N_sparse_fd, double tukey_alpha, double max_r, int project_real);

    // Signal-het fill_global. Same FD + polyphase + r_sparse machinery as
    // get_ll, but reconstructs the dense template via the heterodyne
    // identity (linear-interp r_demod -> re-rotate carrier -> multiply by
    // stored c0_dense_complex on the full active band -> take real part)
    // and scatters into template_fill at the (m_global, n_global) WDM
    // positions of each active layer. Caller pre-zeroes / accumulates.
    void gb_signal_het_fill_global_sparse_wrap(
        double *template_fill,
        cmplx  *X_het_all, int *k_f0_all,
        cmplx  *c0_sparse_all,
        cmplx  *c0_dense_complex_all,
        double *wdm_window, int *n_sparse_local_arr,
        double *params_cand_all, double *params_ref_all,
        double *factors_all,
        int    *data_index_all,
        int     num_bin, int num_data,
        int     nparams, int f0_idx, int fdot_idx,
        int     Nf, int Nt, int Nf_active, int Nt_active,
        int     Nt_layer, int N_sparse_t, int stride,
        int     ind_min_t, int ind_min_f,
        int     m_active_half_width,
        double  layer_df, double dt,
        int     nchannels,
        int     N_sparse_fd, double max_r);

    void gb_signal_het_fill_global_in_kernel_wrap(
        GBTDIonTheFly *tdi_on_fly,
        double *template_fill,
        cmplx  *c0_sparse_all,
        cmplx  *c0_dense_complex_all,
        double *wdm_window, int *n_sparse_local_arr,
        double *params_cand_all, double *params_ref_all,
        double *factors_all,
        int    *data_index_all,
        int     num_bin, int num_data,
        int     nparams, int f0_idx, int fdot_idx,
        int     Nf, int Nt, int Nf_active, int Nt_active,
        int     Nt_layer, int N_sparse_t, int stride,
        int     ind_min_t, int ind_min_f,
        int     m_active_half_width,
        double  layer_df, double dt,
        double  T_obs, double t_start,
        int     nchannels,
        int     N_sparse_fd, double tukey_alpha, double max_r);

    // Signal-het central-difference gradient of logL = d_h - 0.5*h_h over
    // candidate params. param_eps[k] is the per-parameter finite-difference
    // step; eps_k <= 0 freezes dimension k. Also reports the central
    // d_h_central/h_h_central so callers get logL alongside the gradient
    // in a single pass.
    void gb_signal_het_get_ll_grad_in_kernel_wrap(
        GBTDIonTheFly *tdi_on_fly,
        double *grad_out,
        double *d_h_central, double *h_h_central,
        cmplx  *c0_sparse_all,
        cmplx  *A0_all, cmplx *A1_all,
        cmplx  *B0_all, cmplx *B1_all,
        double *wdm_window, int *n_sparse_local_arr,
        double *params_cand_all, double *params_ref_all,
        int    *data_index_all,
        double *param_eps,
        int     num_bin, int num_data,
        int     nparams, int f0_idx, int fdot_idx,
        int     Nf, int Nt, int Nf_active, int Nt_active,
        int     Nt_layer, int N_sparse_t, int stride,
        int     ind_min_t, int ind_min_f,
        int     m_active_half_width,
        double  layer_df, double dt,
        double  T_obs, double t_start,
        int     nchannels, int tdi_type,
        int     N_sparse_fd, double tukey_alpha, double max_r);
};


#endif // __GB_TDI_ON_THE_FLY_HH__
