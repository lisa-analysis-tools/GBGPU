// GB-specific TDI-on-the-fly C++ method bodies + kernel launchers.
// Carved out of lisa-on-gpu's TDIonTheFly.cu at Phase 3L.7d (2026-06-04).
//
// What lives here (this slice):
// - `GBTDIonTheFly::ucb_amplitude / ucb_phase / ucb_f / ucb_fdot` — UCB physics.
// - `GBTDIonTheFly::get_amp / get_phase / get_f / get_fdot` — virtual overrides.
// - `GBTDIonTheFly::~GBTDIonTheFly` dtor.
// - `GBTDIonTheFly::get_gb_buffer_size` / `get_gb_fd_buffer_size` — shared-mem helpers.
// - `gb_run_wave_tdi_kernel` (CUDA_KERNEL) + `gb_run_wave_tdi_wrap` (host) — TD path.
//
// What still lives in lisa-on-gpu's TDIonTheFly.cu (subsequent slices migrate):
// - `gb_run_fd_wave_tdi_kernel` + `gb_run_fd_wave_tdi_wrap` (heterodyne FD path).
// - `gbfd_*` FD helpers (radix-2 FFT, build_one_source, run_one_source,
//   accumulate_ll, dense_bin, bit_reverse, log2_int).
// - `GBComputationGroup` class + its wrap methods (gb_fd_*, gb_wdm_het_*,
//   gb_signal_het_*).
// - `GBTDIonTheFlyWrap` + its pybind11 registration.
//
// **Compile-in-place pattern.** GBGPU's `cgbgpu` CMake target compiles
// this `.cu` file directly. lisa-on-gpu's
// `fastlisaresponse_{cpu,gpu}_tdionthefly` targets ALSO copy-compile
// this same file (via `${GBGPU_CUTILS}/gb_tdi_on_the_fly.cu` shell-out
// + `.cu→.cxx` for CPU), mirroring how lisa-on-gpu copy-compiles
// LAT's `lat_tdi_on_the_fly.cu` since Phase 3L.5. Both packages end
// up with their own compiled copy of the bodies in their own
// extension `.so`; RTLD_LOCAL keeps the symbols package-local so
// they don't collide when both wheels are imported in the same
// Python process.

#include "gb_tdi_on_the_fly.hh"  // GBTDIonTheFly class + LAT/GBT/LISATDIonTheFly machinery


// `N_PARAMS_MAX` is the upper bound on per-source parameter count
// used by the shared-memory layout calculations below. Same value
// as lisa-on-gpu's TDIonTheFly.cu (Phase 2 chunked-het work).
// Defined here so this `.cu` is self-contained (no dependency on
// lisa-on-gpu's TDIonTheFly.cu when both files end up in the same
// CMake target).
#ifndef N_PARAMS_MAX
#define N_PARAMS_MAX 20
#endif


// =============================================================================
// GBTDIonTheFly --- UCB closed-form physics
// =============================================================================
// LDC convention: phase parameter `phi0` enters with a leading minus
// sign (`-phi0 + 2*pi * cumulative-phase`).

CUDA_DEVICE
double GBTDIonTheFly::ucb_phase(double t, double *params)
{
    double f0    = params[f0_index];
    double phi0  = params[phi0_index];
    double fdot  = params[fdot0_index];
    double fddot = params[fddot0_index];

    double t_diff = t - t_ref;
    return -phi0 + 2 * M_PI * (f0 * t_diff
                              + 0.5 * fdot * t_diff * t_diff
                              + (1.0 / 6.0) * fddot * t_diff * t_diff * t_diff);
}

CUDA_DEVICE
double GBTDIonTheFly::ucb_amplitude(double t, double *params)
{
    double A0    = params[amplitude_index];
    double f0    = params[f0_index];
    double fdot  = params[fdot0_index];
    double t_diff = t - t_ref;
    return A0 * (1.0 + (2.0 / 3.0) * fdot / f0 * t_diff);
}

CUDA_DEVICE
double GBTDIonTheFly::ucb_f(double t, double *params)
{
    double f0    = params[f0_index];
    double fdot  = params[fdot0_index];
    double fddot = params[fddot0_index];
    double t_diff = t - t_ref;
    return f0 + fdot * t_diff + 0.5 * fddot * t_diff * t_diff;
}

CUDA_DEVICE
double GBTDIonTheFly::ucb_fdot(double t, double *params)
{
    double fdot  = params[fdot0_index];
    double fddot = params[fddot0_index];
    double t_diff = t - t_ref;
    return fdot + fddot * t_diff;
}


// -----------------------------------------------------------------------------
// LISATDIonTheFly virtual overrides --- delegate to UCB physics.
// -----------------------------------------------------------------------------

CUDA_DEVICE
double GBTDIonTheFly::get_phase(double t, double *params, int bin_i)
{
    // TD is based on sc1 time.
    return ucb_phase(t, params);
}

CUDA_DEVICE
double GBTDIonTheFly::get_amp(double t, double *params, int bin_i)
{
    return ucb_amplitude(t, params);
}

CUDA_DEVICE
double GBTDIonTheFly::get_f(double t, double *params, int bin_i)
{
    return ucb_f(t, params);
}

CUDA_DEVICE
double GBTDIonTheFly::get_fdot(double t, double *params, int bin_i)
{
    return ucb_fdot(t, params);
}


CUDA_DEVICE
GBTDIonTheFly::~GBTDIonTheFly()
{
    return;
}


// =============================================================================
// Time-domain GB TDI kernel + host wrapper.
// =============================================================================

#ifdef __CUDACC__
CUDA_KERNEL
void gb_run_wave_tdi_kernel(GBTDIonTheFly *tdi_on_fly, int buffer_length, cmplx *tdi_channels_arr,
    double *tdi_amp, double *tdi_phase, double *phi_ref,
    double *params, double *t_arr, int N, int num_bin, int n_params, int nchannels)
{
    extern CUDA_SHARED char shared_mem[];
    void *buffer = (void*) shared_mem;

    GBTDIonTheFly tdi_on_fly_here(tdi_on_fly->orbits, tdi_on_fly->tdi_config,
                                   tdi_on_fly->T, tdi_on_fly->t_ref);
    tdi_on_fly_here.run_wave_tdi(buffer, buffer_length, tdi_channels_arr,
                                  tdi_amp, tdi_phase, phi_ref,
                                  params, t_arr, N, num_bin, n_params, nchannels);
}
#endif

void gb_run_wave_tdi_wrap(GBTDIonTheFly *tdi_on_fly, cmplx *tdi_channels_arr,
    double *tdi_amp, double *tdi_phase, double *phi_ref,
    double *params, double *t_arr, int N, int num_bin, int n_params, int nchannels)
{
#ifdef __CUDACC__
    // Host -> device upload of the wrapper struct (sprint-wide rule;
    // see GBGPU CLAUDE.md "Host->device upload of class-wrapper objects").
    GBTDIonTheFly *gb_here = new GBTDIonTheFly(tdi_on_fly->orbits, tdi_on_fly->tdi_config,
                                                tdi_on_fly->T, tdi_on_fly->t_ref);
    Orbits *d_orbits;
    cudaMalloc(&d_orbits, sizeof(Orbits));
    gpuErrchk(cudaMemcpy(d_orbits, tdi_on_fly->orbits, sizeof(Orbits), cudaMemcpyHostToDevice));

    TDIConfig *d_tdi_config;
    cudaMalloc(&d_tdi_config, sizeof(TDIConfig));
    gpuErrchk(cudaMemcpy(d_tdi_config, tdi_on_fly->tdi_config, sizeof(TDIConfig), cudaMemcpyHostToDevice));

    gb_here->orbits     = d_orbits;
    gb_here->tdi_config = d_tdi_config;

    GBTDIonTheFly *d_gb_here;
    cudaMalloc(&d_gb_here, sizeof(GBTDIonTheFly));
    gpuErrchk(cudaMemcpy(d_gb_here, gb_here, sizeof(GBTDIonTheFly), cudaMemcpyHostToDevice));

    int buffer_length = tdi_on_fly->get_gb_buffer_size(N);
    printf("%d\n", buffer_length);
    gb_run_wave_tdi_kernel<<<num_bin, NUM_THREADS_HERE, buffer_length>>>(
        d_gb_here, buffer_length, tdi_channels_arr, tdi_amp, tdi_phase, phi_ref,
        params, t_arr, N, num_bin, n_params, nchannels);

    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());

    gpuErrchk(cudaFree(d_orbits));
    gpuErrchk(cudaFree(d_tdi_config));
    gpuErrchk(cudaFree(d_gb_here));
    delete gb_here;
#else
    // CPU branch: pass host pointers through (CUDA_SHARED collapses to
    // a regular stack/heap buffer; CUDA_KERNEL launchers degenerate to
    // a regular function call).
    int buffer_length = tdi_on_fly->get_gb_buffer_size(N);
    char *buffer = new char[buffer_length];
    tdi_on_fly->run_wave_tdi((void*) buffer, buffer_length, tdi_channels_arr,
                              tdi_amp, tdi_phase, phi_ref,
                              params, t_arr, N, num_bin, n_params, nchannels);
    delete[] buffer;
#endif
}


// =============================================================================
// Shared-memory budget helpers.
// =============================================================================

int GBTDIonTheFly::get_gb_buffer_size(int N)
{
    return N * sizeof(double) + get_tdi_buffer_size(N);
}

int GBTDIonTheFly::get_gb_fd_buffer_size(int N, int nchannels)
{
    // Shared-memory budget per source for the heterodyne FD kernel:
    //   params_here[N_PARAMS_MAX]                        N_PARAMS_MAX * 8
    //   t_arr_local[N]                                              N * 8
    //   tdi_channels_arr[nchannels * N]  (cmplx, FFT)    nchannels * N * 16
    //   tdi_amp[nchannels * N]                           nchannels * N * 8
    //   tdi_phase[nchannels * N]                         nchannels * N * 8
    //   phi_ref[N]                                                  N * 8
    //   get_tdi scratch (flip, pjump, count, fix_count)            21 * N
    return (int) (
          N_PARAMS_MAX * sizeof(double)
        + (size_t) N * sizeof(double)
        + (size_t) nchannels * (size_t) N * sizeof(cmplx)
        + 2 * (size_t) nchannels * (size_t) N * sizeof(double)
        + (size_t) N * sizeof(double)
        + (size_t) get_tdi_buffer_size(N)
    );
}
