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
#include <stdexcept>
#include <vector>

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
    // ``get_gb_buffer_size(N)`` can return >48 KB (the default per-block
    // dynamic-shared-mem cap). The other CUDA launchers in this file
    // (gb_fd_get_ll_kernel, gb_fd_fill_global_kernel, and the chunked-het
    // family in lat_chunked_het_kernels.hh) all opt into the higher cap
    // via cudaFuncSetAttribute before launching; this one was missing
    // it, which produced ``GPUassert: invalid argument`` at the
    // post-launch cudaGetLastError() check (line 183) for any N that
    // pushed the buffer above the default cap. Cluster CUDA report
    // 2026-06-06: N gave buffer_length = 475136 bytes.
    gpuErrchk(cudaFuncSetAttribute(
        gb_run_wave_tdi_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize, buffer_length));
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


// ============================================================================
// Phase 3L.7e (2026-06-04): GB heterodyne FD path -- carved out of
// lisa-on-gpu's TDIonTheFly.cu lines 5787-6167. Source-agnostic FD
// helpers (gbfd_log2_int, gbfd_bit_reverse, gbfd_radix2_fft_inplace,
// gbfd_dense_bin) + GB-specific (gbfd_build_one_source,
// gbfd_run_one_source) + the kernel/wrap launcher
// (gb_run_fd_wave_tdi_kernel + gb_run_fd_wave_tdi_wrap).
//
// `gbfd_accumulate_ll` + the `GBComputationGroup::gb_fd_*_wrap`
// methods that share these helpers stay in lisa-on-gpu for now
// (subsequent Phase 3L.7f slice moves them to GBGPU).
// ============================================================================

// ---------------------------------------------------------------------------
// Heterodyne FD GB kernel
// ---------------------------------------------------------------------------
//
// Per-source block; threads cooperate over the N_sparse time samples and over
// FFT butterflies.  All three channels live in shared memory simultaneously
// so cross-channel (XYZ -> AET, etc.) post-processing can happen before any
// global-memory write.
//
// Algorithm (per source, all in shared memory):
//   1. Re-run the existing get_tdi() on the sparse grid -> tdi_amp[c,n],
//      tdi_phase[c,n], phi_ref[n], tdi_channels_arr[c,n].
//   2. Overwrite tdi_channels_arr[c,n] with the slow positive-frequency
//      signal  s_c(tau_n) = A_c(tau_n) *
//                            exp(+i*(phi_c(tau_n) + phi_ref(tau_n)
//                                    - 2*pi*f0_grid * tau_n)).
//   3. In-place radix-2 Cooley-Tukey FFT per channel on s_c[0..N-1] using
//      cooperative bit-reversal + log2(N) butterfly passes.  Twiddles are
//      computed on the fly with sin/cos (full double precision).
//   4. Multiply by 0.5 * dt_sparse (the 1/2 from x = Re[z]) and write the
//      (num_bin, nchannels, N_sparse) complex result, along with k_f0[bin_i]
//      and f0_grid[bin_i] (dense rfft bin and snapped carrier).

// gbfd_log2_int + gbfd_bit_reverse are header-inline in
// gb_tdi_on_the_fly.hh (Phase 3L.7f.1, 2026-06-04). Same for
// gbfd_dense_bin below.

CUDA_DEVICE
void gbfd_radix2_fft_inplace(cmplx *a, int N, int log2N)
{
    // Cooley-Tukey decimation-in-time, in-place, double precision.  No GSL or
    // other library; permissive MIT-style hand roll.  Cooperative across the
    // threads of the block.

    // Bit-reversal permutation
    for (int n = THREAD_START_X; n < N; n += BLOCK_INCR_X)
    {
        int r = gbfd_bit_reverse(n, log2N);
        if (r > n)
        {
            cmplx t = a[n];
            a[n] = a[r];
            a[r] = t;
        }
    }
    CUDA_SYNC_THREADS;

    // log2(N) butterfly passes
    for (int s = 1; s <= log2N; ++s)
    {
        int m  = 1 << s;
        int mh = m >> 1;
        double base = -2.0 * M_PI / (double) m;  // forward FFT sign
        for (int k = THREAD_START_X; k < (N >> 1); k += BLOCK_INCR_X)
        {
            int g  = k / mh;          // butterfly group
            int j  = k - g * mh;      // position within group
            int i0 = g * m + j;
            int i1 = i0 + mh;
            double th = base * (double) j;
            cmplx w(cos(th), sin(th));
            cmplx u = a[i0];
            cmplx v = w * a[i1];
            a[i0] = u + v;
            a[i1] = u - v;
        }
        CUDA_SYNC_THREADS;
    }
}

// Build the heterodyne FD for one source into the shared-memory buffer.
//
// Side effects after return:
//   tdi_chan[c*N + n] holds  0.5 * dt_sparse * FFT[s_c][n]  (complex),
//   *kf0_out is the dense rfft bin closest to f0,
//   *f0g_out is the snapped carrier f0_grid = *kf0_out * df.
//
// All three (nchannels) channels are resident in shared memory at return,
// in FFT-order, ready for the inner-product / accumulator step.
//
// The shared-mem layout is exactly the one `get_gb_fd_buffer_size` reserves.
// `tdi_chan_out`, if non-NULL, also receives a pointer to the per-channel
// heterodyne FD slab within shared (size = nchannels * N complex).
// tukey_alpha: scipy.signal.windows.tukey alpha applied to the slow signal
// before the in-place FFT. 0.0 = rectangular (no taper), match
// FAST_WDM_TUKEY_ALPHA_HET_NARROW / _HET_WIDE (0.05 / 0.01) to mirror the
// dense rfft(Tukey*td) convention. The same taper formula is used as in
// the chunked-het sparse FD path (TDIonTheFly.cu:2074-2098) so cross-path
// inner products line up at FP precision.
CUDA_DEVICE
void gbfd_build_one_source(GBTDIonTheFly *tof, void *shared_mem,
                           double *params_in, double t_start, double Tobs,
                           int N, int nchannels, int n_params, int bin_i,
                           int log2N,
                           cmplx **tdi_chan_out,
                           int *kf0_out, double *f0g_out, double *dts_out,
                           double tukey_alpha, double edge_frac)
{
    // ---- carve up shared memory ------------------------------------------
    char *cur = (char*) shared_mem;

    double *params_here = (double*) cur;
    cur += N_PARAMS_MAX * sizeof(double);

    double *t_arr_local = (double*) cur;
    cur += (size_t) N * sizeof(double);

    cmplx *tdi_chan = (cmplx*) cur;             // also slow + FFT buffer
    cur += (size_t) nchannels * N * sizeof(cmplx);

    double *tdi_amp = (double*) cur;
    cur += (size_t) nchannels * N * sizeof(double);

    double *tdi_phase = (double*) cur;
    cur += (size_t) nchannels * N * sizeof(double);

    double *phi_ref = (double*) cur;
    cur += (size_t) N * sizeof(double);

    void *get_tdi_scratch = (void*) cur;
    int   get_tdi_scratch_len = tof->get_tdi_buffer_size(N);

    // ---- broadcast params into shared ------------------------------------
    for (int i = THREAD_START_X; i < n_params; i += BLOCK_INCR_X)
        params_here[i] = params_in[bin_i * n_params + i];
    CUDA_SYNC_THREADS;

    const double f0   = params_here[tof->f0_index];
    const double df   = 1.0 / Tobs;
    const int    kf0  = (int) llround(f0 / df);
    const double f0g  = (double) kf0 * df;
    const double dts  = Tobs / (double) N;

    // ---- build sparse absolute-time array in shared ----------------------
    for (int n = THREAD_START_X; n < N; n += BLOCK_INCR_X)
        t_arr_local[n] = t_start + (double) n * dts;
    CUDA_SYNC_THREADS;

    // ---- call existing get_tdi to fill tdi_chan / tdi_amp / tdi_phase /
    //      phi_ref from the sparse t_arr_local
    tof->get_tdi(get_tdi_scratch, get_tdi_scratch_len,
                 tdi_chan, tdi_amp, tdi_phase, phi_ref,
                 params_here, t_arr_local, N, bin_i, nchannels);

    // ---- build slow positive-freq complex signal in-place over tdi_chan --
    // Tukey window factored in-line: cosine taper on the first / last
    // alpha/2 fraction of the N sparse samples (rectangular middle). Matches
    // scipy.signal.windows.tukey(N, alpha) sample-by-sample so the sparse FD
    // matches the dense rfft(Tukey*td) inner product. alpha=0 -> no taper.
    const double n_taper_fd = 0.5 * tukey_alpha * (double) (N - 1);
    const double dlast_fd   = (double) (N - 1);
    // Edge-cut: zero the first / last edge_frac fraction of the sparse grid, so
    // the FD-het template analyses the SAME time region as the WDM grid's
    // [min_time, max_time] = [EC, Nt-EC] layers (edge_frac = EC/Nt). Without it
    // the FD het integrates the full Tobs and its off-source logL is ~13% too
    // steep vs the (edge-cut) WDM. The tukey taper is kept (it suppresses the
    // active-band leakage); when EC > taper the cut subsumes the taper region.
    const int n_edge = (int) llround(edge_frac * (double) N);
    for (int n = THREAD_START_X; n < N; n += BLOCK_INCR_X)
    {
        const double tau     = (double) n * dts;
        const double carrier = 2.0 * M_PI * f0g * tau;
        const double phref   = phi_ref[n];
        double w = 1.0;
        if (n < n_edge || n >= N - n_edge) {
            w = 0.0;
        } else if (tukey_alpha > 0.0 && n_taper_fd > 0.0) {
            const double di = (double) n;
            if (di < n_taper_fd) {
                const double xn = di / n_taper_fd;
                w = 0.5 * (1.0 + cos(M_PI * (xn - 1.0)));
            } else if (di > dlast_fd - n_taper_fd) {
                const double xn = (dlast_fd - di) / n_taper_fd;
                w = 0.5 * (1.0 + cos(M_PI * (xn - 1.0)));
            }
        }
        for (int c = 0; c < nchannels; ++c)
        {
            const double th = tdi_phase[c * N + n] + phref - carrier;
            tdi_chan[c * N + n] =
                gcmplx::polar(tdi_amp[c * N + n] * w, th);  // +i sign
        }
    }
    CUDA_SYNC_THREADS;

    // ---- NaN scrub. Any non-finite sample left over from a singular
    //      response geometry (e.g. the (1-k.n)->0 wave-axis-vs-arm
    //      alignment for one TDI link at one sparse-time sample) would
    //      otherwise be spread across the entire band by the in-place
    //      FFT below, NaN-ing 4096 contiguous output bins. Zero those
    //      samples so the FFT stays finite; we lose at most a handful of
    //      O(N_sparse^-1) sparse samples at the singular locus.
    for (int n = THREAD_START_X; n < N; n += BLOCK_INCR_X)
    {
        for (int c = 0; c < nchannels; ++c)
        {
            cmplx v = tdi_chan[c * N + n];
            if (!isfinite(v.real()) || !isfinite(v.imag()))
            {
                tdi_chan[c * N + n] = cmplx(0.0, 0.0);
            }
        }
    }
    CUDA_SYNC_THREADS;

    // ---- in-place radix-2 FFT, per channel ------------------------------
    for (int c = 0; c < nchannels; ++c)
    {
        gbfd_radix2_fft_inplace(tdi_chan + (size_t) c * N, N, log2N);
        CUDA_SYNC_THREADS;
    }

    // ---- absorb the 1/2 * dt_sparse scale here so callers can use the
    //      values directly as the heterodyne FD piece ---------------------
    const double scale = 0.5 * dts;
    for (int n = THREAD_START_X; n < N; n += BLOCK_INCR_X)
    {
        for (int c = 0; c < nchannels; ++c)
        {
            cmplx v = tdi_chan[c * N + n];
            tdi_chan[c * N + n] = cmplx(v.real() * scale, v.imag() * scale);
        }
    }
    CUDA_SYNC_THREADS;

    if (tdi_chan_out) *tdi_chan_out = tdi_chan;
    if (kf0_out)      *kf0_out      = kf0;
    if (f0g_out)      *f0g_out      = f0g;
    if (dts_out)      *dts_out      = dts;
}

// gbfd_dense_bin header-inline in gb_tdi_on_the_fly.hh
// (Phase 3L.7f.1, 2026-06-04).

CUDA_DEVICE
void gbfd_run_one_source(GBTDIonTheFly *tof, void *shared_mem,
                         cmplx *X_het, int *k_f0_out, double *f0_grid_out,
                         double *params_in, double t_start, double Tobs,
                         int N, int nchannels, int n_params, int bin_i,
                         int log2N, double tukey_alpha)
{
    cmplx *tdi_chan = NULL;
    int    kf0      = 0;
    double f0g      = 0.0;
    double dts      = 0.0;
    gbfd_build_one_source(tof, shared_mem, params_in, t_start, Tobs,
                          N, nchannels, n_params, bin_i, log2N,
                          &tdi_chan, &kf0, &f0g, &dts, tukey_alpha, 0.0);

    // Write heterodyne FD to global, in FFT order.
    for (int n = THREAD_START_X; n < N; n += BLOCK_INCR_X)
    {
        for (int c = 0; c < nchannels; ++c)
        {
            X_het[(size_t) bin_i * nchannels * N + (size_t) c * N + n] =
                tdi_chan[c * N + n];
        }
    }

    if (THREAD_ZERO)
    {
        k_f0_out[bin_i]    = kf0;
        f0_grid_out[bin_i] = f0g;
    }
    CUDA_SYNC_THREADS;
}

#ifdef __CUDACC__
CUDA_KERNEL
void gb_run_fd_wave_tdi_kernel(GBTDIonTheFly *tdi_on_fly,
    cmplx *X_het, int *k_f0_out, double *f0_grid_out,
    double *params, double t_start, double Tobs,
    int N, int num_bin, int n_params, int nchannels, int log2N,
    double tukey_alpha)
{
    extern CUDA_SHARED char shared_mem[];
    GBTDIonTheFly tof(tdi_on_fly->orbits, tdi_on_fly->tdi_config,
                      tdi_on_fly->T, tdi_on_fly->t_ref);
    for (int bin_i = BLOCK_START_X; bin_i < num_bin; bin_i += GRID_INCR_X)
    {
        gbfd_run_one_source(&tof, (void*) shared_mem,
                            X_het, k_f0_out, f0_grid_out,
                            params, t_start, Tobs,
                            N, nchannels, n_params, bin_i, log2N,
                            tukey_alpha);
    }
}
#endif

void gb_run_fd_wave_tdi_wrap(GBTDIonTheFly *tdi_on_fly,
    cmplx *X_het, int *k_f0_out, double *f0_grid_out,
    double *params, double t_start, double Tobs,
    int N_sparse, int num_bin, int n_params, int nchannels,
    double tukey_alpha)
{
    // Validate power-of-two
    int log2N = 0;
    {
        int m = N_sparse;
        while ((m & 1) == 0 && m > 1) { m >>= 1; ++log2N; }
#ifndef __CUDACC__
        if (m != 1) {
            throw std::invalid_argument(
                "gb_run_fd_wave_tdi_wrap: N_sparse must be a power of two.");
        }
#endif
    }

#ifdef __CUDACC__
    GBTDIonTheFly *gb_host = new GBTDIonTheFly(
        tdi_on_fly->orbits, tdi_on_fly->tdi_config,
        tdi_on_fly->T, tdi_on_fly->t_ref);

    Orbits *d_orbits;
    cudaMalloc(&d_orbits, sizeof(Orbits));
    gpuErrchk(cudaMemcpy(d_orbits, tdi_on_fly->orbits, sizeof(Orbits),
                         cudaMemcpyHostToDevice));

    TDIConfig *d_tdi_config;
    cudaMalloc(&d_tdi_config, sizeof(TDIConfig));
    gpuErrchk(cudaMemcpy(d_tdi_config, tdi_on_fly->tdi_config, sizeof(TDIConfig),
                         cudaMemcpyHostToDevice));

    gb_host->orbits     = d_orbits;
    gb_host->tdi_config = d_tdi_config;

    GBTDIonTheFly *d_gb;
    cudaMalloc(&d_gb, sizeof(GBTDIonTheFly));
    gpuErrchk(cudaMemcpy(d_gb, gb_host, sizeof(GBTDIonTheFly),
                         cudaMemcpyHostToDevice));

    int shared_bytes =
        tdi_on_fly->get_gb_fd_buffer_size(N_sparse, nchannels);

    // Allow shared usage past the 48 KB static default for large N_sparse.
    if (shared_bytes > 48 * 1024)
    {
        cudaFuncSetAttribute(
            gb_run_fd_wave_tdi_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            shared_bytes);
    }

    gb_run_fd_wave_tdi_kernel<<<num_bin, NUM_THREADS_HERE, shared_bytes>>>(
        d_gb, X_het, k_f0_out, f0_grid_out,
        params, t_start, Tobs,
        N_sparse, num_bin, n_params, nchannels, log2N, tukey_alpha);

    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());

    gpuErrchk(cudaFree(d_orbits));
    gpuErrchk(cudaFree(d_tdi_config));
    gpuErrchk(cudaFree(d_gb));
    delete gb_host;
#else
    const int shared_bytes =
        tdi_on_fly->get_gb_fd_buffer_size(N_sparse, nchannels);
    char *shared_mem = new char[shared_bytes];
    for (int bin_i = 0; bin_i < num_bin; ++bin_i)
    {
        gbfd_run_one_source(tdi_on_fly, (void*) shared_mem,
                            X_het, k_f0_out, f0_grid_out,
                            params, t_start, Tobs,
                            N_sparse, nchannels, n_params, bin_i, log2N,
                            tukey_alpha);
    }
    delete[] shared_mem;
#endif
}


// ============================================================================
// Phase 3L.7f.2 (2026-06-04): GB FD likelihood family -- carved out of
// lisa-on-gpu's TDIonTheFly.cu lines 5789-6657. Includes:
// - `gbfd_accumulate_ll` device-inline FD inner-product accumulator
//   (handles TDI_XYZ cross-channel and TDI_AET diagonal invC)
// - `gb_fd_get_ll_kernel` + `GBComputationGroup::gb_fd_get_ll_wrap`
// - `gb_fd_fill_global_kernel` + `GBComputationGroup::gb_fd_fill_global_wrap`
// - `gb_fd_swap_ll_kernel` + `GBComputationGroup::gb_fd_swap_ll_wrap`
// - `gbfd_grad_one_sided_partial` helper for gradient paths
// - `gb_fd_get_ll_grad_kernel` + `gb_fd_get_ll_grad_wrap`
// - `gb_fd_swap_ll_grad_kernel` + `gb_fd_swap_ll_grad_wrap`
// ============================================================================

// ===========================================================================
// FD analogs of the WDM GBComputationGroup methods.
// ===========================================================================
//
// All three share gbfd_build_one_source(...) to materialise the
// (nchannels, N_sparse) heterodyne FD piece in shared memory; only the
// accumulator / scatter step differs.
//
// Inner product convention (matches lisatools.diagnostic.inner_product):
//   (a|b) = 4 Re sum_{c1,c2} sum_k conj(a_c1[k]) b_c2[k] invC[c1,c2][k] * df
// for tdi_type = TDI_XYZ;  the (c1==c2) diagonal terms only for TDI_AET / AE.

CUDA_DEVICE
inline void gbfd_accumulate_ll(double *d_h_acc, double *h_h_acc,
                               cmplx *tdi_chan, int N_sparse, int nchannels,
                               FDDomain *fd, int kf0,
                               int data_index, int noise_index, int tdi_type,
                               double tau_d_h, double tau_h_h)
{
    // tau_*: previous accumulator values to add into (so caller can pre-zero
    // its registers and pass them in).  We simply accumulate per (c1,c2).
    double dh = tau_d_h;
    double hh = tau_h_h;

    const int N = N_sparse;
    const int C = nchannels;

    if (tdi_type == TDI_XYZ)
    {
        // cross-channel 3x3 inv-covariance
        for (int m = THREAD_START_X; m < N; m += BLOCK_INCR_X)
        {
            int k = gbfd_dense_bin(m, N, kf0);
            if (!fd->in_band(k)) continue;
            for (int c1 = 0; c1 < C; ++c1)
            {
                cmplx d_c1 = fd->get_data(k, c1, data_index);
                for (int c2 = 0; c2 < C; ++c2)
                {
                    cmplx h_c2 = tdi_chan[c2 * N + m];
                    double invc = fd->get_invC_cross(k, c1, c2, noise_index);
                    cmplx prod_dh = gcmplx::conj(d_c1) * h_c2;
                    cmplx prod_hh =
                        gcmplx::conj(tdi_chan[c1 * N + m]) * h_c2;
                    dh += prod_dh.real() * invc;
                    hh += prod_hh.real() * invc;
                }
            }
        }
    }
    else
    {
        // TDI_AET (3 diag) or TDI_AE (2 diag): diagonal inv-covariance.
        int Cd = (tdi_type == TDI_AE) ? 2 : C;
        for (int m = THREAD_START_X; m < N; m += BLOCK_INCR_X)
        {
            int k = gbfd_dense_bin(m, N, kf0);
            if (!fd->in_band(k)) continue;
            for (int c = 0; c < Cd; ++c)
            {
                cmplx d_c = fd->get_data(k, c, data_index);
                cmplx h_c = tdi_chan[c * N + m];
                double invc = fd->get_invC_diag(k, c, noise_index);
                cmplx prod_dh = gcmplx::conj(d_c) * h_c;
                double mag_h2 = h_c.real() * h_c.real()
                               + h_c.imag() * h_c.imag();
                dh += prod_dh.real() * invc;
                hh += mag_h2 * invc;
            }
        }
    }

    *d_h_acc = dh;
    *h_h_acc = hh;
}

#ifdef __CUDACC__
CUDA_KERNEL
void gb_fd_get_ll_kernel(double *d_h_out, double *h_h_out,
    GBTDIonTheFly *tdi_on_fly_handle, FDDomain *fd,
    double *params, int *data_index_all, int *noise_index_all,
    double t_start, double Tobs,
    int N, int num_bin, int n_params, int nchannels, int log2N, int tdi_type,
    double tukey_alpha, double edge_frac)
{
    extern CUDA_SHARED char shared_mem[];
    CUDA_SHARED double d_h_tmp[NUM_THREADS_HERE];
    CUDA_SHARED double h_h_tmp[NUM_THREADS_HERE];

    GBTDIonTheFly tof(tdi_on_fly_handle->orbits, tdi_on_fly_handle->tdi_config,
                      tdi_on_fly_handle->T, tdi_on_fly_handle->t_ref);

    for (int bin_i = BLOCK_START_X; bin_i < num_bin; bin_i += GRID_INCR_X)
    {
        for (int i = THREAD_START_X; i < NUM_THREADS_HERE; i += BLOCK_INCR_X)
        {
            d_h_tmp[i] = 0.0;
            h_h_tmp[i] = 0.0;
        }
        CUDA_SYNC_THREADS;

        cmplx *tdi_chan = NULL;
        int    kf0      = 0;
        double f0g      = 0.0;
        double dts      = 0.0;
        gbfd_build_one_source(&tof, (void*) shared_mem, params, t_start, Tobs,
                              N, nchannels, n_params, bin_i, log2N,
                              &tdi_chan, &kf0, &f0g, &dts, tukey_alpha, edge_frac);

        double dh_local = 0.0, hh_local = 0.0;
        gbfd_accumulate_ll(&dh_local, &hh_local, tdi_chan, N, nchannels,
                           fd, kf0,
                           data_index_all[bin_i], noise_index_all[bin_i],
                           tdi_type, 0.0, 0.0);

        int tid = threadIdx.x;
        d_h_tmp[tid] = dh_local;
        h_h_tmp[tid] = hh_local;
        CUDA_SYNC_THREADS;

        double dh_sum = block_reduce(d_h_tmp);
        double hh_sum = block_reduce(h_h_tmp);
        if (THREAD_ZERO)
        {
            d_h_out[bin_i] = 4.0 * fd->df * dh_sum;
            h_h_out[bin_i] = 4.0 * fd->df * hh_sum;
        }
        CUDA_SYNC_THREADS;
    }
}
#endif

void GBComputationGroup::gb_fd_get_ll_wrap(double *d_h_out, double *h_h_out,
    Orbits* orbits, TDIConfig *tdi_config, FDDomain *fd,
    double *params_all, int *data_index_all, int *noise_index_all,
    int num_bin, int nparams, double T, double t_start, double t_ref,
    int N_sparse, int nchannels, int tdi_type, double tukey_alpha, double edge_frac)
{
    int log2N = 0;
    {
        int m = N_sparse;
        while ((m & 1) == 0 && m > 1) { m >>= 1; ++log2N; }
#ifndef __CUDACC__
        if (m != 1) {
            throw std::invalid_argument(
                "gb_fd_get_ll_wrap: N_sparse must be a power of two.");
        }
#endif
    }

#ifdef __CUDACC__
    GBTDIonTheFly *gb_host = new GBTDIonTheFly(orbits, tdi_config, T, t_ref);
    Orbits *d_orbits;
    cudaMalloc(&d_orbits, sizeof(Orbits));
    gpuErrchk(cudaMemcpy(d_orbits, orbits, sizeof(Orbits), cudaMemcpyHostToDevice));
    TDIConfig *d_tdi_config;
    cudaMalloc(&d_tdi_config, sizeof(TDIConfig));
    gpuErrchk(cudaMemcpy(d_tdi_config, tdi_config, sizeof(TDIConfig), cudaMemcpyHostToDevice));
    gb_host->orbits = d_orbits;
    gb_host->tdi_config = d_tdi_config;
    GBTDIonTheFly *d_gb;
    cudaMalloc(&d_gb, sizeof(GBTDIonTheFly));
    gpuErrchk(cudaMemcpy(d_gb, gb_host, sizeof(GBTDIonTheFly), cudaMemcpyHostToDevice));
    FDDomain *d_fd;
    cudaMalloc(&d_fd, sizeof(FDDomain));
    gpuErrchk(cudaMemcpy(d_fd, fd, sizeof(FDDomain), cudaMemcpyHostToDevice));

    int shared_bytes = gb_host->get_gb_fd_buffer_size(N_sparse, nchannels);
    if (shared_bytes > 48 * 1024) {
        cudaFuncSetAttribute(gb_fd_get_ll_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize, shared_bytes);
    }
    gb_fd_get_ll_kernel<<<num_bin, NUM_THREADS_HERE, shared_bytes>>>(
        d_h_out, h_h_out, d_gb, d_fd,
        params_all, data_index_all, noise_index_all,
        t_start, T, N_sparse, num_bin, nparams, nchannels, log2N, tdi_type,
        tukey_alpha, edge_frac);
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());
    cudaFree(d_orbits);
    cudaFree(d_tdi_config);
    cudaFree(d_gb);
    cudaFree(d_fd);
    delete gb_host;
#else
    GBTDIonTheFly tof(orbits, tdi_config, T, t_ref);
    int shared_bytes = tof.get_gb_fd_buffer_size(N_sparse, nchannels);
    char *shared_mem = new char[shared_bytes];
    for (int bin_i = 0; bin_i < num_bin; ++bin_i)
    {
        cmplx *tdi_chan = NULL;
        int    kf0      = 0;
        double f0g      = 0.0;
        double dts      = 0.0;
        gbfd_build_one_source(&tof, (void*) shared_mem,
                              params_all, t_start, T,
                              N_sparse, nchannels, nparams, bin_i, log2N,
                              &tdi_chan, &kf0, &f0g, &dts, tukey_alpha, edge_frac);

        double dh = 0.0, hh = 0.0;
        gbfd_accumulate_ll(&dh, &hh, tdi_chan, N_sparse, nchannels, fd, kf0,
                           data_index_all[bin_i], noise_index_all[bin_i],
                           tdi_type, 0.0, 0.0);
        d_h_out[bin_i] = 4.0 * fd->df * dh;
        h_h_out[bin_i] = 4.0 * fd->df * hh;
    }
    delete[] shared_mem;
#endif
}

// fill_global: add factor_i * h_i to a global FD template buffer
// of shape (num_data, nchannels, n_rfft).  The buffer is addressed via
// data_index_all[bin_i]; multiple bins routed to the same data_index are
// accumulated.  In the GPU build the writes go through atomicAdd to handle
// overlapping sources.
#ifdef __CUDACC__
CUDA_KERNEL
void gb_fd_fill_global_kernel(cmplx *template_fill,
    GBTDIonTheFly *tdi_on_fly_handle, FDDomain *fd,
    double *params, int *data_index_all, double *factors_all,
    double t_start, double Tobs,
    int N, int num_bin, int n_params, int nchannels, int log2N,
    double tukey_alpha, double edge_frac)
{
    extern CUDA_SHARED char shared_mem[];
    GBTDIonTheFly tof(tdi_on_fly_handle->orbits, tdi_on_fly_handle->tdi_config,
                      tdi_on_fly_handle->T, tdi_on_fly_handle->t_ref);
    for (int bin_i = BLOCK_START_X; bin_i < num_bin; bin_i += GRID_INCR_X)
    {
        cmplx *tdi_chan = NULL;
        int    kf0      = 0;
        double f0g      = 0.0;
        double dts      = 0.0;
        gbfd_build_one_source(&tof, (void*) shared_mem, params, t_start, Tobs,
                              N, nchannels, n_params, bin_i, log2N,
                              &tdi_chan, &kf0, &f0g, &dts, tukey_alpha, edge_frac);

        int data_index = data_index_all[bin_i];
        double factor  = factors_all[bin_i];
        for (int m = THREAD_START_X; m < N; m += BLOCK_INCR_X)
        {
            int k = gbfd_dense_bin(m, N, kf0);
            if (!fd->in_band(k)) continue;
            for (int c = 0; c < nchannels; ++c)
            {
                cmplx v = tdi_chan[c * N + m];
                size_t idx = (size_t) data_index * nchannels * fd->n_rfft
                             + (size_t) c * fd->n_rfft + k;
                double re = factor * v.real();
                double im = factor * v.imag();
                atomicAdd(((double*)&template_fill[idx]) + 0, re);
                atomicAdd(((double*)&template_fill[idx]) + 1, im);
            }
        }
        CUDA_SYNC_THREADS;
    }
}
#endif

void GBComputationGroup::gb_fd_fill_global_wrap(cmplx *template_fill,
    Orbits* orbits, TDIConfig *tdi_config, FDDomain *fd,
    double *params_all, int *data_index_all, double *factors_all,
    int num_bin, int nparams, double T, double t_start, double t_ref,
    int N_sparse, int nchannels, double tukey_alpha, double edge_frac)
{
    int log2N = 0;
    {
        int m = N_sparse;
        while ((m & 1) == 0 && m > 1) { m >>= 1; ++log2N; }
#ifndef __CUDACC__
        if (m != 1) {
            throw std::invalid_argument(
                "gb_fd_fill_global_wrap: N_sparse must be a power of two.");
        }
#endif
    }

#ifdef __CUDACC__
    GBTDIonTheFly *gb_host = new GBTDIonTheFly(orbits, tdi_config, T, t_ref);
    Orbits *d_orbits;
    cudaMalloc(&d_orbits, sizeof(Orbits));
    gpuErrchk(cudaMemcpy(d_orbits, orbits, sizeof(Orbits), cudaMemcpyHostToDevice));
    TDIConfig *d_tdi_config;
    cudaMalloc(&d_tdi_config, sizeof(TDIConfig));
    gpuErrchk(cudaMemcpy(d_tdi_config, tdi_config, sizeof(TDIConfig), cudaMemcpyHostToDevice));
    gb_host->orbits = d_orbits;
    gb_host->tdi_config = d_tdi_config;
    GBTDIonTheFly *d_gb;
    cudaMalloc(&d_gb, sizeof(GBTDIonTheFly));
    gpuErrchk(cudaMemcpy(d_gb, gb_host, sizeof(GBTDIonTheFly), cudaMemcpyHostToDevice));
    FDDomain *d_fd;
    cudaMalloc(&d_fd, sizeof(FDDomain));
    gpuErrchk(cudaMemcpy(d_fd, fd, sizeof(FDDomain), cudaMemcpyHostToDevice));

    int shared_bytes = gb_host->get_gb_fd_buffer_size(N_sparse, nchannels);
    if (shared_bytes > 48 * 1024) {
        cudaFuncSetAttribute(gb_fd_fill_global_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize, shared_bytes);
    }
    gb_fd_fill_global_kernel<<<num_bin, NUM_THREADS_HERE, shared_bytes>>>(
        template_fill, d_gb, d_fd, params_all, data_index_all, factors_all,
        t_start, T, N_sparse, num_bin, nparams, nchannels, log2N, tukey_alpha,
        edge_frac);
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());
    cudaFree(d_orbits);
    cudaFree(d_tdi_config);
    cudaFree(d_gb);
    cudaFree(d_fd);
    delete gb_host;
#else
    GBTDIonTheFly tof(orbits, tdi_config, T, t_ref);
    int shared_bytes = tof.get_gb_fd_buffer_size(N_sparse, nchannels);
    char *shared_mem = new char[shared_bytes];
    for (int bin_i = 0; bin_i < num_bin; ++bin_i)
    {
        cmplx *tdi_chan = NULL;
        int    kf0      = 0;
        double f0g      = 0.0;
        double dts      = 0.0;
        gbfd_build_one_source(&tof, (void*) shared_mem,
                              params_all, t_start, T,
                              N_sparse, nchannels, nparams, bin_i, log2N,
                              &tdi_chan, &kf0, &f0g, &dts, tukey_alpha, edge_frac);

        int data_index = data_index_all[bin_i];
        double factor  = factors_all[bin_i];
        for (int m = 0; m < N_sparse; ++m)
        {
            int k = gbfd_dense_bin(m, N_sparse, kf0);
            if (!fd->in_band(k)) continue;
            for (int c = 0; c < nchannels; ++c)
            {
                size_t idx = (size_t) data_index * nchannels * fd->n_rfft
                             + (size_t) c * fd->n_rfft + k;
                template_fill[idx] = template_fill[idx]
                    + cmplx(factor * tdi_chan[c * N_sparse + m].real(),
                            factor * tdi_chan[c * N_sparse + m].imag());
            }
        }
    }
    delete[] shared_mem;
#endif
}

// swap_ll: returns the five swap accumulators in lisatools convention,
//    (d|h_add), (d|h_rem), (h_add|h_add), (h_rem|h_rem), (h_add|h_rem).
//
// To stay GPU-friendly (one block per source), we run the heterodyne FD for
// the add and remove sources back-to-back in shared memory; the second pass
// overwrites the first's slow-signal buffer, so we accumulate (h_add|*) into
// per-thread registers before the second pass.
//
// For now the implementation is straightforward sequential per side, which
// keeps the math identical to the WDM swap convention; further unification
// (single-pass dual heterodyne) is a follow-up.
void GBComputationGroup::gb_fd_swap_ll_wrap(
    double *d_h_add_out, double *d_h_remove_out,
    double *add_add_out, double *remove_remove_out, double *add_remove_out,
    Orbits* orbits, TDIConfig *tdi_config, FDDomain *fd,
    double *params_add_all, double *params_remove_all,
    int *data_index_all, int *noise_index_all,
    int num_bin, int nparams, double T, double t_start, double t_ref,
    int N_sparse, int nchannels, int tdi_type, double tukey_alpha, double edge_frac)
{
    // Reuse get_ll for the diagonal-in-source accumulators, then explicitly
    // form the cross term (h_add | h_remove) per source.
    GBTDIonTheFly tof(orbits, tdi_config, T, t_ref);
    int log2N = 0;
    {
        int m = N_sparse;
        while ((m & 1) == 0 && m > 1) { m >>= 1; ++log2N; }
#ifndef __CUDACC__
        if (m != 1) {
            throw std::invalid_argument(
                "gb_fd_swap_ll_wrap: N_sparse must be a power of two.");
        }
#endif
    }

#ifdef __CUDACC__
    // GPU path: TODO -- a dedicated dual-source kernel.  The CPU build is
    // fully wired below and the GPU/CPU outputs match at the math level so
    // adding a kernel is mechanical; deferred to keep this commit focused.
    (void) tof; (void) log2N;
    (void) d_h_add_out; (void) d_h_remove_out;
    (void) add_add_out; (void) remove_remove_out; (void) add_remove_out;
    (void) orbits; (void) tdi_config; (void) fd;
    (void) params_add_all; (void) params_remove_all;
    (void) data_index_all; (void) noise_index_all;
    (void) num_bin; (void) nparams; (void) T;
    (void) t_start; (void) t_ref; (void) N_sparse;
    (void) nchannels; (void) tdi_type; (void) tukey_alpha; (void) edge_frac;
    printf("gb_fd_swap_ll_wrap GPU path not implemented yet.\n");
#else
    int shared_bytes = tof.get_gb_fd_buffer_size(N_sparse, nchannels);
    char *shared_mem_a = new char[shared_bytes];
    char *shared_mem_b = new char[shared_bytes];
    // Per-source loop.
    for (int bin_i = 0; bin_i < num_bin; ++bin_i)
    {
        cmplx *h_add = NULL;
        int    kf0_a = 0;
        double f0g_a = 0.0;
        double dts_a = 0.0;
        gbfd_build_one_source(&tof, (void*) shared_mem_a, params_add_all,
                              t_start, T, N_sparse, nchannels, nparams,
                              bin_i, log2N, &h_add, &kf0_a, &f0g_a, &dts_a, tukey_alpha, edge_frac);
        // (d|h_add), (h_add|h_add)
        double dh_a = 0.0, hh_aa = 0.0;
        gbfd_accumulate_ll(&dh_a, &hh_aa, h_add, N_sparse, nchannels, fd,
                           kf0_a, data_index_all[bin_i],
                           noise_index_all[bin_i], tdi_type, 0.0, 0.0);
        // h_add lives in shared_mem_a; build h_remove in a separate buffer.
        cmplx *h_rem = NULL;
        int    kf0_r = 0;
        double f0g_r = 0.0;
        double dts_r = 0.0;
        gbfd_build_one_source(&tof, (void*) shared_mem_b, params_remove_all,
                              t_start, T, N_sparse, nchannels, nparams,
                              bin_i, log2N, &h_rem, &kf0_r, &f0g_r, &dts_r, tukey_alpha, edge_frac);
        double dh_r = 0.0, hh_rr = 0.0;
        gbfd_accumulate_ll(&dh_r, &hh_rr, h_rem, N_sparse, nchannels, fd,
                           kf0_r, data_index_all[bin_i],
                           noise_index_all[bin_i], tdi_type, 0.0, 0.0);
        // Cross term (h_add | h_remove): direct sum over dense bins shared
        // by the two sparse supports.  Each side knows its own (kf0, m_signed)
        // mapping; iterate over h_add bins and look up the matching h_remove
        // bin by absolute dense bin (k - kf0_r) modulo N_sparse.
        double hh_ar = 0.0;
        const int N = N_sparse;
        if (tdi_type == TDI_XYZ)
        {
            for (int m = 0; m < N; ++m)
            {
                int k = gbfd_dense_bin(m, N, kf0_a);
                if (!fd->in_band(k)) continue;
                // matching m on remove side: (k - kf0_r) mod N
                int mr_signed = k - kf0_r;
                int mr = ((mr_signed % N) + N) % N;
                int kr = gbfd_dense_bin(mr, N, kf0_r);
                if (kr != k) continue;  // remove-source slot does not cover k
                for (int c1 = 0; c1 < nchannels; ++c1)
                {
                    for (int c2 = 0; c2 < nchannels; ++c2)
                    {
                        cmplx ha = h_add[c1 * N + m];
                        cmplx hr = h_rem[c2 * N + mr];
                        double invc = fd->get_invC_cross(
                            k, c1, c2, noise_index_all[bin_i]);
                        cmplx prod = gcmplx::conj(ha) * hr;
                        hh_ar += prod.real() * invc;
                    }
                }
            }
        }
        else
        {
            int Cd = (tdi_type == TDI_AE) ? 2 : nchannels;
            for (int m = 0; m < N; ++m)
            {
                int k = gbfd_dense_bin(m, N, kf0_a);
                if (!fd->in_band(k)) continue;
                int mr_signed = k - kf0_r;
                int mr = ((mr_signed % N) + N) % N;
                int kr = gbfd_dense_bin(mr, N, kf0_r);
                if (kr != k) continue;
                for (int c = 0; c < Cd; ++c)
                {
                    cmplx ha = h_add[c * N + m];
                    cmplx hr = h_rem[c * N + mr];
                    double invc = fd->get_invC_diag(
                        k, c, noise_index_all[bin_i]);
                    cmplx prod = gcmplx::conj(ha) * hr;
                    hh_ar += prod.real() * invc;
                }
            }
        }

        double k4df = 4.0 * fd->df;
        d_h_add_out[bin_i]      = k4df * dh_a;
        d_h_remove_out[bin_i]   = k4df * dh_r;
        add_add_out[bin_i]      = k4df * hh_aa;
        remove_remove_out[bin_i] = k4df * hh_rr;
        add_remove_out[bin_i]   = k4df * hh_ar;
    }
    delete[] shared_mem_a;
    delete[] shared_mem_b;
#endif
}


// =============================================================================
//  FD-domain chain-rule gradients of gb_fd_get_ll / gb_fd_swap_ll.
//
//  Mirrors the WDM chain-rule gradient kernels: per parameter k we perturb
//  theta_k by +/- eps_k, rebuild the per-source heterodyne FD piece, and
//  accumulate the inner product (d - h_C | dh/dtheta_k) for get_ll, or the
//  post-swap analog (d - h_add_C + h_rem_C | dh_{add,rem}/dtheta_{add,rem}_k)
//  for swap_ll.  The parameter derivative is central FD,
//
//      dh/dtheta_k(p) = (h_+ - h_-) / (2 eps_k).
//
//  Each perturbed signal has its own snapped carrier kf0_{+,-}; we match it
//  back to the central side by absolute dense rfft bin (exactly the trick
//  used in gb_fd_swap_ll_wrap's cross term).  The matching is robust to the
//  rare case where +/- eps_f0 flips the rounding of f0 -> kf0.
//
//  The CPU build is fully wired; the GPU paths follow the same status as
//  gb_fd_swap_ll_wrap (printf-and-return placeholder).
// =============================================================================

// Per-pair gradient accumulator: returns the partial inner product
//    Re sum_{c1,c2} sum_m  conj(r_C[c1, m_C(k_pert(m))]) * h_pert[c2, m]
//                          * invC[c1,c2,k_pert(m)]
// iterated over the perturbed side's sparse bins.  The "residual" r_C is
// built from the central-side stash(es) by absolute-dense-bin reverse lookup:
//    get_ll: r_C[c, k] = d[c, k] - h_add_C[c, m_add(k)]    (h_rem_C = NULL)
//    swap:   r_C[c, k] = d[c, k] - h_add_C[c, m_add(k)] + h_rem_C[c, m_rem(k)]
// Missing coverage on a central side contributes 0 for that side (residual
// reduces to the remaining terms).
//
// All inputs are in FFT order (length N per channel); kf0_* are the
// dense-rfft-bin snaps that gbfd_build_one_source returned for each signal.
CUDA_DEVICE
inline double gbfd_grad_one_sided_partial(
    cmplx *h_pert, int kf0_pert,
    cmplx *h_add_C, int kf0_add_C,
    cmplx *h_rem_C, int kf0_rem_C,
    int N, int nchannels, FDDomain *fd,
    int data_index, int noise_index, int tdi_type)
{
    double acc = 0.0;
    const int Cd = (tdi_type == TDI_AE) ? 2 : nchannels;
    for (int mp = 0; mp < N; ++mp)
    {
        int kk = gbfd_dense_bin(mp, N, kf0_pert);
        if (!fd->in_band(kk)) continue;

        int ma_signed = kk - kf0_add_C;
        int ma = ((ma_signed % N) + N) % N;
        bool ka_match = (gbfd_dense_bin(ma, N, kf0_add_C) == kk);

        int mr = 0;
        bool kr_match = false;
        if (h_rem_C != NULL)
        {
            int mr_signed = kk - kf0_rem_C;
            mr = ((mr_signed % N) + N) % N;
            kr_match = (gbfd_dense_bin(mr, N, kf0_rem_C) == kk);
        }

        if (tdi_type == TDI_XYZ)
        {
            for (int c1 = 0; c1 < 3; ++c1)
            {
                cmplx d_c1 = fd->get_data(kk, c1, data_index);
                cmplx ha = ka_match ? h_add_C[c1 * N + ma] : cmplx(0., 0.);
                cmplx hr = (h_rem_C != NULL && kr_match)
                                ? h_rem_C[c1 * N + mr] : cmplx(0., 0.);
                cmplx r_c1 = d_c1 - ha + hr;
                for (int c2 = 0; c2 < 3; ++c2)
                {
                    cmplx hp = h_pert[c2 * N + mp];
                    double invc = fd->get_invC_cross(kk, c1, c2, noise_index);
                    cmplx prod = gcmplx::conj(r_c1) * hp;
                    acc += prod.real() * invc;
                }
            }
        }
        else
        {
            for (int c = 0; c < Cd; ++c)
            {
                cmplx d_c = fd->get_data(kk, c, data_index);
                cmplx ha = ka_match ? h_add_C[c * N + ma] : cmplx(0., 0.);
                cmplx hr = (h_rem_C != NULL && kr_match)
                                ? h_rem_C[c * N + mr] : cmplx(0., 0.);
                cmplx r_c = d_c - ha + hr;
                cmplx hp = h_pert[c * N + mp];
                double invc = fd->get_invC_diag(kk, c, noise_index);
                cmplx prod = gcmplx::conj(r_c) * hp;
                acc += prod.real() * invc;
            }
        }
    }
    return acc;
}


void GBComputationGroup::gb_fd_get_ll_grad_wrap(double *grad_out,
    Orbits* orbits, TDIConfig *tdi_config, FDDomain *fd,
    double *params_all, int *data_index_all, int *noise_index_all,
    double *param_eps,
    int num_bin, int nparams, double T, double t_start, double t_ref,
    int N_sparse, int nchannels, int tdi_type)
{
    int log2N = 0;
    {
        int m = N_sparse;
        while ((m & 1) == 0 && m > 1) { m >>= 1; ++log2N; }
#ifndef __CUDACC__
        if (m != 1) {
            throw std::invalid_argument(
                "gb_fd_get_ll_grad_wrap: N_sparse must be a power of two.");
        }
#endif
    }

#ifdef __CUDACC__
    (void) grad_out; (void) orbits; (void) tdi_config; (void) fd;
    (void) params_all; (void) data_index_all; (void) noise_index_all;
    (void) param_eps;
    (void) num_bin; (void) nparams; (void) T;
    (void) t_start; (void) t_ref; (void) N_sparse;
    (void) nchannels; (void) tdi_type; (void) log2N;
    printf("gb_fd_get_ll_grad_wrap GPU path not implemented yet.\n");
#else
    GBTDIonTheFly tof(orbits, tdi_config, T, t_ref);
    int shared_bytes = tof.get_gb_fd_buffer_size(N_sparse, nchannels);
    char  *scratch       = new char[shared_bytes];
    cmplx *central_stash = new cmplx[(size_t) nchannels * N_sparse];

    double params_priv[N_PARAMS_MAX];

    for (int bin_i = 0; bin_i < num_bin; ++bin_i)
    {
        for (int i = 0; i < nparams; ++i)
            params_priv[i] = params_all[bin_i * nparams + i];

        // Central build.
        cmplx *h_C_shared = NULL;
        int    kf0_C = 0;
        double f0g_C = 0.0, dts_C = 0.0;
        gbfd_build_one_source(&tof, (void*) scratch, params_priv,
                              t_start, T, N_sparse, nchannels, nparams,
                              /*bin_i=*/0, log2N,
                              &h_C_shared, &kf0_C, &f0g_C, &dts_C, 0.0, 0.0);
        // The scratch's tdi_chan slab will be overwritten by perturbed builds
        // below, so stash the central signal in our own buffer.
        for (size_t idx = 0;
             idx < (size_t) nchannels * (size_t) N_sparse; ++idx)
            central_stash[idx] = h_C_shared[idx];

        int data_index  = data_index_all[bin_i];
        int noise_index = noise_index_all[bin_i];

        for (int k = 0; k < nparams; ++k)
        {
            double eps_k = param_eps[k];
            if (eps_k <= 0.0)
            {
                grad_out[bin_i * nparams + k] = 0.0;
                continue;
            }
            double saved = params_priv[k];
            const double inv_2eps = 1.0 / (2.0 * eps_k);

            // +eps build (overwrites scratch's tdi_chan slab).
            params_priv[k] = saved + eps_k;
            cmplx *h_P_shared = NULL;
            int    kf0_P = 0;
            double f0g_P = 0.0, dts_P = 0.0;
            gbfd_build_one_source(&tof, (void*) scratch, params_priv,
                                  t_start, T, N_sparse, nchannels, nparams,
                                  0, log2N,
                                  &h_P_shared, &kf0_P, &f0g_P, &dts_P, 0.0, 0.0);
            double acc_p = gbfd_grad_one_sided_partial(
                h_P_shared, kf0_P,
                central_stash, kf0_C,
                /*h_rem_C=*/NULL, /*kf0_rem_C=*/0,
                N_sparse, nchannels, fd,
                data_index, noise_index, tdi_type);

            // -eps build (overwrites again).
            params_priv[k] = saved - eps_k;
            cmplx *h_M_shared = NULL;
            int    kf0_M = 0;
            double f0g_M = 0.0, dts_M = 0.0;
            gbfd_build_one_source(&tof, (void*) scratch, params_priv,
                                  t_start, T, N_sparse, nchannels, nparams,
                                  0, log2N,
                                  &h_M_shared, &kf0_M, &f0g_M, &dts_M, 0.0, 0.0);
            double acc_m = gbfd_grad_one_sided_partial(
                h_M_shared, kf0_M,
                central_stash, kf0_C,
                /*h_rem_C=*/NULL, /*kf0_rem_C=*/0,
                N_sparse, nchannels, fd,
                data_index, noise_index, tdi_type);

            params_priv[k] = saved;
            grad_out[bin_i * nparams + k] =
                4.0 * fd->df * (acc_p - acc_m) * inv_2eps;
        }
    }

    delete[] central_stash;
    delete[] scratch;
#endif
}


void GBComputationGroup::gb_fd_swap_ll_grad_wrap(
    double *grad_add_out, double *grad_remove_out,
    Orbits* orbits, TDIConfig *tdi_config, FDDomain *fd,
    double *params_add_all, double *params_remove_all,
    int *data_index_all, int *noise_index_all,
    double *param_eps_add, double *param_eps_remove,
    int num_bin, int nparams, double T, double t_start, double t_ref,
    int N_sparse, int nchannels, int tdi_type)
{
    int log2N = 0;
    {
        int m = N_sparse;
        while ((m & 1) == 0 && m > 1) { m >>= 1; ++log2N; }
#ifndef __CUDACC__
        if (m != 1) {
            throw std::invalid_argument(
                "gb_fd_swap_ll_grad_wrap: N_sparse must be a power of two.");
        }
#endif
    }

#ifdef __CUDACC__
    (void) grad_add_out; (void) grad_remove_out;
    (void) orbits; (void) tdi_config; (void) fd;
    (void) params_add_all; (void) params_remove_all;
    (void) data_index_all; (void) noise_index_all;
    (void) param_eps_add; (void) param_eps_remove;
    (void) num_bin; (void) nparams; (void) T;
    (void) t_start; (void) t_ref; (void) N_sparse;
    (void) nchannels; (void) tdi_type; (void) log2N;
    printf("gb_fd_swap_ll_grad_wrap GPU path not implemented yet.\n");
#else
    GBTDIonTheFly tof(orbits, tdi_config, T, t_ref);
    int shared_bytes = tof.get_gb_fd_buffer_size(N_sparse, nchannels);
    char  *scratch    = new char[shared_bytes];
    cmplx *add_stash  = new cmplx[(size_t) nchannels * N_sparse];
    cmplx *rem_stash  = new cmplx[(size_t) nchannels * N_sparse];

    double params_add_priv[N_PARAMS_MAX];
    double params_rem_priv[N_PARAMS_MAX];

    for (int bin_i = 0; bin_i < num_bin; ++bin_i)
    {
        for (int i = 0; i < nparams; ++i)
        {
            params_add_priv[i] = params_add_all[bin_i * nparams + i];
            params_rem_priv[i] = params_remove_all[bin_i * nparams + i];
        }

        // Central builds for the add and remove sources.
        cmplx *h_addC_shared = NULL;
        int    kf0_addC = 0;
        double f0g_addC = 0.0, dts_addC = 0.0;
        gbfd_build_one_source(&tof, (void*) scratch, params_add_priv,
                              t_start, T, N_sparse, nchannels, nparams,
                              0, log2N,
                              &h_addC_shared, &kf0_addC, &f0g_addC, &dts_addC, 0.0, 0.0);
        for (size_t idx = 0;
             idx < (size_t) nchannels * (size_t) N_sparse; ++idx)
            add_stash[idx] = h_addC_shared[idx];

        cmplx *h_remC_shared = NULL;
        int    kf0_remC = 0;
        double f0g_remC = 0.0, dts_remC = 0.0;
        gbfd_build_one_source(&tof, (void*) scratch, params_rem_priv,
                              t_start, T, N_sparse, nchannels, nparams,
                              0, log2N,
                              &h_remC_shared, &kf0_remC, &f0g_remC, &dts_remC, 0.0, 0.0);
        for (size_t idx = 0;
             idx < (size_t) nchannels * (size_t) N_sparse; ++idx)
            rem_stash[idx] = h_remC_shared[idx];

        int data_index  = data_index_all[bin_i];
        int noise_index = noise_index_all[bin_i];

        // ------ add-side gradient: sign = +1, perturb the add params ------
        for (int k = 0; k < nparams; ++k)
        {
            double eps_k = param_eps_add[k];
            if (eps_k <= 0.0)
            {
                grad_add_out[bin_i * nparams + k] = 0.0;
                continue;
            }
            double saved = params_add_priv[k];
            const double inv_2eps = 1.0 / (2.0 * eps_k);

            params_add_priv[k] = saved + eps_k;
            cmplx *h_aP = NULL;
            int kf0_aP = 0; double f0g_aP = 0.0, dts_aP = 0.0;
            gbfd_build_one_source(&tof, (void*) scratch, params_add_priv,
                                  t_start, T, N_sparse, nchannels, nparams,
                                  0, log2N,
                                  &h_aP, &kf0_aP, &f0g_aP, &dts_aP, 0.0, 0.0);
            double acc_p = gbfd_grad_one_sided_partial(
                h_aP, kf0_aP,
                add_stash, kf0_addC,
                rem_stash, kf0_remC,
                N_sparse, nchannels, fd,
                data_index, noise_index, tdi_type);

            params_add_priv[k] = saved - eps_k;
            cmplx *h_aM = NULL;
            int kf0_aM = 0; double f0g_aM = 0.0, dts_aM = 0.0;
            gbfd_build_one_source(&tof, (void*) scratch, params_add_priv,
                                  t_start, T, N_sparse, nchannels, nparams,
                                  0, log2N,
                                  &h_aM, &kf0_aM, &f0g_aM, &dts_aM, 0.0, 0.0);
            double acc_m = gbfd_grad_one_sided_partial(
                h_aM, kf0_aM,
                add_stash, kf0_addC,
                rem_stash, kf0_remC,
                N_sparse, nchannels, fd,
                data_index, noise_index, tdi_type);

            params_add_priv[k] = saved;
            grad_add_out[bin_i * nparams + k] =
                +4.0 * fd->df * (acc_p - acc_m) * inv_2eps;
        }

        // ------ remove-side gradient: sign = -1, perturb the remove params ------
        for (int k = 0; k < nparams; ++k)
        {
            double eps_k = param_eps_remove[k];
            if (eps_k <= 0.0)
            {
                grad_remove_out[bin_i * nparams + k] = 0.0;
                continue;
            }
            double saved = params_rem_priv[k];
            const double inv_2eps = 1.0 / (2.0 * eps_k);

            params_rem_priv[k] = saved + eps_k;
            cmplx *h_rP = NULL;
            int kf0_rP = 0; double f0g_rP = 0.0, dts_rP = 0.0;
            gbfd_build_one_source(&tof, (void*) scratch, params_rem_priv,
                                  t_start, T, N_sparse, nchannels, nparams,
                                  0, log2N,
                                  &h_rP, &kf0_rP, &f0g_rP, &dts_rP, 0.0, 0.0);
            double acc_p = gbfd_grad_one_sided_partial(
                h_rP, kf0_rP,
                add_stash, kf0_addC,
                rem_stash, kf0_remC,
                N_sparse, nchannels, fd,
                data_index, noise_index, tdi_type);

            params_rem_priv[k] = saved - eps_k;
            cmplx *h_rM = NULL;
            int kf0_rM = 0; double f0g_rM = 0.0, dts_rM = 0.0;
            gbfd_build_one_source(&tof, (void*) scratch, params_rem_priv,
                                  t_start, T, N_sparse, nchannels, nparams,
                                  0, log2N,
                                  &h_rM, &kf0_rM, &f0g_rM, &dts_rM, 0.0, 0.0);
            double acc_m = gbfd_grad_one_sided_partial(
                h_rM, kf0_rM,
                add_stash, kf0_addC,
                rem_stash, kf0_remC,
                N_sparse, nchannels, fd,
                data_index, noise_index, tdi_type);

            params_rem_priv[k] = saved;
            grad_remove_out[bin_i * nparams + k] =
                -4.0 * fd->df * (acc_p - acc_m) * inv_2eps;
        }
    }

    delete[] add_stash;
    delete[] rem_stash;
    delete[] scratch;
#endif
}


// ============================================================================
// Phase 3L.7f.3 (2026-06-04): GB chunked-WDM-heterodyne wrap methods.
// Carved from lisa-on-gpu's TDIonTheFly.cu:6727-6854.
//
// Each method instantiates LAT's templated
// `wdm_het_*_impl<SourceT>(...)` (defined in
// `lisatools/cutils/lat_chunked_het_kernels.hh`, Phase 3L.7a slice 3)
// against `<GBTDIonTheFly>` and dispatches through the Python-facing
// GBComputationGroup surface.
// ============================================================================

// ---- GB-flavored wrappers --------------------------------------------------
void GBComputationGroup::gb_wdm_het_fill_global_wrap(
    double *template_fill, Orbits *orbits, TDIConfig *tdi_config,
    WDMSettings *wdm_settings,
    double *params_all, double *factors_all,
    double *chunk_t_starts, int *chunk_keep_lo, int *chunk_keep_hi,
    int *chunk_n_global_offset, double *wdm_window,
    int n_chunks, int num_bin, int nparams,
    int Nt_sub, int log2_Nt_sub,
    int N_sparse, int log2_N_sparse,
    int nchannels, int n_rfft_chunk,
    double T_chunk, double dt, double T, double t_ref,
    double tukey_alpha, int grid_dim, int N_cp_sig, int N_cp_orbit,
    int m_band_half_width, bool active_band)
{
    wdm_het_fill_global_impl<GBTDIonTheFly>(
        template_fill, orbits, tdi_config,
        wdm_settings,
        params_all, factors_all,
        chunk_t_starts, chunk_keep_lo, chunk_keep_hi, chunk_n_global_offset,
        wdm_window, n_chunks, num_bin, nparams,
        Nt_sub, log2_Nt_sub, N_sparse, log2_N_sparse,
        nchannels, n_rfft_chunk, T_chunk, dt, T, t_ref, tukey_alpha,
        grid_dim, N_cp_sig, N_cp_orbit, m_band_half_width, active_band);
}

void GBComputationGroup::gb_wdm_het_get_ll_wrap(
    double *d_h_out, double *h_h_out, Orbits *orbits, TDIConfig *tdi_config,
    WDMSettings *wdm_settings,
    double *params_all, int *data_index_all, int *noise_index_all,
    double *chunk_t_starts, int *chunk_keep_lo, int *chunk_keep_hi,
    int *chunk_n_global_offset, double *wdm_window,
    double *data_d, double *invC,
    int n_chunks, int num_bin, int nparams,
    int Nt_sub, int log2_Nt_sub,
    int N_sparse, int log2_N_sparse,
    int nchannels, int n_rfft_chunk,
    double T_chunk, double dt, double T, double t_ref, int tdi_type,
    double tukey_alpha, int grid_dim, int N_cp_sig, int N_cp_orbit,
    int *binary_perm, int *group_starts, int *group_ends,
    int *group_m_lo, int *group_m_hi, int n_groups,
    int m_band_half_width)
{
    wdm_het_get_ll_impl<GBTDIonTheFly>(
        d_h_out, h_h_out, orbits, tdi_config,
        wdm_settings,
        params_all,
        data_index_all, noise_index_all,
        chunk_t_starts, chunk_keep_lo, chunk_keep_hi, chunk_n_global_offset,
        wdm_window, data_d, invC, n_chunks, num_bin, nparams,
        Nt_sub, log2_Nt_sub, N_sparse, log2_N_sparse,
        nchannels, n_rfft_chunk,
        T_chunk, dt, T, t_ref, tdi_type, tukey_alpha,
        grid_dim, N_cp_sig, N_cp_orbit,
        binary_perm, group_starts, group_ends,
        group_m_lo, group_m_hi, n_groups, m_band_half_width);
}

// ---- STFT/Fresnel GB likelihood wraps (get_ll + fill_global + swap_ll) ------
// Instantiate LAT's templated stft_*_impl<SourceT> (lisatools/cutils/
// lat_stft_kernels.hh) against <GBTDIonTheFly>. The STFT/Fresnel device
// primitives (get_freq_index / add_ip_contrib / get_amp_phase /
// get_fourier_value / diff_comp) live in lisatools/cutils/domains.{hpp,cu}.
void GBComputationGroup::gb_stft_get_ll_wrap(
    cmplx *d_h_out, cmplx *h_h_out,
    Orbits *orbits, TDIConfig *tdi_config,
    STFTFresnel *fresnel, STFTDomain *stft,
    double *params_all, int *data_index_all, int *noise_index_all,
    int num_bin, int nparams, double T, double t_ref,
    int n_side_bins, double window_factor, bool freq_from_tdi_phase)
{
    stft_get_ll_impl<GBTDIonTheFly>(
        d_h_out, h_h_out, orbits, tdi_config, fresnel, stft,
        params_all, data_index_all, noise_index_all,
        num_bin, nparams, T, t_ref, n_side_bins, window_factor,
        freq_from_tdi_phase);
}

void GBComputationGroup::gb_stft_get_ll_fft_wrap(
    cmplx *d_h_out, cmplx *h_h_out,
    Orbits *orbits, TDIConfig *tdi_config,
    STFTFresnel *fresnel, STFTDomain *stft,
    double *params_all, int *data_index_all, int *noise_index_all,
    int num_bin, int nparams, double T, double t_ref,
    int n_side_bins, int n_sub, int n_cp_orbit, double window_factor, bool freq_from_tdi_phase)
{
    stft_get_ll_fft_impl<GBTDIonTheFly>(
        d_h_out, h_h_out, orbits, tdi_config, fresnel, stft,
        params_all, data_index_all, noise_index_all,
        num_bin, nparams, T, t_ref, n_side_bins, n_sub, n_cp_orbit, window_factor,
        freq_from_tdi_phase);
}

void GBComputationGroup::gb_stft_fill_global_wrap(
    cmplx *template_fill,
    Orbits *orbits, TDIConfig *tdi_config,
    STFTFresnel *fresnel, STFTDomain *stft,
    double *params_all, int *data_index_all, double *factors_all,
    int num_bin, int nparams, double T, double t_ref,
    int n_side_bins, double window_factor, bool freq_from_tdi_phase,
    bool active_band)
{
    stft_fill_global_impl<GBTDIonTheFly>(
        template_fill, orbits, tdi_config, fresnel, stft,
        params_all, data_index_all, factors_all,
        num_bin, nparams, T, t_ref, n_side_bins, window_factor,
        freq_from_tdi_phase, active_band);
}

void GBComputationGroup::gb_stft_swap_ll_wrap(
    cmplx *d_h_add_out, cmplx *d_h_remove_out,
    cmplx *add_add_out, cmplx *remove_remove_out, cmplx *add_remove_out,
    Orbits *orbits, TDIConfig *tdi_config,
    STFTFresnel *fresnel, STFTDomain *stft,
    double *params_add_all, double *params_remove_all,
    int *data_index_all, int *noise_index_all,
    int num_bin, int nparams, double T, double t_ref,
    int n_side_bins, double window_factor, bool freq_from_tdi_phase)
{
    stft_swap_ll_impl<GBTDIonTheFly>(
        d_h_add_out, d_h_remove_out, add_add_out, remove_remove_out, add_remove_out,
        orbits, tdi_config, fresnel, stft,
        params_add_all, params_remove_all, data_index_all, noise_index_all,
        num_bin, nparams, T, t_ref, n_side_bins, window_factor,
        freq_from_tdi_phase);
}

// ---- STFT/Fresnel GB F-statistic wrap (Stage 4: get_fstat_ll) --------------
// Instantiate LAT's stft_get_fstat_ll_impl<SourceT> against <GBTDIonTheFly>:
// N_i=(d|A_i) [num_bin,4] + upper-triangle M_ij=(A_i|A_j) [num_bin,10] from the
// 4 Cornish & Crowder basis filters; caller forms 2F = N^T M^-1 N.
void GBComputationGroup::gb_stft_get_fstat_ll_wrap(
    double *N_re_out, double *N_im_out,
    double *M_re_out, double *M_im_out,
    Orbits *orbits, TDIConfig *tdi_config,
    STFTFresnel *fresnel, STFTDomain *stft,
    double *params_all, int *data_index_all, int *noise_index_all,
    int num_bin, int nparams, double T, double t_ref,
    int n_side_bins, double window_factor, bool freq_from_tdi_phase)
{
    stft_get_fstat_ll_impl<GBTDIonTheFly>(
        N_re_out, N_im_out, M_re_out, M_im_out,
        orbits, tdi_config, fresnel, stft,
        params_all, data_index_all, noise_index_all,
        num_bin, nparams, T, t_ref, n_side_bins, window_factor,
        freq_from_tdi_phase);
}

// ---- STFT/Fresnel GB gradient wraps (Stage 3: get_ll_grad + swap_ll_grad) ---
// Instantiate LAT's templated stft_*_grad_impl<SourceT> against <GBTDIonTheFly>;
// per-parameter central finite difference of the STFT log-likelihood.
void GBComputationGroup::gb_stft_get_ll_grad_wrap(
    double *grad_out,
    Orbits *orbits, TDIConfig *tdi_config,
    STFTFresnel *fresnel, STFTDomain *stft,
    double *params_all, int *data_index_all, int *noise_index_all,
    double *param_eps,
    int num_bin, int nparams, double T, double t_ref,
    int n_side_bins, double window_factor, bool freq_from_tdi_phase)
{
    stft_get_ll_grad_impl<GBTDIonTheFly>(
        grad_out, orbits, tdi_config, fresnel, stft,
        params_all, data_index_all, noise_index_all, param_eps,
        num_bin, nparams, T, t_ref, n_side_bins, window_factor,
        freq_from_tdi_phase);
}

void GBComputationGroup::gb_stft_swap_ll_grad_wrap(
    double *grad_add_out, double *grad_remove_out,
    Orbits *orbits, TDIConfig *tdi_config,
    STFTFresnel *fresnel, STFTDomain *stft,
    double *params_add_all, double *params_remove_all,
    int *data_index_all, int *noise_index_all,
    double *param_eps_add, double *param_eps_remove,
    int num_bin, int nparams, double T, double t_ref,
    int n_side_bins, double window_factor, bool freq_from_tdi_phase)
{
    stft_swap_ll_grad_impl<GBTDIonTheFly>(
        grad_add_out, grad_remove_out, orbits, tdi_config, fresnel, stft,
        params_add_all, params_remove_all, data_index_all, noise_index_all,
        param_eps_add, param_eps_remove,
        num_bin, nparams, T, t_ref, n_side_bins, window_factor,
        freq_from_tdi_phase);
}

void GBComputationGroup::gb_wdm_het_swap_ll_wrap(
    double *d_h_add_out, double *d_h_remove_out,
    double *add_add_out, double *remove_remove_out, double *add_remove_out,
    Orbits *orbits, TDIConfig *tdi_config,
    WDMSettings *wdm_settings,
    double *params_add_all, double *params_remove_all,
    int *data_index_all, int *noise_index_all,
    double *chunk_t_starts, int *chunk_keep_lo, int *chunk_keep_hi,
    int *chunk_n_global_offset, double *wdm_window,
    double *data_d, double *invC,
    int n_chunks, int num_bin, int nparams,
    int Nt_sub, int log2_Nt_sub,
    int N_sparse, int log2_N_sparse,
    int nchannels, int n_rfft_chunk,
    double T_chunk, double dt, double T, double t_ref, int tdi_type,
    double tukey_alpha, int grid_dim, int N_cp_sig, int N_cp_orbit,
    int *binary_perm, int *group_starts, int *group_ends,
    int *group_m_lo, int *group_m_hi, int n_groups,
    int *pair_m_lo_b, int *pair_m_hi_b,
    int m_band_half_width)
{
    wdm_het_swap_ll_impl<GBTDIonTheFly>(
        d_h_add_out, d_h_remove_out, add_add_out, remove_remove_out, add_remove_out,
        orbits, tdi_config,
        wdm_settings,
        params_add_all, params_remove_all,
        data_index_all, noise_index_all,
        chunk_t_starts, chunk_keep_lo, chunk_keep_hi, chunk_n_global_offset,
        wdm_window, data_d, invC, n_chunks, num_bin, nparams,
        Nt_sub, log2_Nt_sub, N_sparse, log2_N_sparse,
        nchannels, n_rfft_chunk,
        T_chunk, dt, T, t_ref, tdi_type, tukey_alpha,
        grid_dim, N_cp_sig, N_cp_orbit,
        binary_perm, group_starts, group_ends,
        group_m_lo, group_m_hi, n_groups,
        pair_m_lo_b, pair_m_hi_b, m_band_half_width);
}


void GBComputationGroup::gb_wdm_het_get_fstat_ll_wrap(
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
    int grid_dim, int m_band_half_width)
{
    wdm_het_get_fstat_ll_impl<GBTDIonTheFly>(
        N_arr_re_out, N_arr_im_out, M_mat_re_out, M_mat_im_out,
        orbits, tdi_config, wdm_settings,
        params_all, data_index_all, noise_index_all,
        chunk_t_starts, chunk_keep_lo, chunk_keep_hi, chunk_n_global_offset,
        wdm_window, data_d, invC,
        n_chunks, num_bin, nparams,
        Nt_sub, log2_Nt_sub, N_sparse, log2_N_sparse,
        nchannels, n_rfft_chunk,
        T_chunk, dt, T, t_ref, tdi_type, tukey_alpha,
        grid_dim, m_band_half_width);
}


// ============================================================================
// Phase 3L.7f.4 (2026-06-04): GB signal-heterodyne v2 polyphase family.
// Carved from lisa-on-gpu's TDIonTheFly.cu:6856-7995 (~1140 lines).
//
// Includes:
// - The signal-het v2 polyphase intro comment block.
// - Anonymous-namespace device helpers (polyphase fold + bin-fold
//   inner products + reconstruction primitives -- per-binary single-
//   channel c1_sparse builder used by all 6 wrap methods below).
// - 6 GBComputationGroup wrap methods:
//     gb_signal_het_get_ll_wrap         (dense FD path)
//     gb_signal_het_get_ll_sparse_wrap  (sparse FD validation path)
//     gb_signal_het_get_ll_in_kernel_wrap         (Stage 2b in-kernel)
//     gb_signal_het_fill_global_sparse_wrap        (template fill)
//     gb_signal_het_fill_global_in_kernel_wrap     (Stage 2b in-kernel)
//     gb_signal_het_get_ll_grad_in_kernel_wrap     (central-diff grad)
//
// GPU branches are TODO (CPU fully wired). Pairs with the v2 signal-
// het Python prototype at
// LISAanalysistools/scripts/gb_chunked_het/gb_signal_het_wdm_v2.py.
// ============================================================================

// ============================================================================
// Signal-heterodyne (v2 polyphase) -- CPU implementation
//
// First port of the v2 polyphase signal-het Python prototype at
// LISAanalysistools/scripts/gb_chunked_het/gb_signal_het_wdm_v2.py.
//
// Algorithm per binary (matches Python prototype exactly):
//
//   1. f0_cand = params_cand[bin, f0_idx]
//      m_floor = floor(f0_cand / layer_df)
//      m_active = [m_floor - half, ..., m_floor + half], clipped to active band
//
//   2. For each m_active layer:
//        Polyphase fold + iFFT of length Nt_layer over the windowed FD slice
//        around bin (m * Nt/2), with pre-phase shift to land outputs at
//        n_global = n_start + n_layer * stride.
//        Apply lisatools complex-WDM coefficient layout
//          kappa * (-1)^((m+1)n) * conj(C_mn) * after_ifft * (-1)^n / stride
//        to produce c1_sparse[c, m_active_idx, n_layer].
//
//   3. r[c, m, b] = c1_sparse[c, m_active_idx, b] / c0_sparse[data_idx,
//                  c, m_local_in_full, b]    (safe divide with floor mask).
//      dr/dn via centred FD over b with mean bin width = stride.
//
//   4. Bin-folded inner products (NO carrier de-rotation -- matches the
//      Python v1 sparse path):
//        <d|h> = 0.5 * Re sum_{c',m_act,b} (A0 * r + A1 * dr/dn)
//        <h|h> = 0.5 * Re sum (B0 * r_outer + B1 * cross_drr)
//      with r_outer / cross_drr handled differently for XYZ vs AE/AET.
//
// GPU branch: TODO (prints and returns; CPU fully wired).
// ============================================================================

namespace {

// Per-binary, single channel: compute c1_sparse[m_active_layers, Nt_layer]
// via polyphase fold + naive O(Nt_layer^2) DFT-of-length-Nt_layer.
// Naive DFT is sufficient for correctness validation; swap for radix-2 FFT
// later for performance.
static void signal_het_polyphase_one_channel(
    const cmplx *fd_rfft_chan,            // (n_rfft,) complex
    const int   *m_active,                // (m_active_layers,)
    int          m_active_layers,
    const double *window,                 // (Nt,)
    int          Nt,
    int          Nt_layer,                // iFFT length
    int          N_sparse_t,              // number of sparse outputs kept (<= Nt_layer)
    int          stride,
    int          Nf,
    int          ind_min_t,
    const int   *n_sparse_local_arr,      // (N_sparse_t,) -- only first entry used for n_start
    double       dt,
    int          n_rfft,
    cmplx       *c1_sparse_out)           // (m_active_layers, N_sparse_t)
{
    const int   N        = Nf * Nt;
    const int   half_Nt  = Nt / 2;
    const cmplx I_c      = cmplx(0.0, 1.0);
    const double TWO_PI  = 2.0 * M_PI;
    const double kappa   = 2.0 * std::sqrt(M_PI * dt) / (double) Nf;

    // n_start = first sparse position (ind_min_t + stride/2 by construction).
    const int n_start = ind_min_t + n_sparse_local_arr[0];

    // Working buffers on the stack (per binary x per channel).
    // Max sane Nt for CPU is ~32k, Nt_layer ~ 2k.
    std::vector<cmplx> weighted((size_t) Nt);
    std::vector<cmplx> folded((size_t) Nt_layer);

    for (int im = 0; im < m_active_layers; ++im) {
        const int m_global = m_active[im];
        const int centre   = m_global * half_Nt;

        // Step 1: gather Nt FD bins around the layer centre, with Hermitian
        // wraparound (real TD -> rfft conjugate symmetry).
        for (int i = 0; i < Nt; ++i) {
            const int j_off = i - half_Nt;
            int k = centre + j_off;
            bool conj_flag = false;
            int k_use;
            if (k < 0) {
                k_use = -k;
                conj_flag = true;
            } else if (k > N / 2) {
                k_use = N - k;
                conj_flag = true;
            } else {
                k_use = k;
            }

            cmplx h(0.0, 0.0);
            if (k_use >= 0 && k_use < n_rfft) {
                h = fd_rfft_chan[k_use];
                if (conj_flag) h = gcmplx::conj(h);
            }

            // Window + prephase: phitilde(j_off) * exp(+i 2*pi*j_off*n_start/Nt)
            const double phase_arg = TWO_PI * (double) j_off * (double) n_start / (double) Nt;
            const cmplx  prephase  = gcmplx::exp(I_c * phase_arg);
            weighted[i] = h * window[i] * prephase;
        }

        // Step 2: polyphase fold (length Nt -> length Nt_layer).
        for (int r = 0; r < Nt_layer; ++r) folded[r] = cmplx(0.0, 0.0);
        for (int i = 0; i < Nt; ++i) {
            const int r = i % Nt_layer;
            folded[r] += weighted[i];
        }

        // Step 3: naive iFFT of length Nt_layer (matches numpy ifft conv).
        //   ifft_out[n_layer] = (1/Nt_layer) * sum_r Y[r] * exp(+i 2*pi*r*n_layer/Nt_layer)
        // We only need the first N_sparse_t outputs (sparse positions tile
        // the active n-range; the polyphase identity is exact for all of them).
        for (int n_layer = 0; n_layer < N_sparse_t; ++n_layer) {
            cmplx acc(0.0, 0.0);
            for (int r = 0; r < Nt_layer; ++r) {
                const double pa = TWO_PI * (double) r * (double) n_layer / (double) Nt_layer;
                acc += folded[r] * gcmplx::exp(I_c * pa);
            }
            acc *= (1.0 / (double) Nt_layer);

            // Lisatools complex-WDM conversion at the sparse pixel n_global =
            // n_start + n_layer * stride.
            const int n_global = n_start + n_layer * stride;
            // sign_scale = (-1)^n_global / stride
            const double sign_scale = ((n_global & 1) ? -1.0 : 1.0) / (double) stride;
            const cmplx after_ifft_lt = acc * sign_scale;

            // kappa * (-1)^((m+1)n) * conj(C_mn) where C_mn = 1 (even m+n)
            // or 1j (odd m+n); conj(C_mn) = 1 (even) or -1j (odd).
            const int  m_plus_n  = (m_global + n_global) & 1;
            const cmplx conj_cmn = (m_plus_n == 0) ? cmplx(1.0, 0.0)
                                                   : cmplx(0.0, -1.0);
            const int  sign_mn_int = ((m_global + 1) * n_global) & 1;
            const double sign_mn   = sign_mn_int ? -1.0 : 1.0;
            const cmplx coef = kappa * sign_mn * conj_cmn;

            c1_sparse_out[(size_t) im * N_sparse_t + n_layer] = after_ifft_lt * coef;
        }
    }
}

}  // anonymous namespace

void GBComputationGroup::gb_signal_het_get_ll_wrap(
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
    int     n_rfft, double max_r)
{
    (void) params_ref_all;  // not used in bin-fold path (kept for future de-rotation)
    (void) fdot_idx;
    (void) num_data;
    (void) Nt_active;

#ifdef __CUDACC__
    // GPU port deferred. CPU branch fully wired below; matches the pattern
    // of gb_fd_swap_ll_grad_wrap (header comment line 861-862).
    throw std::runtime_error(
        "[gb_signal_het_get_ll_wrap] GPU implementation is a TODO -- the v2 signal-het CUDA "
        "kernels are not implemented yet. Construct the Python class with "
        "force_backend=\"cpu\" until then. (Silent zero-return previously "
        "masqueraded as a successful call -- see the audit at GBGPU 2026-06-06.)");
#endif

    const int M = 2 * m_active_half_width + 1;
    const int Nf_active_idx_max = Nf_active - 1;
    const double FLOOR_EPS = 1e-12;

    std::vector<cmplx> c1_sparse((size_t) nchannels * M * N_sparse_t);
    std::vector<cmplx> r_sparse((size_t)  nchannels * M * N_sparse_t);
    std::vector<cmplx> dr_sparse((size_t) nchannels * M * N_sparse_t);

    for (int bin = 0; bin < num_bin; ++bin) {
        const double f0_cand = params_cand_all[(size_t) bin * nparams + f0_idx];
        const int    m_floor = (int) std::floor(f0_cand / layer_df);
        int m_active[16];
        for (int im = 0; im < M; ++im) {
            int m_g = m_floor + (im - m_active_half_width);
            if (m_g < ind_min_f) m_g = ind_min_f;
            if (m_g > ind_min_f + Nf_active_idx_max) m_g = ind_min_f + Nf_active_idx_max;
            m_active[im] = m_g;
        }

        const int data_idx = data_index_all[bin];

        for (int c = 0; c < nchannels; ++c) {
            const cmplx *fd_chan = fd_rfft_all + (size_t) bin * nchannels * n_rfft
                                              + (size_t) c * n_rfft;
            cmplx *c1_chan = c1_sparse.data()
                + (size_t) c * M * N_sparse_t;
            signal_het_polyphase_one_channel(
                fd_chan, m_active, M, wdm_window, Nt, Nt_layer, N_sparse_t,
                stride, Nf, ind_min_t, n_sparse_local_arr, dt, n_rfft, c1_chan);
        }

        // r at sparse bin centres (safe divide vs c0)
        for (int c = 0; c < nchannels; ++c) {
            for (int im = 0; im < M; ++im) {
                const int m_local = m_active[im] - ind_min_f;
                double max_mag = 0.0;
                for (int b = 0; b < N_sparse_t; ++b) {
                    const cmplx c0v = c0_sparse_all[
                        ((size_t) data_idx * nchannels + c) * Nf_active * N_sparse_t
                        + (size_t) m_local * N_sparse_t + b];
                    const double mag = gcmplx::abs(c0v);
                    if (mag > max_mag) max_mag = mag;
                }
                const double floor_th = std::max(FLOOR_EPS * max_mag, 1e-300);

                for (int b = 0; b < N_sparse_t; ++b) {
                    const cmplx c0v = c0_sparse_all[
                        ((size_t) data_idx * nchannels + c) * Nf_active * N_sparse_t
                        + (size_t) m_local * N_sparse_t + b];
                    const cmplx c1v = c1_sparse[
                        (size_t) c * M * N_sparse_t
                        + (size_t) im * N_sparse_t + b];
                    const size_t r_idx = (size_t) c * M * N_sparse_t
                                       + (size_t) im * N_sparse_t + b;
                    if (gcmplx::abs(c0v) > floor_th) {
                        cmplx r_val = c1v / c0v;
                        // Amp/phase clip: cap |r| at max_r (channel-cell
                        // direction preserved). Bounds the bin-fold sum
                        // when c0 is small but nonzero. max_r <= 0 disables.
                        if (max_r > 0.0) {
                            const double abs_r = gcmplx::abs(r_val);
                            if (abs_r > max_r) {
                                r_val = r_val * (max_r / abs_r);
                            }
                        }
                        r_sparse[r_idx] = r_val;
                    } else {
                        r_sparse[r_idx] = cmplx(0.0, 0.0);
                    }
                }
            }
        }

        // dr/dn via centred FD over b
        const double Dn = (double) stride;
        for (int c = 0; c < nchannels; ++c) {
            for (int im = 0; im < M; ++im) {
                for (int b = 0; b < N_sparse_t; ++b) {
                    const size_t i_cmb = (size_t) c * M * N_sparse_t
                                       + (size_t) im * N_sparse_t + b;
                    cmplx d(0.0, 0.0);
                    if (N_sparse_t >= 3) {
                        if (b == 0) {
                            d = (r_sparse[i_cmb + 1] - r_sparse[i_cmb]) / Dn;
                        } else if (b == N_sparse_t - 1) {
                            d = (r_sparse[i_cmb] - r_sparse[i_cmb - 1]) / Dn;
                        } else {
                            d = (r_sparse[i_cmb + 1] - r_sparse[i_cmb - 1]) / (2.0 * Dn);
                        }
                    } else if (N_sparse_t == 2) {
                        const size_t i0 = (size_t) c * M * N_sparse_t + (size_t) im * N_sparse_t;
                        d = (r_sparse[i0 + 1] - r_sparse[i0]) / Dn;
                    }
                    dr_sparse[i_cmb] = d;
                }
            }
        }

        // Inner products
        cmplx d_h_raw(0.0, 0.0);
        cmplx h_h_raw(0.0, 0.0);

        for (int c = 0; c < nchannels; ++c) {
            for (int im = 0; im < M; ++im) {
                const int m_local = m_active[im] - ind_min_f;
                for (int b = 0; b < N_sparse_t; ++b) {
                    const cmplx r  = r_sparse[ (size_t) c * M * N_sparse_t
                                            + (size_t) im * N_sparse_t + b];
                    const cmplx dr = dr_sparse[(size_t) c * M * N_sparse_t
                                            + (size_t) im * N_sparse_t + b];
                    const cmplx a0 = A0_all[((size_t) data_idx * nchannels + c)
                                            * Nf_active * N_sparse_t
                                            + (size_t) m_local * N_sparse_t + b];
                    const cmplx a1 = A1_all[((size_t) data_idx * nchannels + c)
                                            * Nf_active * N_sparse_t
                                            + (size_t) m_local * N_sparse_t + b];
                    d_h_raw += a0 * r + a1 * dr;
                }
            }
        }

        if (tdi_type == 0) {
            // XYZ: cross-channel B0/B1 of shape (num_data, nch, nch, Nf_active, Nt_layer)
            for (int c = 0; c < nchannels; ++c) {
                for (int c2 = 0; c2 < nchannels; ++c2) {
                    for (int im = 0; im < M; ++im) {
                        const int m_local = m_active[im] - ind_min_f;
                        for (int b = 0; b < N_sparse_t; ++b) {
                            const cmplx r_c  = r_sparse[ (size_t) c  * M * N_sparse_t
                                                     + (size_t) im * N_sparse_t + b];
                            const cmplx r_c2 = r_sparse[ (size_t) c2 * M * N_sparse_t
                                                     + (size_t) im * N_sparse_t + b];
                            const cmplx dr_c  = dr_sparse[(size_t) c  * M * N_sparse_t
                                                     + (size_t) im * N_sparse_t + b];
                            const cmplx dr_c2 = dr_sparse[(size_t) c2 * M * N_sparse_t
                                                     + (size_t) im * N_sparse_t + b];
                            const cmplx b0 = B0_all[
                                (((size_t) data_idx * nchannels + c) * nchannels + c2)
                                  * Nf_active * N_sparse_t
                                + (size_t) m_local * N_sparse_t + b];
                            const cmplx b1 = B1_all[
                                (((size_t) data_idx * nchannels + c) * nchannels + c2)
                                  * Nf_active * N_sparse_t
                                + (size_t) m_local * N_sparse_t + b];
                            const cmplx r_outer  = gcmplx::conj(r_c) * r_c2;
                            const cmplx cross_drr = gcmplx::conj(r_c)  * dr_c2
                                                  + gcmplx::conj(dr_c) * r_c2;
                            h_h_raw += b0 * r_outer + b1 * cross_drr;
                        }
                    }
                }
            }
        } else {
            // AE / AET: diagonal B0/B1 of shape (num_data, nch, Nf_active, Nt_layer)
            for (int c = 0; c < nchannels; ++c) {
                for (int im = 0; im < M; ++im) {
                    const int m_local = m_active[im] - ind_min_f;
                    for (int b = 0; b < N_sparse_t; ++b) {
                        const cmplx r  = r_sparse[ (size_t) c * M * N_sparse_t
                                                 + (size_t) im * N_sparse_t + b];
                        const cmplx dr = dr_sparse[(size_t) c * M * N_sparse_t
                                                 + (size_t) im * N_sparse_t + b];
                        const cmplx b0 = B0_all[((size_t) data_idx * nchannels + c)
                                                * Nf_active * N_sparse_t
                                                + (size_t) m_local * N_sparse_t + b];
                        const cmplx b1 = B1_all[((size_t) data_idx * nchannels + c)
                                                * Nf_active * N_sparse_t
                                                + (size_t) m_local * N_sparse_t + b];
                        const double rsq = (gcmplx::conj(r) * r).real();
                        const cmplx cross_drr = gcmplx::conj(r) * dr
                                              + gcmplx::conj(dr) * r;
                        h_h_raw += b0 * rsq + b1 * cross_drr;
                    }
                }
            }
        }

        d_h_out[bin] = 0.5 * d_h_raw.real();
        h_h_out[bin] = 0.5 * h_h_raw.real();
    }
}


// ============================================================================
// Signal-heterodyne (v2 polyphase) -- Stage 2a: SPARSE-FD entry point.
// See gb_signal_het_get_ll_sparse_wrap declaration in TDIonTheFly.hh for the
// design rationale. Polyphase fold iterates only the N_sparse_fd nonzero
// bins (the source's spectral support around f0 in absolute frame);
// implicit zero everywhere else.
// ============================================================================

void GBComputationGroup::gb_signal_het_get_ll_sparse_wrap(
    double *d_h_out, double *h_h_out,
    cmplx  *X_het_all, int *k_f0_all,
    cmplx  *c0_sparse_all,
    cmplx  *A0_all, cmplx *A1_all,
    cmplx  *B0_all, cmplx *B1_all,
    cmplx  *B0nc_all, cmplx *B1nc_all,
    double *wdm_window, int *n_sparse_local_arr,
    double *params_cand_all, double *params_ref_all,
    int    *data_index_all,
    int     num_bin, int num_data,
    int     nparams, int f0_idx, int fdot_idx,
    int     Nf, int Nt, int Nf_active, int Nt_active,
    int     Nt_layer, int N_sparse_t, int stride,
    int     ind_min_t, int ind_min_f,
    int     m_active_half_width,
    double  layer_df, double dt,
    int     nchannels, int tdi_type,
    int     N_sparse_fd, double max_r, int project_real)
{
    // project_real != 0: compute the REAL WDM likelihood. <d|h> is exact via the
    // repacked A0/A1 coefficients (no kernel change here -- the same 0.5*Re(A0*r)
    // gives the real value). <h|h> adds the NONCONJ blocks B0nc/B1nc so that
    // 0.5*Re(B0*conj(rc)rc2 + B0nc*rc*rc2 + ...) is the real projection (drops the
    // dr^2 term -> standard linear-r budget off-reference). project_real == 0
    // reproduces the legacy (complex/Hermitian) behaviour and ignores B0nc/B1nc.
    (void) params_ref_all; (void) fdot_idx; (void) num_data; (void) Nt_active;

#ifdef __CUDACC__
    throw std::runtime_error(
        "[gb_signal_het_get_ll_sparse_wrap] GPU implementation is a TODO -- the v2 signal-het CUDA "
        "kernels are not implemented yet. Construct the Python class with "
        "force_backend=\"cpu\" until then. (Silent zero-return previously "
        "masqueraded as a successful call -- see the audit at GBGPU 2026-06-06.)");
#endif

    const int M = 2 * m_active_half_width + 1;
    const int Nf_active_idx_max = Nf_active - 1;
    const double FLOOR_EPS = 1e-12;
    const double TWO_PI = 2.0 * M_PI;
    const cmplx I_c = cmplx(0.0, 1.0);
    const int   half_Nt = Nt / 2;
    const int   half_NS = N_sparse_fd / 2;
    const double kappa = 2.0 * std::sqrt(M_PI * dt) / (double) Nf;
    const int   n_start = ind_min_t + n_sparse_local_arr[0];

    std::vector<cmplx> fold((size_t) nchannels * M * Nt_layer);
    std::vector<cmplx> c1_sparse((size_t) nchannels * M * N_sparse_t);
    std::vector<cmplx> r_sparse((size_t)  nchannels * M * N_sparse_t);
    std::vector<cmplx> dr_sparse((size_t) nchannels * M * N_sparse_t);

    for (int bin = 0; bin < num_bin; ++bin) {
        const double f0_cand = params_cand_all[(size_t) bin * nparams + f0_idx];
        const int    m_floor = (int) std::floor(f0_cand / layer_df);
        int m_active[16];
        for (int im = 0; im < M; ++im) {
            int m_g = m_floor + (im - m_active_half_width);
            if (m_g < ind_min_f) m_g = ind_min_f;
            if (m_g > ind_min_f + Nf_active_idx_max) m_g = ind_min_f + Nf_active_idx_max;
            m_active[im] = m_g;
        }
        const int data_idx = data_index_all[bin];
        const int k_f0     = k_f0_all[bin];

        // Polyphase fold: iterate only N_sparse_fd nonzero bins.
        std::fill(fold.begin(), fold.end(), cmplx(0.0, 0.0));
        for (int c = 0; c < nchannels; ++c) {
            const cmplx *X_chan = X_het_all + (size_t) bin * nchannels * N_sparse_fd
                                            + (size_t) c * N_sparse_fd;
            for (int i = 0; i < N_sparse_fd; ++i) {
                const cmplx Xi = X_chan[i];
                if (Xi.real() == 0.0 && Xi.imag() == 0.0) continue;
                const int k_abs = k_f0 + (i - half_NS);
                for (int im = 0; im < M; ++im) {
                    const int j = k_abs - m_active[im] * half_Nt + half_Nt;
                    if (j < 0 || j >= Nt) continue;
                    const int j_off = j - half_Nt;
                    const double phase_arg = TWO_PI * (double) j_off
                                             * (double) n_start / (double) Nt;
                    const cmplx prephase = gcmplx::exp(I_c * phase_arg);
                    const cmplx weighted = Xi * wdm_window[j] * prephase;
                    const int r = j % Nt_layer;
                    fold[(size_t) c * M * Nt_layer
                       + (size_t) im * Nt_layer + r] += weighted;
                }
            }
        }

        // iFFT of length Nt_layer (naive DFT; keep first N_sparse_t outputs).
        for (int c = 0; c < nchannels; ++c) {
            for (int im = 0; im < M; ++im) {
                const cmplx *fold_cm = fold.data()
                    + (size_t) c * M * Nt_layer + (size_t) im * Nt_layer;
                for (int n_layer = 0; n_layer < N_sparse_t; ++n_layer) {
                    cmplx acc(0.0, 0.0);
                    for (int rr = 0; rr < Nt_layer; ++rr) {
                        const double pa = TWO_PI * (double) rr
                                          * (double) n_layer / (double) Nt_layer;
                        acc += fold_cm[rr] * gcmplx::exp(I_c * pa);
                    }
                    acc *= (1.0 / (double) Nt_layer);
                    const int n_global = n_start + n_layer * stride;
                    const double sign_scale = ((n_global & 1) ? -1.0 : 1.0)
                                              / (double) stride;
                    const cmplx after_ifft_lt = acc * sign_scale;
                    const int  m_global = m_active[im];
                    const int  m_plus_n = (m_global + n_global) & 1;
                    const cmplx conj_cmn = (m_plus_n == 0) ? cmplx(1.0, 0.0)
                                                           : cmplx(0.0, -1.0);
                    const int  sign_mn_int = ((m_global + 1) * n_global) & 1;
                    const double sign_mn = sign_mn_int ? -1.0 : 1.0;
                    const cmplx coef = kappa * sign_mn * conj_cmn;
                    c1_sparse[(size_t) c * M * N_sparse_t
                            + (size_t) im * N_sparse_t + n_layer] = after_ifft_lt * coef;
                }
            }
        }

        // r, dr, inner products: same as Stage 1.
        for (int c = 0; c < nchannels; ++c) {
            for (int im = 0; im < M; ++im) {
                const int m_local = m_active[im] - ind_min_f;
                double max_mag = 0.0;
                for (int b = 0; b < N_sparse_t; ++b) {
                    const cmplx c0v = c0_sparse_all[
                        ((size_t) data_idx * nchannels + c) * Nf_active * N_sparse_t
                        + (size_t) m_local * N_sparse_t + b];
                    const double mag = gcmplx::abs(c0v);
                    if (mag > max_mag) max_mag = mag;
                }
                const double floor_th = std::max(FLOOR_EPS * max_mag, 1e-300);
                for (int b = 0; b < N_sparse_t; ++b) {
                    const cmplx c0v = c0_sparse_all[
                        ((size_t) data_idx * nchannels + c) * Nf_active * N_sparse_t
                        + (size_t) m_local * N_sparse_t + b];
                    const cmplx c1v = c1_sparse[
                        (size_t) c * M * N_sparse_t + (size_t) im * N_sparse_t + b];
                    const size_t r_idx = (size_t) c * M * N_sparse_t
                                       + (size_t) im * N_sparse_t + b;
                    if (gcmplx::abs(c0v) > floor_th) {
                        cmplx r_val = c1v / c0v;
                        // Amp/phase clip: cap |r| at max_r (preserve dir).
                        if (max_r > 0.0) {
                            const double abs_r = gcmplx::abs(r_val);
                            if (abs_r > max_r) r_val = r_val * (max_r / abs_r);
                        }
                        r_sparse[r_idx] = r_val;
                    } else {
                        r_sparse[r_idx] = cmplx(0.0, 0.0);
                    }
                }
            }
        }

        const double Dn = (double) stride;
        for (int c = 0; c < nchannels; ++c) {
            for (int im = 0; im < M; ++im) {
                for (int b = 0; b < N_sparse_t; ++b) {
                    const size_t i_cmb = (size_t) c * M * N_sparse_t
                                       + (size_t) im * N_sparse_t + b;
                    cmplx d(0.0, 0.0);
                    if (N_sparse_t >= 3) {
                        if (b == 0) d = (r_sparse[i_cmb + 1] - r_sparse[i_cmb]) / Dn;
                        else if (b == N_sparse_t - 1) d = (r_sparse[i_cmb] - r_sparse[i_cmb - 1]) / Dn;
                        else d = (r_sparse[i_cmb + 1] - r_sparse[i_cmb - 1]) / (2.0 * Dn);
                    } else if (N_sparse_t == 2) {
                        const size_t i0 = (size_t) c * M * N_sparse_t + (size_t) im * N_sparse_t;
                        d = (r_sparse[i0 + 1] - r_sparse[i0]) / Dn;
                    }
                    dr_sparse[i_cmb] = d;
                }
            }
        }

        cmplx d_h_raw(0.0, 0.0), h_h_raw(0.0, 0.0);
        for (int c = 0; c < nchannels; ++c) {
            for (int im = 0; im < M; ++im) {
                const int m_local = m_active[im] - ind_min_f;
                for (int b = 0; b < N_sparse_t; ++b) {
                    const cmplx r  = r_sparse[ (size_t) c * M * N_sparse_t + (size_t) im * N_sparse_t + b];
                    const cmplx dr = dr_sparse[(size_t) c * M * N_sparse_t + (size_t) im * N_sparse_t + b];
                    const cmplx a0 = A0_all[((size_t) data_idx * nchannels + c) * Nf_active * N_sparse_t + (size_t) m_local * N_sparse_t + b];
                    const cmplx a1 = A1_all[((size_t) data_idx * nchannels + c) * Nf_active * N_sparse_t + (size_t) m_local * N_sparse_t + b];
                    d_h_raw += a0 * r + a1 * dr;
                }
            }
        }
        if (tdi_type == 0) {
            for (int c = 0; c < nchannels; ++c) for (int c2 = 0; c2 < nchannels; ++c2)
                for (int im = 0; im < M; ++im) {
                    const int m_local = m_active[im] - ind_min_f;
                    for (int b = 0; b < N_sparse_t; ++b) {
                        const cmplx r_c  = r_sparse[(size_t) c  * M * N_sparse_t + (size_t) im * N_sparse_t + b];
                        const cmplx r_c2 = r_sparse[(size_t) c2 * M * N_sparse_t + (size_t) im * N_sparse_t + b];
                        const cmplx dr_c  = dr_sparse[(size_t) c  * M * N_sparse_t + (size_t) im * N_sparse_t + b];
                        const cmplx dr_c2 = dr_sparse[(size_t) c2 * M * N_sparse_t + (size_t) im * N_sparse_t + b];
                        const cmplx b0 = B0_all[(((size_t) data_idx * nchannels + c) * nchannels + c2) * Nf_active * N_sparse_t + (size_t) m_local * N_sparse_t + b];
                        const cmplx b1 = B1_all[(((size_t) data_idx * nchannels + c) * nchannels + c2) * Nf_active * N_sparse_t + (size_t) m_local * N_sparse_t + b];
                        const cmplx r_outer = gcmplx::conj(r_c) * r_c2;
                        const cmplx cross_drr = gcmplx::conj(r_c) * dr_c2 + gcmplx::conj(dr_c) * r_c2;
                        h_h_raw += b0 * r_outer + b1 * cross_drr;
                        if (project_real) {
                            const cmplx b0nc = B0nc_all[(((size_t) data_idx * nchannels + c) * nchannels + c2) * Nf_active * N_sparse_t + (size_t) m_local * N_sparse_t + b];
                            const cmplx b1nc = B1nc_all[(((size_t) data_idx * nchannels + c) * nchannels + c2) * Nf_active * N_sparse_t + (size_t) m_local * N_sparse_t + b];
                            // nonconj pairing r_c*r_c2 (NOT conj) -> with the conj
                            // term, 0.5*Re(...) is the real WDM projection.
                            h_h_raw += b0nc * (r_c * r_c2)
                                     + b1nc * (r_c * dr_c2 + dr_c * r_c2);
                        }
                    }
                }
        } else {
            for (int c = 0; c < nchannels; ++c) for (int im = 0; im < M; ++im) {
                const int m_local = m_active[im] - ind_min_f;
                for (int b = 0; b < N_sparse_t; ++b) {
                    const cmplx r  = r_sparse[ (size_t) c * M * N_sparse_t + (size_t) im * N_sparse_t + b];
                    const cmplx dr = dr_sparse[(size_t) c * M * N_sparse_t + (size_t) im * N_sparse_t + b];
                    const cmplx b0 = B0_all[((size_t) data_idx * nchannels + c) * Nf_active * N_sparse_t + (size_t) m_local * N_sparse_t + b];
                    const cmplx b1 = B1_all[((size_t) data_idx * nchannels + c) * Nf_active * N_sparse_t + (size_t) m_local * N_sparse_t + b];
                    const double rsq = (gcmplx::conj(r) * r).real();
                    const cmplx cross_drr = gcmplx::conj(r) * dr + gcmplx::conj(dr) * r;
                    h_h_raw += b0 * rsq + b1 * cross_drr;
                    if (project_real) {
                        const cmplx b0nc = B0nc_all[((size_t) data_idx * nchannels + c) * Nf_active * N_sparse_t + (size_t) m_local * N_sparse_t + b];
                        const cmplx b1nc = B1nc_all[((size_t) data_idx * nchannels + c) * Nf_active * N_sparse_t + (size_t) m_local * N_sparse_t + b];
                        h_h_raw += b0nc * (r * r) + b1nc * (r * dr + dr * r);
                    }
                }
            }
        }

        d_h_out[bin] = 0.5 * d_h_raw.real();
        h_h_out[bin] = 0.5 * h_h_raw.real();
    }
}



// ============================================================================
// Signal-heterodyne (v2 polyphase) -- Stage 2b: IN-KERNEL sparse-FD entry.
//
// Fuses the existing gb_run_fd_wave_tdi_wrap (sparse heterodyned rfft) with
// the Stage 2a polyphase + bin-fold pipeline. Per-source X_het is allocated
// transiently inside this call (heap on CPU; per-block shared memory on GPU
// at Stage 3). NO per-source FD storage in global memory.
//
// CPU implementation: two-pass for clarity --
//   (1) gb_run_fd_wave_tdi_wrap fills X_het + k_f0 buffers
//   (2) gb_signal_het_get_ll_sparse_wrap consumes those buffers
//
// GPU Stage 3 will fuse these into a single kernel with X_het in __shared__.
// ============================================================================

void GBComputationGroup::gb_signal_het_get_ll_in_kernel_wrap(
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
    int     N_sparse_fd, double tukey_alpha, double max_r, int project_real)
{
#ifdef __CUDACC__
    throw std::runtime_error(
        "[gb_signal_het_get_ll_in_kernel_wrap] GPU implementation is a TODO -- the v2 signal-het CUDA "
        "kernels are not implemented yet. Construct the Python class with "
        "force_backend=\"cpu\" until then. (Silent zero-return previously "
        "masqueraded as a successful call -- see the audit at GBGPU 2026-06-06.)");
#endif

    std::vector<cmplx>  X_het_raw((size_t) num_bin * nchannels * N_sparse_fd);
    std::vector<int>    k_f0_buf(num_bin);
    std::vector<double> f0_grid_buf(num_bin);

    gb_run_fd_wave_tdi_wrap(
        tdi_on_fly,
        X_het_raw.data(), k_f0_buf.data(), f0_grid_buf.data(),
        params_cand_all, t_start, T_obs,
        N_sparse_fd, num_bin, nparams, nchannels,
        tukey_alpha);

    // Convert gb_run_fd_wave_tdi output to the centered-slice / dense-rfft
    // convention that gb_signal_het_get_ll_sparse_wrap expects:
    //   * raw layout: X_het_raw[b,c,m_fft] = FFT-order, carrier-removed
    //     sparse FFT, scaled by 0.5*dts where dts = T_obs/N_sparse_fd.
    //   * target:     X_het[b,c,i] = dense_rfft(Tukey*td)[k_f0 + (i - half_NS)],
    //                  i.e. centered slice of the absolute (carrier-intact)
    //                  dense rfft.
    // The continuous-FT representations differ by a 0.5 factor that is
    // already absorbed in the raw 0.5*dts scale, leaving only the
    // sparse-to-dense Riemann conversion (1/dt) and the FFT-order ->
    // centered-slice reordering (an fftshift). Both signals share the
    // t_start time origin so there is no extra linear-phase factor.
    // Empirical bin-by-bin agreement is ~1% (Tukey vs no-window edge bias)
    // which the polyphase fold averages out at the inner-product level.
    std::vector<cmplx> X_het((size_t) num_bin * nchannels * N_sparse_fd);
    const int    half_NS = N_sparse_fd / 2;
    const double dt_inv  = 1.0 / dt;
    for (int b = 0; b < num_bin; ++b) {
        for (int c = 0; c < nchannels; ++c) {
            const size_t base = ((size_t) b * nchannels + c) * N_sparse_fd;
            for (int i = 0; i < N_sparse_fd; ++i) {
                const int m_signed = i - half_NS;
                const int m_fft    = (m_signed >= 0)
                                         ? m_signed
                                         : (m_signed + N_sparse_fd);
                X_het[base + i] = X_het_raw[base + m_fft] * dt_inv;
            }
        }
    }

    this->gb_signal_het_get_ll_sparse_wrap(
        d_h_out, h_h_out,
        X_het.data(), k_f0_buf.data(),
        c0_sparse_all,
        A0_all, A1_all, B0_all, B1_all,
        B0nc_all, B1nc_all,
        wdm_window, n_sparse_local_arr,
        params_cand_all, params_ref_all, data_index_all,
        num_bin, num_data,
        nparams, f0_idx, fdot_idx,
        Nf, Nt, Nf_active, Nt_active,
        Nt_layer, N_sparse_t, stride,
        ind_min_t, ind_min_f,
        m_active_half_width,
        layer_df, dt,
        nchannels, tdi_type,
        N_sparse_fd, max_r, project_real);
}


// ============================================================================
// Signal-heterodyne (v2 polyphase) -- fill_global path.
// ============================================================================
//
// Same FD generation + polyphase + r_sparse machinery as
// gb_signal_het_get_ll_sparse_wrap, but instead of bin-folding r against
// precomputed A0/A1/B0/B1 to produce <d|h>/<h|h>, we reconstruct the dense
// candidate WDM template via the heterodyne identity:
//
//   1. r_sparse(c, m_active, n_sparse) = c1_sparse / c0_sparse
//   2. r_demod_sparse = r_sparse * exp(-i * phase_pred(n_sparse))
//      where phase_pred(t) = 2pi Df0 t + pi Dfdot t^2 from
//      params_cand - params_ref (the KNOWN analytic carrier).
//   3. Linear interpolate r_demod onto dense n.
//   4. r_dense = r_demod_dense * exp(+i * phase_pred(n_dense))  -- re-rotated.
//   5. c1_dense = r_dense * c0_dense_complex  (full active band).
//   6. template_fill[data_idx, c, m_global, n_global] += factor * Re(c1_dense)
//
// c0_dense_complex_all is shape (num_data, nch, Nf_active, Nt_active);
// template_fill is shape (num_data, nch, Nf, Nt) and is accumulated into
// (caller pre-zeroes / atomic-adds to support multiple binaries colliding
// at the same WDM pixel).
// ============================================================================

void GBComputationGroup::gb_signal_het_fill_global_sparse_wrap(
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
    int     N_sparse_fd, double max_r)
{
    (void) num_data;

#ifdef __CUDACC__
    throw std::runtime_error(
        "[gb_signal_het_fill_global_sparse_wrap] GPU implementation is a TODO -- the v2 signal-het CUDA "
        "kernels are not implemented yet. Construct the Python class with "
        "force_backend=\"cpu\" until then. (Silent zero-return previously "
        "masqueraded as a successful call -- see the audit at GBGPU 2026-06-06.)");
#endif

    const int M = 2 * m_active_half_width + 1;
    const int Nf_active_idx_max = Nf_active - 1;
    const double FLOOR_EPS = 1e-12;
    const double TWO_PI = 2.0 * M_PI;
    const cmplx I_c = cmplx(0.0, 1.0);
    const int   half_Nt = Nt / 2;
    const int   half_NS = N_sparse_fd / 2;
    const double kappa = 2.0 * std::sqrt(M_PI * dt) / (double) Nf;
    const int   n_start = ind_min_t + n_sparse_local_arr[0];
    const double layer_dt = (double) Nf * dt;

    std::vector<cmplx> fold((size_t) nchannels * M * Nt_layer);
    std::vector<cmplx> c1_sparse((size_t) nchannels * M * N_sparse_t);
    std::vector<cmplx> r_sparse((size_t)  nchannels * M * N_sparse_t);

    for (int bin = 0; bin < num_bin; ++bin) {
        const double f0_cand   = params_cand_all[(size_t) bin * nparams + f0_idx];
        const double fdot_cand = params_cand_all[(size_t) bin * nparams + fdot_idx];
        const int    m_floor   = (int) std::floor(f0_cand / layer_df);
        int m_active[16];
        for (int im = 0; im < M; ++im) {
            int m_g = m_floor + (im - m_active_half_width);
            if (m_g < ind_min_f) m_g = ind_min_f;
            if (m_g > ind_min_f + Nf_active_idx_max) m_g = ind_min_f + Nf_active_idx_max;
            m_active[im] = m_g;
        }
        const int    data_idx = data_index_all[bin];
        const int    k_f0     = k_f0_all[bin];
        const double factor   = factors_all[bin];

        const double f0_ref   = params_ref_all[(size_t) data_idx * nparams + f0_idx];
        const double fdot_ref = params_ref_all[(size_t) data_idx * nparams + fdot_idx];
        const double Df0      = f0_cand   - f0_ref;
        const double Dfdot    = fdot_cand - fdot_ref;

        // ---- Polyphase fold + iFFT + c1_sparse (identical to Stage 2a) ----
        std::fill(fold.begin(), fold.end(), cmplx(0.0, 0.0));
        for (int c = 0; c < nchannels; ++c) {
            const cmplx *X_chan = X_het_all + (size_t) bin * nchannels * N_sparse_fd
                                            + (size_t) c * N_sparse_fd;
            for (int i = 0; i < N_sparse_fd; ++i) {
                const cmplx Xi = X_chan[i];
                if (Xi.real() == 0.0 && Xi.imag() == 0.0) continue;
                const int k_abs = k_f0 + (i - half_NS);
                for (int im = 0; im < M; ++im) {
                    const int j = k_abs - m_active[im] * half_Nt + half_Nt;
                    if (j < 0 || j >= Nt) continue;
                    const int j_off = j - half_Nt;
                    const double phase_arg = TWO_PI * (double) j_off
                                             * (double) n_start / (double) Nt;
                    const cmplx prephase = gcmplx::exp(I_c * phase_arg);
                    const cmplx weighted = Xi * wdm_window[j] * prephase;
                    const int r = j % Nt_layer;
                    fold[(size_t) c * M * Nt_layer
                       + (size_t) im * Nt_layer + r] += weighted;
                }
            }
        }

        for (int c = 0; c < nchannels; ++c) {
            for (int im = 0; im < M; ++im) {
                const cmplx *fold_cm = fold.data()
                    + (size_t) c * M * Nt_layer + (size_t) im * Nt_layer;
                for (int n_layer = 0; n_layer < N_sparse_t; ++n_layer) {
                    cmplx acc(0.0, 0.0);
                    for (int rr = 0; rr < Nt_layer; ++rr) {
                        const double pa = TWO_PI * (double) rr
                                          * (double) n_layer / (double) Nt_layer;
                        acc += fold_cm[rr] * gcmplx::exp(I_c * pa);
                    }
                    acc *= (1.0 / (double) Nt_layer);
                    const int n_global = n_start + n_layer * stride;
                    const double sign_scale = ((n_global & 1) ? -1.0 : 1.0)
                                              / (double) stride;
                    const cmplx after_ifft_lt = acc * sign_scale;
                    const int  m_global = m_active[im];
                    const int  m_plus_n = (m_global + n_global) & 1;
                    const cmplx conj_cmn = (m_plus_n == 0) ? cmplx(1.0, 0.0)
                                                           : cmplx(0.0, -1.0);
                    const int  sign_mn_int = ((m_global + 1) * n_global) & 1;
                    const double sign_mn = sign_mn_int ? -1.0 : 1.0;
                    const cmplx coef = kappa * sign_mn * conj_cmn;
                    c1_sparse[(size_t) c * M * N_sparse_t
                            + (size_t) im * N_sparse_t + n_layer] = after_ifft_lt * coef;
                }
            }
        }

        // ---- r_sparse = c1_sparse / c0_sparse (with safe-divide floor) ----
        for (int c = 0; c < nchannels; ++c) {
            for (int im = 0; im < M; ++im) {
                const int m_local = m_active[im] - ind_min_f;
                double max_mag = 0.0;
                for (int b = 0; b < N_sparse_t; ++b) {
                    const cmplx c0v = c0_sparse_all[
                        ((size_t) data_idx * nchannels + c) * Nf_active * N_sparse_t
                        + (size_t) m_local * N_sparse_t + b];
                    const double mag = gcmplx::abs(c0v);
                    if (mag > max_mag) max_mag = mag;
                }
                const double floor_th = std::max(FLOOR_EPS * max_mag, 1e-300);
                for (int b = 0; b < N_sparse_t; ++b) {
                    const cmplx c0v = c0_sparse_all[
                        ((size_t) data_idx * nchannels + c) * Nf_active * N_sparse_t
                        + (size_t) m_local * N_sparse_t + b];
                    const cmplx c1v = c1_sparse[
                        (size_t) c * M * N_sparse_t + (size_t) im * N_sparse_t + b];
                    const size_t r_idx = (size_t) c * M * N_sparse_t
                                       + (size_t) im * N_sparse_t + b;
                    if (gcmplx::abs(c0v) > floor_th) {
                        cmplx r_val = c1v / c0v;
                        // Amp/phase clip: cap |r| at max_r (preserve dir).
                        if (max_r > 0.0) {
                            const double abs_r = gcmplx::abs(r_val);
                            if (abs_r > max_r) r_val = r_val * (max_r / abs_r);
                        }
                        r_sparse[r_idx] = r_val;
                    } else {
                        r_sparse[r_idx] = cmplx(0.0, 0.0);
                    }
                }
            }
        }

        // ---- Carrier de-rotate r_sparse (in place) ----
        // phase_pred(t_n) = 2*pi*Df0*t_n + pi*Dfdot*t_n^2, t_n at sparse n.
        for (int b = 0; b < N_sparse_t; ++b) {
            const int n_sparse_local = n_sparse_local_arr[b];
            const double t_n = (double)(ind_min_t + n_sparse_local) * layer_dt;
            const double phase_pred = TWO_PI * Df0 * t_n
                                    + M_PI * Dfdot * t_n * t_n;
            const cmplx rot = gcmplx::exp(I_c * (-phase_pred));
            for (int c = 0; c < nchannels; ++c) {
                for (int im = 0; im < M; ++im) {
                    const size_t idx = (size_t) c * M * N_sparse_t
                                     + (size_t) im * N_sparse_t + b;
                    r_sparse[idx] = r_sparse[idx] * rot;
                }
            }
        }

        // ---- For each dense n: linear-interp r_demod, re-rotate, multiply
        //      c0_dense_complex, take real, scatter into template_fill.
        //      Assumes n_sparse_local[b] = n_sparse_local[0] + b*stride.
        const int n_sparse_local_0 = n_sparse_local_arr[0];
        for (int n_dense = 0; n_dense < Nt_active; ++n_dense) {
            const int    n_off = n_dense - n_sparse_local_0;
            int    b_lo  = n_off / stride;
            double frac = (double)(n_off - b_lo * stride) / (double) stride;
            // Clamp to interpolation domain; extrapolate-as-flat at edges.
            if (b_lo < 0) { b_lo = 0; frac = 0.0; }
            if (b_lo >= N_sparse_t - 1) { b_lo = N_sparse_t - 1; frac = 0.0; }
            const int b_hi = (b_lo + 1 < N_sparse_t) ? (b_lo + 1) : b_lo;

            const double t_n_dense = (double)(ind_min_t + n_dense) * layer_dt;
            const double phase_dense = TWO_PI * Df0 * t_n_dense
                                     + M_PI * Dfdot * t_n_dense * t_n_dense;
            const cmplx rot_back = gcmplx::exp(I_c * phase_dense);

            const int n_global = ind_min_t + n_dense;

            for (int c = 0; c < nchannels; ++c) {
                for (int im = 0; im < M; ++im) {
                    const cmplx r_lo = r_sparse[(size_t) c * M * N_sparse_t
                                             + (size_t) im * N_sparse_t + b_lo];
                    const cmplx r_hi = r_sparse[(size_t) c * M * N_sparse_t
                                             + (size_t) im * N_sparse_t + b_hi];
                    const cmplx r_demod_dense = r_lo * (1.0 - frac) + r_hi * frac;
                    const cmplx r_dense = r_demod_dense * rot_back;

                    const int m_local = m_active[im] - ind_min_f;
                    const cmplx c0v = c0_dense_complex_all[
                        ((size_t) data_idx * nchannels + c) * Nf_active * Nt_active
                        + (size_t) m_local * Nt_active + n_dense];
                    const cmplx c1_dense = r_dense * c0v;

                    const int m_global = m_active[im];
                    const size_t out_idx =
                        ((size_t) data_idx * nchannels + c) * Nf * Nt
                        + (size_t) m_global * Nt + n_global;
                    template_fill[out_idx] += factor * c1_dense.real();
                }
            }
        }
    }
}


// In-kernel variant: takes a GBTDIonTheFly* and generates X_het via the
// existing gb_run_fd_wave_tdi_wrap (Tukey-aware), applies the same
// fftshift + (1/dt) conversion as Stage 2b's get_ll path, then calls the
// sparse fill_global. Mirrors gb_signal_het_get_ll_in_kernel_wrap.
void GBComputationGroup::gb_signal_het_fill_global_in_kernel_wrap(
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
    int     N_sparse_fd, double tukey_alpha, double max_r)
{
#ifdef __CUDACC__
    throw std::runtime_error(
        "[gb_signal_het_fill_global_in_kernel_wrap] GPU implementation is a TODO -- the v2 signal-het CUDA "
        "kernels are not implemented yet. Construct the Python class with "
        "force_backend=\"cpu\" until then. (Silent zero-return previously "
        "masqueraded as a successful call -- see the audit at GBGPU 2026-06-06.)");
#endif

    std::vector<cmplx>  X_het_raw((size_t) num_bin * nchannels * N_sparse_fd);
    std::vector<int>    k_f0_buf(num_bin);
    std::vector<double> f0_grid_buf(num_bin);

    gb_run_fd_wave_tdi_wrap(
        tdi_on_fly,
        X_het_raw.data(), k_f0_buf.data(), f0_grid_buf.data(),
        params_cand_all, t_start, T_obs,
        N_sparse_fd, num_bin, nparams, nchannels,
        tukey_alpha);

    // Same convention conversion as Stage 2b get_ll: fftshift + (1/dt).
    std::vector<cmplx> X_het((size_t) num_bin * nchannels * N_sparse_fd);
    const int    half_NS = N_sparse_fd / 2;
    const double dt_inv  = 1.0 / dt;
    for (int b = 0; b < num_bin; ++b) {
        for (int c = 0; c < nchannels; ++c) {
            const size_t base = ((size_t) b * nchannels + c) * N_sparse_fd;
            for (int i = 0; i < N_sparse_fd; ++i) {
                const int m_signed = i - half_NS;
                const int m_fft    = (m_signed >= 0)
                                         ? m_signed
                                         : (m_signed + N_sparse_fd);
                X_het[base + i] = X_het_raw[base + m_fft] * dt_inv;
            }
        }
    }

    this->gb_signal_het_fill_global_sparse_wrap(
        template_fill,
        X_het.data(), k_f0_buf.data(),
        c0_sparse_all,
        c0_dense_complex_all,
        wdm_window, n_sparse_local_arr,
        params_cand_all, params_ref_all, factors_all, data_index_all,
        num_bin, num_data,
        nparams, f0_idx, fdot_idx,
        Nf, Nt, Nf_active, Nt_active,
        Nt_layer, N_sparse_t, stride,
        ind_min_t, ind_min_f,
        m_active_half_width,
        layer_df, dt,
        nchannels,
        N_sparse_fd, max_r);
}


// ============================================================================
// Signal-heterodyne (v2 polyphase) -- get_ll_grad path.
// ============================================================================
//
// Central-difference gradient of logL = d_h - 0.5 * h_h over each candidate
// param. Mirrors the convention of gb_fd_get_ll_grad_wrap: param_eps is a
// per-parameter finite-difference step (eps_k <= 0 -> freeze that dim).
// Per binary, performs (1 central + 2 * nparams perturbed) calls into the
// Stage 2b get_ll_in_kernel pipeline. Each call regenerates X_het via
// gb_run_fd_wave_tdi_wrap so the FD reflects the perturbed params; the
// shared bin-fold A0/A1/B0/B1 are reused across all perturbations.
// ============================================================================

void GBComputationGroup::gb_signal_het_get_ll_grad_in_kernel_wrap(
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
    int     N_sparse_fd, double tukey_alpha, double max_r)
{
#ifdef __CUDACC__
    throw std::runtime_error(
        "[gb_signal_het_get_ll_grad_in_kernel_wrap] GPU implementation is a TODO -- the v2 signal-het CUDA "
        "kernels are not implemented yet. Construct the Python class with "
        "force_backend=\"cpu\" until then. (Silent zero-return previously "
        "masqueraded as a successful call -- see the audit at GBGPU 2026-06-06.)");
#endif

    std::vector<double> params_priv((size_t) nparams);
    double d_h_C = 0.0, h_h_C = 0.0;
    double d_h_P = 0.0, h_h_P = 0.0;
    double d_h_M = 0.0, h_h_M = 0.0;
    int data_idx_local = 0;

    for (int bin = 0; bin < num_bin; ++bin) {
        data_idx_local = data_index_all[bin];

        // ---- Central evaluation ----
        for (int i = 0; i < nparams; ++i)
            params_priv[i] = params_cand_all[(size_t) bin * nparams + i];

        this->gb_signal_het_get_ll_in_kernel_wrap(
            tdi_on_fly,
            &d_h_C, &h_h_C,
            c0_sparse_all,
            A0_all, A1_all, B0_all, B1_all,
            nullptr, nullptr,   /* B0nc/B1nc: grad stays complex for now */
            wdm_window, n_sparse_local_arr,
            params_priv.data(), params_ref_all, &data_idx_local,
            1, num_data,
            nparams, f0_idx, fdot_idx,
            Nf, Nt, Nf_active, Nt_active,
            Nt_layer, N_sparse_t, stride,
            ind_min_t, ind_min_f,
            m_active_half_width,
            layer_df, dt,
            T_obs, t_start,
            nchannels, tdi_type,
            N_sparse_fd, tukey_alpha, max_r, 0);

        d_h_central[bin] = d_h_C;
        h_h_central[bin] = h_h_C;
        const double ll_C = d_h_C - 0.5 * h_h_C;
        (void) ll_C;

        for (int k = 0; k < nparams; ++k) {
            const double eps = param_eps[k];
            if (eps <= 0.0) {
                grad_out[(size_t) bin * nparams + k] = 0.0;
                continue;
            }
            const double saved = params_priv[k];

            // +eps
            params_priv[k] = saved + eps;
            this->gb_signal_het_get_ll_in_kernel_wrap(
                tdi_on_fly,
                &d_h_P, &h_h_P,
                c0_sparse_all,
                A0_all, A1_all, B0_all, B1_all,
                nullptr, nullptr,   /* B0nc/B1nc: grad stays complex for now */
                wdm_window, n_sparse_local_arr,
                params_priv.data(), params_ref_all, &data_idx_local,
                1, num_data,
                nparams, f0_idx, fdot_idx,
                Nf, Nt, Nf_active, Nt_active,
                Nt_layer, N_sparse_t, stride,
                ind_min_t, ind_min_f,
                m_active_half_width,
                layer_df, dt,
                T_obs, t_start,
                nchannels, tdi_type,
                N_sparse_fd, tukey_alpha, max_r, 0);

            // -eps
            params_priv[k] = saved - eps;
            this->gb_signal_het_get_ll_in_kernel_wrap(
                tdi_on_fly,
                &d_h_M, &h_h_M,
                c0_sparse_all,
                A0_all, A1_all, B0_all, B1_all,
                nullptr, nullptr,   /* B0nc/B1nc: grad stays complex for now */
                wdm_window, n_sparse_local_arr,
                params_priv.data(), params_ref_all, &data_idx_local,
                1, num_data,
                nparams, f0_idx, fdot_idx,
                Nf, Nt, Nf_active, Nt_active,
                Nt_layer, N_sparse_t, stride,
                ind_min_t, ind_min_f,
                m_active_half_width,
                layer_df, dt,
                T_obs, t_start,
                nchannels, tdi_type,
                N_sparse_fd, tukey_alpha, max_r, 0);

            params_priv[k] = saved;

            const double ll_P = d_h_P - 0.5 * h_h_P;
            const double ll_M = d_h_M - 0.5 * h_h_M;
            grad_out[(size_t) bin * nparams + k] = (ll_P - ll_M) / (2.0 * eps);
        }
    }
}
