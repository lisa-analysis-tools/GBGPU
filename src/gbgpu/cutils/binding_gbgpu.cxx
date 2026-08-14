// GBGPU pybind11 module entry.
//
// Sibling of BBHx's binding_bbhx.cxx -- built fresh against the
// post-Phase-3L LAT setup (GBGPU had no prior pybind work to pull
// forward; BBHx's came from origin/pybind).

#include "binding_gbgpu.hpp"

#if defined(__CUDA_COMPILATION__) || defined(__CUDACC__)
#endif

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

// ============================================================================
// Phase 3L.7g (2026-06-04): GBTDIonTheFlyWrap + GBComputationGroupWrap method
// bodies. Carved from lisa-on-gpu/src/fastlisaresponse/cutils/binding_tof.cxx
// (where they previously lived alongside the SOBBH equivalents). The free
// host wrappers they forward to -- gb_run_*_wrap, gb_fd_*_wrap,
// wdm_het_*_impl<GBTDIonTheFly>, gb_signal_het_*_wrap -- now live in
// gb_tdi_on_the_fly.cu (also in this cutils dir).
// ============================================================================

void GBTDIonTheFlyWrap::run_wave_tdi_wrap(
    array_type<std::complex<double>>tdi_channels_arr,
    array_type<double>tdi_amp, array_type<double>tdi_phase, array_type<double>phi_ref,
    array_type<double>params, array_type<double>t_arr, int N, int num_bin, int n_params, int nchannels
)
{
    gb_run_wave_tdi_wrap(
        waveform,
        (cmplx*)return_pointer_and_check_length(tdi_channels_arr, "tdi_channels_arr", N, num_bin * nchannels),
        return_pointer_and_check_length(tdi_amp, "tdi_amp", N, num_bin * nchannels),
        return_pointer_and_check_length(tdi_phase, "tdi_phase", N, num_bin * nchannels),
        return_pointer_and_check_length(phi_ref, "phi_ref", N, num_bin),
        return_pointer_and_check_length(params, "params", n_params, num_bin),
        return_pointer_and_check_length(t_arr, "t_arr", N, num_bin),
        N, num_bin, n_params, nchannels
    );
}

void GBTDIonTheFlyWrap::run_fd_wave_tdi_wrap(
    array_type<std::complex<double>> X_het,
    array_type<int>    k_f0_out,
    array_type<double> f0_grid_out,
    array_type<double> params,
    double t_start, double Tobs,
    int N_sparse, int num_bin, int n_params, int nchannels,
    double tukey_alpha
)
{
    gb_run_fd_wave_tdi_wrap(
        waveform,
        (cmplx*) return_pointer_and_check_length(X_het, "X_het",
                     N_sparse, num_bin * nchannels),
        return_pointer_and_check_length(k_f0_out, "k_f0_out", num_bin, 1),
        return_pointer_and_check_length(f0_grid_out, "f0_grid_out",
                     num_bin, 1),
        return_pointer_and_check_length(params, "params",
                     n_params, num_bin),
        t_start, Tobs,
        N_sparse, num_bin, n_params, nchannels,
        tukey_alpha
    );
}


// ---- FD analogs --------------------------------------------------

void GBComputationGroupWrap::gb_fd_fill_global(
    array_type<std::complex<double>> template_fill,
    OrbitsWrap* orbits_wrap, TDIConfigWrap *tdi_config_wrap,
    FDDomainWrap *fd_wrap,
    array_type<double> params_all, array_type<int> data_index_all,
    array_type<double> factors_all,
    array_type<int> template_start_inds,
    int num_bin, int nparams, double T, double t_start, double t_ref,
    int N_sparse, int nchannels, double tukey_alpha, double edge_frac)
{
    int n_rfft = fd_wrap->fd->n_rfft;
    // template_fill rows need not match fd->num_data: fill_global writes into
    // caller-provided template buffers (e.g. the Fisher's per-source dh
    // workspace), each row carrying its own window start
    // (template_start_inds, length = row count, indexed by data_index).
    if (template_fill.size() % ((size_t) n_rfft * nchannels) != 0 ||
        template_fill.size() == 0) {
        throw std::invalid_argument(
            "template_fill: length must be a positive multiple of "
            "n_rfft * nchannels.");
    }
    size_t num_rows = template_fill.size() / ((size_t) n_rfft * nchannels);
    gb_fd_fill_global_wrap(
        (cmplx*) return_pointer(template_fill, "template_fill"),
        orbits_wrap->orbits, tdi_config_wrap->tdi_config, fd_wrap->fd,
        return_pointer_and_check_length(params_all, "params_all", nparams, num_bin),
        return_pointer_and_check_length(data_index_all, "data_index_all", num_bin, 1),
        return_pointer_and_check_length(factors_all, "factors_all", num_bin, 1),
        return_pointer_and_check_length(template_start_inds,
            "template_start_inds", (int) num_rows, 1),
        num_bin, nparams, T, t_start, t_ref, N_sparse, nchannels, tukey_alpha, edge_frac);
}

void GBComputationGroupWrap::gb_fd_get_ll(
    array_type<double> d_h_out, array_type<double> h_h_out,
    OrbitsWrap* orbits_wrap, TDIConfigWrap *tdi_config_wrap,
    FDDomainWrap *fd_wrap,
    array_type<double> params_all,
    array_type<int> data_index_all, array_type<int> noise_index_all,
    int num_bin, int nparams, double T, double t_start, double t_ref,
    int N_sparse, int nchannels, int tdi_type, double tukey_alpha, double edge_frac)
{
    gb_fd_get_ll_wrap(
        return_pointer_and_check_length(d_h_out, "d_h_out", num_bin, 1),
        return_pointer_and_check_length(h_h_out, "h_h_out", num_bin, 1),
        orbits_wrap->orbits, tdi_config_wrap->tdi_config, fd_wrap->fd,
        return_pointer_and_check_length(params_all, "params_all", nparams, num_bin),
        return_pointer_and_check_length(data_index_all, "data_index_all", num_bin, 1),
        return_pointer_and_check_length(noise_index_all, "noise_index_all", num_bin, 1),
        num_bin, nparams, T, t_start, t_ref, N_sparse, nchannels, tdi_type, tukey_alpha, edge_frac);
}

void GBComputationGroupWrap::gb_fd_swap_ll(
    array_type<double> d_h_add_out, array_type<double> d_h_remove_out,
    array_type<double> add_add_out, array_type<double> remove_remove_out,
    array_type<double> add_remove_out,
    OrbitsWrap* orbits_wrap, TDIConfigWrap *tdi_config_wrap,
    FDDomainWrap *fd_wrap,
    array_type<double> params_add_all, array_type<double> params_remove_all,
    array_type<int> data_index_all, array_type<int> noise_index_all,
    int num_bin, int nparams, double T, double t_start, double t_ref,
    int N_sparse, int nchannels, int tdi_type, double tukey_alpha, double edge_frac)
{
    gb_fd_swap_ll_wrap(
        return_pointer_and_check_length(d_h_add_out, "d_h_add_out", num_bin, 1),
        return_pointer_and_check_length(d_h_remove_out, "d_h_remove_out", num_bin, 1),
        return_pointer_and_check_length(add_add_out, "add_add_out", num_bin, 1),
        return_pointer_and_check_length(remove_remove_out, "remove_remove_out", num_bin, 1),
        return_pointer_and_check_length(add_remove_out, "add_remove_out", num_bin, 1),
        orbits_wrap->orbits, tdi_config_wrap->tdi_config, fd_wrap->fd,
        return_pointer_and_check_length(params_add_all, "params_add_all", nparams, num_bin),
        return_pointer_and_check_length(params_remove_all, "params_remove_all", nparams, num_bin),
        return_pointer_and_check_length(data_index_all, "data_index_all", num_bin, 1),
        return_pointer_and_check_length(noise_index_all, "noise_index_all", num_bin, 1),
        num_bin, nparams, T, t_start, t_ref, N_sparse, nchannels, tdi_type, tukey_alpha, edge_frac);
}

void GBComputationGroupWrap::gb_fd_get_ll_grad(
    array_type<double> grad_out,
    OrbitsWrap* orbits_wrap, TDIConfigWrap *tdi_config_wrap,
    FDDomainWrap *fd_wrap,
    array_type<double> params_all,
    array_type<int> data_index_all, array_type<int> noise_index_all,
    array_type<double> param_eps,
    int num_bin, int nparams, double T, double t_start, double t_ref,
    int N_sparse, int nchannels, int tdi_type)
{
    gb_fd_get_ll_grad_wrap(
        return_pointer_and_check_length(grad_out,        "grad_out",        nparams, num_bin),
        orbits_wrap->orbits, tdi_config_wrap->tdi_config, fd_wrap->fd,
        return_pointer_and_check_length(params_all,      "params_all",      nparams, num_bin),
        return_pointer_and_check_length(data_index_all,  "data_index_all",  num_bin, 1),
        return_pointer_and_check_length(noise_index_all, "noise_index_all", num_bin, 1),
        return_pointer_and_check_length(param_eps,       "param_eps",       nparams, 1),
        num_bin, nparams, T, t_start, t_ref, N_sparse, nchannels, tdi_type);
}

void GBComputationGroupWrap::gb_fd_swap_ll_grad(
    array_type<double> grad_add_out, array_type<double> grad_remove_out,
    OrbitsWrap* orbits_wrap, TDIConfigWrap *tdi_config_wrap,
    FDDomainWrap *fd_wrap,
    array_type<double> params_add_all, array_type<double> params_remove_all,
    array_type<int> data_index_all, array_type<int> noise_index_all,
    array_type<double> param_eps_add, array_type<double> param_eps_remove,
    int num_bin, int nparams, double T, double t_start, double t_ref,
    int N_sparse, int nchannels, int tdi_type)
{
    gb_fd_swap_ll_grad_wrap(
        return_pointer_and_check_length(grad_add_out,      "grad_add_out",      nparams, num_bin),
        return_pointer_and_check_length(grad_remove_out,   "grad_remove_out",   nparams, num_bin),
        orbits_wrap->orbits, tdi_config_wrap->tdi_config, fd_wrap->fd,
        return_pointer_and_check_length(params_add_all,    "params_add_all",    nparams, num_bin),
        return_pointer_and_check_length(params_remove_all, "params_remove_all", nparams, num_bin),
        return_pointer_and_check_length(data_index_all,    "data_index_all",    num_bin, 1),
        return_pointer_and_check_length(noise_index_all,   "noise_index_all",   num_bin, 1),
        return_pointer_and_check_length(param_eps_add,     "param_eps_add",     nparams, 1),
        return_pointer_and_check_length(param_eps_remove,  "param_eps_remove",  nparams, 1),
        num_bin, nparams, T, t_start, t_ref, N_sparse, nchannels, tdi_type);
}


// ===========================================================================
// Chunked-heterodyne wrappers: thin pybind shims that unpack numpy arrays
// into raw pointers and forward to the C++ host wrappers in
// gb_tdi_on_the_fly.cu (the wraps themselves instantiate the templated
// wdm_het_*_impl<GBTDIonTheFly> from LAT's lat_chunked_het_kernels.hh).
// ===========================================================================

void GBComputationGroupWrap::gb_wdm_het_fill_global(
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
    int Nf_slab, array_type<int> slab_min_f)   // task-b per-band slab (0/empty = off)
{
    const int Nf = wdm_settings_wrap->wdm_settings->Nf;
    const int Nt = wdm_settings_wrap->wdm_settings->Nt;
    // Task-b: a narrow per-band slab covers Nf_slab layers instead of the full
    // Nf_active (only in active_band mode). Nf_slab<=0 keeps the full extent.
    const int slab_Nf = (Nf_slab > 0)
        ? Nf_slab : wdm_settings_wrap->wdm_settings->Nf_active;
    // active_band selects the template_fill layout: false -> dense parent grid
    // (nchannels, Nf, Nt); true -> active-band (nchannels, slab_Nf, Nt_active),
    // mirroring WDMDomain so a settings-path AnalysisContainer buffer is
    // written/subtracted directly. per_template is one template slab; the
    // buffer holds num_templates such slabs and data_index[bin] routes each
    // binary into its own slab (0 -> offset 0, backward compatible).
    const size_t per_template = active_band
        ? (size_t) nchannels * slab_Nf
                             * wdm_settings_wrap->wdm_settings->Nt_active
        : (size_t) nchannels * Nf * Nt;
    // The template_fill buffer length must be an integer multiple of one
    // template slab. Compute num_templates and check divisibility here (the
    // per-slab exact check below then confirms the length). data_index is
    // validated buffer-local (0..num_templates-1) so routing stays in-bounds.
    const size_t templ_total = template_fill.size();
    if (per_template == 0 || (templ_total % per_template) != 0) {
        throw std::invalid_argument(
            std::string("template_fill: length ") + std::to_string(templ_total)
            + " is not an integer multiple of one template slab ("
            + std::to_string(per_template) + ").");
    }
    const int num_templates = (int) (templ_total / per_template);
    gb_wdm_het_fill_global_wrap(
        return_pointer_and_check_length(template_fill, "template_fill",
                                        (int) per_template, num_templates),
        orbits_wrap->orbits, tdi_config_wrap->tdi_config,
        wdm_settings_wrap->wdm_settings,
        return_pointer_and_check_length(params_all, "params_all", nparams, num_bin),
        return_pointer_and_check_length(factors_all, "factors_all", num_bin, 1),
        return_pointer_and_check_length(data_index, "data_index", num_bin, 1),
        return_pointer_and_check_length(chunk_t_starts, "chunk_t_starts", n_chunks, 1),
        return_pointer_and_check_length(chunk_keep_lo, "chunk_keep_lo", n_chunks, 1),
        return_pointer_and_check_length(chunk_keep_hi, "chunk_keep_hi", n_chunks, 1),
        return_pointer_and_check_length(chunk_n_global_offset, "chunk_n_global_offset", n_chunks, 1),
        return_pointer_and_check_length(wdm_window, "wdm_window", Nt_sub, 1),
        n_chunks, num_bin, nparams,
        Nt_sub, log2_Nt_sub,
        N_sparse, log2_N_sparse,
        nchannels, n_rfft_chunk,
        T_chunk, dt, T, t_ref, tukey_alpha, grid_dim, N_cp_sig, N_cp_orbit,
        m_band_half_width, active_band,
        Nf_slab,
        (slab_min_f.size() > 0
             ? return_pointer(slab_min_f, "slab_min_f") : nullptr));
}

void GBComputationGroupWrap::gb_wdm_het_get_ll(
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
    int Nf_slab, array_type<int> slab_min_f)   // task-b per-band slab (0/empty = off)
{
    const int gn = (n_groups > 0) ? n_groups : 1;
    // Task-b: per-band slab covers Nf_slab layers (full Nf_active when Nf_slab<=0).
    // The data_d/invC per-slab size checks below key off this extent.
    const int Nf_active = (Nf_slab > 0)
        ? Nf_slab : wdm_settings_wrap->wdm_settings->Nf_active;
    const int Nt_active = wdm_settings_wrap->wdm_settings->Nt_active;
    gb_wdm_het_get_ll_wrap(
        return_pointer_and_check_length(d_h_out, "d_h_out", num_bin, 1),
        return_pointer_and_check_length(h_h_out, "h_h_out", num_bin, 1),
        orbits_wrap->orbits, tdi_config_wrap->tdi_config,
        wdm_settings_wrap->wdm_settings,
        return_pointer_and_check_length(params_all, "params_all", nparams, num_bin),
        return_pointer_and_check_length(data_index_all, "data_index_all", num_bin, 1),
        return_pointer_and_check_length(noise_index_all, "noise_index_all", num_bin, 1),
        return_pointer_and_check_length(chunk_t_starts, "chunk_t_starts", n_chunks, 1),
        return_pointer_and_check_length(chunk_keep_lo, "chunk_keep_lo", n_chunks, 1),
        return_pointer_and_check_length(chunk_keep_hi, "chunk_keep_hi", n_chunks, 1),
        return_pointer_and_check_length(chunk_n_global_offset, "chunk_n_global_offset", n_chunks, 1),
        return_pointer_and_check_length(wdm_window, "wdm_window", Nt_sub, 1),
        // data_d / invC may hold ANY whole number of (walker, band) slabs --
        // the kernels stride into them via data_index / noise_index. Require
        // whole slabs only (per-slab strides below).
        (data_d.size() % ((size_t) nchannels * Nf_active * Nt_active) == 0
             && data_d.size() > 0
             ? return_pointer(data_d, "data_d")
             : throw std::invalid_argument(
                   "data_d: length must be a positive multiple of "
                   "nchannels * Nf_active * Nt_active.")),
        (invC.size() % ((tdi_type == TDI_XYZ)
                            ? (size_t) nchannels * nchannels * Nf_active * Nt_active
                            : (size_t) nchannels * Nf_active * Nt_active) == 0
             && invC.size() > 0
             ? return_pointer(invC, "invC")
             : throw std::invalid_argument(
                   "invC: length must be a positive multiple of the "
                   "per-slab inverse-covariance size.")),
        n_chunks, num_bin, nparams,
        Nt_sub, log2_Nt_sub,
        N_sparse, log2_N_sparse,
        nchannels, n_rfft_chunk,
        T_chunk, dt, T, t_ref, tdi_type, tukey_alpha, grid_dim, N_cp_sig, N_cp_orbit,
        return_pointer_and_check_length(binary_perm,  "binary_perm",  num_bin, 1),
        return_pointer_and_check_length(group_starts, "group_starts", gn, 1),
        return_pointer_and_check_length(group_ends,   "group_ends",   gn, 1),
        return_pointer_and_check_length(group_m_lo,   "group_m_lo",   gn, 1),
        return_pointer_and_check_length(group_m_hi,   "group_m_hi",   gn, 1),
        n_groups, m_band_half_width,
        Nf_slab,
        (slab_min_f.size() > 0
             ? return_pointer(slab_min_f, "slab_min_f") : nullptr));
}

void GBComputationGroupWrap::gb_wdm_het_swap_ll(
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
    int Nf_slab, array_type<int> slab_min_f)   // task-b per-band slab (0/empty = off)
{
    const int gn = (n_groups > 0) ? n_groups : 1;
    // Task-b: per-band slab covers Nf_slab layers (full Nf_active when Nf_slab<=0).
    // The data_d/invC per-slab size checks below key off this extent.
    const int Nf_active = (Nf_slab > 0)
        ? Nf_slab : wdm_settings_wrap->wdm_settings->Nf_active;
    const int Nt_active = wdm_settings_wrap->wdm_settings->Nt_active;
    gb_wdm_het_swap_ll_wrap(
        return_pointer_and_check_length(d_h_add_out,       "d_h_add_out",       num_bin, 1),
        return_pointer_and_check_length(d_h_remove_out,    "d_h_remove_out",    num_bin, 1),
        return_pointer_and_check_length(add_add_out,       "add_add_out",       num_bin, 1),
        return_pointer_and_check_length(remove_remove_out, "remove_remove_out", num_bin, 1),
        return_pointer_and_check_length(add_remove_out,    "add_remove_out",    num_bin, 1),
        orbits_wrap->orbits, tdi_config_wrap->tdi_config,
        wdm_settings_wrap->wdm_settings,
        return_pointer_and_check_length(params_add_all,    "params_add_all",    nparams, num_bin),
        return_pointer_and_check_length(params_remove_all, "params_remove_all", nparams, num_bin),
        return_pointer_and_check_length(data_index_all,    "data_index_all",    num_bin, 1),
        return_pointer_and_check_length(noise_index_all,   "noise_index_all",   num_bin, 1),
        return_pointer_and_check_length(chunk_t_starts,    "chunk_t_starts",    n_chunks, 1),
        return_pointer_and_check_length(chunk_keep_lo,     "chunk_keep_lo",     n_chunks, 1),
        return_pointer_and_check_length(chunk_keep_hi,     "chunk_keep_hi",     n_chunks, 1),
        return_pointer_and_check_length(chunk_n_global_offset, "chunk_n_global_offset", n_chunks, 1),
        return_pointer_and_check_length(wdm_window, "wdm_window", Nt_sub, 1),
        // data_d / invC may hold ANY whole number of (walker, band) slabs --
        // the kernels stride into them via data_index / noise_index. Require
        // whole slabs only (per-slab strides below).
        (data_d.size() % ((size_t) nchannels * Nf_active * Nt_active) == 0
             && data_d.size() > 0
             ? return_pointer(data_d, "data_d")
             : throw std::invalid_argument(
                   "data_d: length must be a positive multiple of "
                   "nchannels * Nf_active * Nt_active.")),
        (invC.size() % ((tdi_type == TDI_XYZ)
                            ? (size_t) nchannels * nchannels * Nf_active * Nt_active
                            : (size_t) nchannels * Nf_active * Nt_active) == 0
             && invC.size() > 0
             ? return_pointer(invC, "invC")
             : throw std::invalid_argument(
                   "invC: length must be a positive multiple of the "
                   "per-slab inverse-covariance size.")),
        n_chunks, num_bin, nparams,
        Nt_sub, log2_Nt_sub,
        N_sparse, log2_N_sparse,
        nchannels, n_rfft_chunk,
        T_chunk, dt, T, t_ref, tdi_type, tukey_alpha, grid_dim, N_cp_sig, N_cp_orbit,
        return_pointer_and_check_length(binary_perm,  "binary_perm",  num_bin, 1),
        return_pointer_and_check_length(group_starts, "group_starts", gn, 1),
        return_pointer_and_check_length(group_ends,   "group_ends",   gn, 1),
        return_pointer_and_check_length(group_m_lo,   "group_m_lo",   gn, 1),
        return_pointer_and_check_length(group_m_hi,   "group_m_hi",   gn, 1),
        n_groups,
        return_pointer_and_check_length(pair_m_lo_b, "pair_m_lo_b", num_bin, 1),
        return_pointer_and_check_length(pair_m_hi_b, "pair_m_hi_b", num_bin, 1),
        m_band_half_width,
        Nf_slab,
        (slab_min_f.size() > 0
             ? return_pointer(slab_min_f, "slab_min_f") : nullptr));
}

void GBComputationGroupWrap::gb_wdm_het_get_fstat_ll(
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
    int Nf_slab, array_type<int> slab_min_f)   // task-b per-band slab (0/empty = off)
{
    // Task-b: per-band slab covers Nf_slab layers (full Nf_active when Nf_slab<=0).
    // The data_d/invC per-slab size checks below key off this extent.
    const int Nf_active = (Nf_slab > 0)
        ? Nf_slab : wdm_settings_wrap->wdm_settings->Nf_active;
    const int Nt_active = wdm_settings_wrap->wdm_settings->Nt_active;
    gb_wdm_het_get_fstat_ll_wrap(
        return_pointer_and_check_length(N_arr_re_out, "N_arr_re_out", num_bin, 4),
        return_pointer_and_check_length(N_arr_im_out, "N_arr_im_out", num_bin, 4),
        return_pointer_and_check_length(M_mat_re_out, "M_mat_re_out", num_bin, 10),
        return_pointer_and_check_length(M_mat_im_out, "M_mat_im_out", num_bin, 10),
        orbits_wrap->orbits, tdi_config_wrap->tdi_config,
        wdm_settings_wrap->wdm_settings,
        return_pointer_and_check_length(params_all, "params_all", nparams, num_bin),
        return_pointer_and_check_length(data_index_all, "data_index_all", num_bin, 1),
        return_pointer_and_check_length(noise_index_all, "noise_index_all", num_bin, 1),
        return_pointer_and_check_length(chunk_t_starts, "chunk_t_starts", n_chunks, 1),
        return_pointer_and_check_length(chunk_keep_lo, "chunk_keep_lo", n_chunks, 1),
        return_pointer_and_check_length(chunk_keep_hi, "chunk_keep_hi", n_chunks, 1),
        return_pointer_and_check_length(chunk_n_global_offset, "chunk_n_global_offset", n_chunks, 1),
        return_pointer_and_check_length(wdm_window, "wdm_window", Nt_sub, 1),
        // data_d / invC may hold ANY whole number of (walker, band) slabs --
        // the kernels stride into them via data_index / noise_index. Require
        // whole slabs only (per-slab strides below).
        (data_d.size() % ((size_t) nchannels * Nf_active * Nt_active) == 0
             && data_d.size() > 0
             ? return_pointer(data_d, "data_d")
             : throw std::invalid_argument(
                   "data_d: length must be a positive multiple of "
                   "nchannels * Nf_active * Nt_active.")),
        (invC.size() % ((tdi_type == TDI_XYZ)
                            ? (size_t) nchannels * nchannels * Nf_active * Nt_active
                            : (size_t) nchannels * Nf_active * Nt_active) == 0
             && invC.size() > 0
             ? return_pointer(invC, "invC")
             : throw std::invalid_argument(
                   "invC: length must be a positive multiple of the "
                   "per-slab inverse-covariance size.")),
        n_chunks, num_bin, nparams,
        Nt_sub, log2_Nt_sub,
        N_sparse, log2_N_sparse,
        nchannels, n_rfft_chunk,
        T_chunk, dt, T, t_ref, tdi_type,
        tukey_alpha, grid_dim, m_band_half_width,
        Nf_slab,
        (slab_min_f.size() > 0
             ? return_pointer(slab_min_f, "slab_min_f") : nullptr));
}


// ---- Signal-heterodyne (v2 polyphase) pybind shims ------------------------

void GBComputationGroupWrap::gb_signal_het_get_ll(
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
    int n_rfft, double max_r)
{
    (void) Nt_layer;
    const size_t b_xyz = (size_t) num_data * nchannels * nchannels
                       * Nf_active * N_sparse_t;
    const size_t b_diag = (size_t) num_data * nchannels * Nf_active * N_sparse_t;
    gb_signal_het_get_ll_wrap(
        return_pointer_and_check_length(d_h_out, "d_h_out", num_bin, 1),
        return_pointer_and_check_length(h_h_out, "h_h_out", num_bin, 1),
        reinterpret_cast<cmplx*>(return_pointer_and_check_length(
            fd_rfft_all, "fd_rfft_all",
            (size_t) num_bin * nchannels * n_rfft, 1)),
        reinterpret_cast<cmplx*>(return_pointer_and_check_length(
            c0_sparse_all, "c0_sparse_all",
            (size_t) num_data * nchannels * Nf_active * N_sparse_t, 1)),
        reinterpret_cast<cmplx*>(return_pointer_and_check_length(
            A0_all, "A0_all",
            (size_t) num_data * nchannels * Nf_active * N_sparse_t, 1)),
        reinterpret_cast<cmplx*>(return_pointer_and_check_length(
            A1_all, "A1_all",
            (size_t) num_data * nchannels * Nf_active * N_sparse_t, 1)),
        reinterpret_cast<cmplx*>(return_pointer_and_check_length(
            B0_all, "B0_all",
            (tdi_type == 0) ? b_xyz : b_diag, 1)),
        reinterpret_cast<cmplx*>(return_pointer_and_check_length(
            B1_all, "B1_all",
            (tdi_type == 0) ? b_xyz : b_diag, 1)),
        return_pointer_and_check_length(wdm_window, "wdm_window", Nt, 1),
        return_pointer_and_check_length(n_sparse_local_arr, "n_sparse_local",
                                         N_sparse_t, 1),
        return_pointer_and_check_length(params_cand_all, "params_cand_all",
                                         nparams, num_bin),
        return_pointer_and_check_length(params_ref_all, "params_ref_all",
                                         nparams, num_data),
        return_pointer_and_check_length(data_index_all, "data_index_all",
                                         num_bin, 1),
        num_bin, num_data,
        nparams, f0_idx, fdot_idx,
        Nf, Nt, Nf_active, Nt_active,
        Nt_layer, N_sparse_t, stride,
        ind_min_t, ind_min_f,
        m_active_half_width,
        layer_df, dt,
        nchannels, tdi_type,
        n_rfft, max_r);
}

void GBComputationGroupWrap::gb_signal_het_get_ll_sparse(
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
    int N_sparse_fd, double max_r)
{
    (void) Nt_layer;
    const size_t b_xyz = (size_t) num_data * nchannels * nchannels
                       * Nf_active * N_sparse_t;
    const size_t b_diag = (size_t) num_data * nchannels * Nf_active * N_sparse_t;
    gb_signal_het_get_ll_sparse_wrap(
        return_pointer_and_check_length(d_h_out, "d_h_out", num_bin, 1),
        return_pointer_and_check_length(h_h_out, "h_h_out", num_bin, 1),
        reinterpret_cast<cmplx*>(return_pointer_and_check_length(
            X_het_all, "X_het_all",
            (size_t) num_bin * nchannels * N_sparse_fd, 1)),
        return_pointer_and_check_length(k_f0_all, "k_f0_all", num_bin, 1),
        reinterpret_cast<cmplx*>(return_pointer_and_check_length(
            c0_sparse_all, "c0_sparse_all",
            (size_t) num_data * nchannels * Nf_active * N_sparse_t, 1)),
        reinterpret_cast<cmplx*>(return_pointer_and_check_length(
            A0_all, "A0_all",
            (size_t) num_data * nchannels * Nf_active * N_sparse_t, 1)),
        reinterpret_cast<cmplx*>(return_pointer_and_check_length(
            A1_all, "A1_all",
            (size_t) num_data * nchannels * Nf_active * N_sparse_t, 1)),
        reinterpret_cast<cmplx*>(return_pointer_and_check_length(
            B0_all, "B0_all", (tdi_type == 0) ? b_xyz : b_diag, 1)),
        reinterpret_cast<cmplx*>(return_pointer_and_check_length(
            B1_all, "B1_all", (tdi_type == 0) ? b_xyz : b_diag, 1)),
        nullptr, nullptr,   /* B0nc/B1nc: this validation path stays complex */
        return_pointer_and_check_length(wdm_window, "wdm_window", Nt, 1),
        return_pointer_and_check_length(n_sparse_local_arr, "n_sparse_local",
                                         N_sparse_t, 1),
        return_pointer_and_check_length(params_cand_all, "params_cand_all",
                                         nparams, num_bin),
        return_pointer_and_check_length(params_ref_all, "params_ref_all",
                                         nparams, num_data),
        return_pointer_and_check_length(data_index_all, "data_index_all",
                                         num_bin, 1),
        num_bin, num_data,
        nparams, f0_idx, fdot_idx,
        Nf, Nt, Nf_active, Nt_active,
        Nt_layer, N_sparse_t, stride,
        ind_min_t, ind_min_f,
        m_active_half_width,
        layer_df, dt,
        nchannels, tdi_type,
        N_sparse_fd, max_r, 0);
}

void GBComputationGroupWrap::gb_signal_het_get_ll_in_kernel(
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
    int N_sparse_fd, double tukey_alpha, double max_r, int project_real,
    int n_cp_sig)
{
    (void) Nt_layer;
    const size_t b_xyz  = (size_t) num_data * nchannels * nchannels
                        * Nf_active * N_sparse_t;
    const size_t b_diag = (size_t) num_data * nchannels * Nf_active * N_sparse_t;

    gb_signal_het_get_ll_in_kernel_wrap(
        tdi_wrap->waveform,
        return_pointer_and_check_length(d_h_out, "d_h_out", num_bin, 1),
        return_pointer_and_check_length(h_h_out, "h_h_out", num_bin, 1),
        reinterpret_cast<cmplx*>(return_pointer_and_check_length(
            c0_sparse_all, "c0_sparse_all",
            (size_t) num_data * nchannels * Nf_active * N_sparse_t, 1)),
        reinterpret_cast<cmplx*>(return_pointer_and_check_length(
            A0_all, "A0_all",
            (size_t) num_data * nchannels * Nf_active * N_sparse_t, 1)),
        reinterpret_cast<cmplx*>(return_pointer_and_check_length(
            A1_all, "A1_all",
            (size_t) num_data * nchannels * Nf_active * N_sparse_t, 1)),
        reinterpret_cast<cmplx*>(return_pointer_and_check_length(
            B0_all, "B0_all", (tdi_type == 0) ? b_xyz : b_diag, 1)),
        reinterpret_cast<cmplx*>(return_pointer_and_check_length(
            B1_all, "B1_all", (tdi_type == 0) ? b_xyz : b_diag, 1)),
        reinterpret_cast<cmplx*>(return_pointer_and_check_length(
            B0nc_all, "B0nc_all", (tdi_type == 0) ? b_xyz : b_diag, 1)),
        reinterpret_cast<cmplx*>(return_pointer_and_check_length(
            B1nc_all, "B1nc_all", (tdi_type == 0) ? b_xyz : b_diag, 1)),
        return_pointer_and_check_length(wdm_window, "wdm_window", Nt, 1),
        return_pointer_and_check_length(n_sparse_local_arr, "n_sparse_local",
                                         N_sparse_t, 1),
        return_pointer_and_check_length(params_cand_all, "params_cand_all",
                                         nparams, num_bin),
        return_pointer_and_check_length(params_ref_all, "params_ref_all",
                                         nparams, num_data),
        return_pointer_and_check_length(data_index_all, "data_index_all",
                                         num_bin, 1),
        num_bin, num_data,
        nparams, f0_idx, fdot_idx,
        Nf, Nt, Nf_active, Nt_active,
        Nt_layer, N_sparse_t, stride,
        ind_min_t, ind_min_f,
        m_active_half_width,
        layer_df, dt,
        T_obs, t_start,
        nchannels, tdi_type,
        N_sparse_fd, tukey_alpha, max_r, project_real, n_cp_sig);
}

void GBComputationGroupWrap::gb_signal_het_v3_get_ll(
    GBTDIonTheFlyWrap *tdi_wrap,
    array_type<double> d_h_out, array_type<double> h_h_out,
    array_type<std::complex<double>> c0_sparse_all,
    array_type<std::complex<double>> A0_all,
    array_type<std::complex<double>> A1_all,
    array_type<std::complex<double>> B0_all,
    array_type<std::complex<double>> B1_all,
    array_type<std::complex<double>> B0nc_all,
    array_type<std::complex<double>> B1nc_all,
    array_type<int> n_sparse_local_arr,
    array_type<double> params_cand_all,
    array_type<double> params_ref_all,
    array_type<int> data_index_all,
    int num_bin, int num_data,
    int n_nodes, int nparams, int f0_idx, int fdot_idx,
    int Nf, int Nt, int Nf_active, int Nt_active,
    int Nt_layer, int N_sparse_t, int stride,
    int ind_min_t, int ind_min_f,
    int m_active_half_width,
    double layer_df, double dt,
    double T_obs, double t_start,
    int nchannels, int tdi_type, int project_real)
{
    const size_t b_xyz  = (size_t) num_data * nchannels * nchannels
                        * Nf_active * N_sparse_t;
    const size_t b_diag = (size_t) num_data * nchannels * Nf_active * N_sparse_t;

    gb_signal_het_v3_get_ll_wrap(
        tdi_wrap->waveform,
        return_pointer_and_check_length(d_h_out, "d_h_out", num_bin, 1),
        return_pointer_and_check_length(h_h_out, "h_h_out", num_bin, 1),
        reinterpret_cast<cmplx*>(return_pointer_and_check_length(
            c0_sparse_all, "c0_sparse_all",
            (size_t) num_data * nchannels * Nf_active * N_sparse_t, 1)),
        reinterpret_cast<cmplx*>(return_pointer_and_check_length(
            A0_all, "A0_all",
            (size_t) num_data * nchannels * Nf_active * N_sparse_t, 1)),
        reinterpret_cast<cmplx*>(return_pointer_and_check_length(
            A1_all, "A1_all",
            (size_t) num_data * nchannels * Nf_active * N_sparse_t, 1)),
        reinterpret_cast<cmplx*>(return_pointer_and_check_length(
            B0_all, "B0_all", (tdi_type == 0) ? b_xyz : b_diag, 1)),
        reinterpret_cast<cmplx*>(return_pointer_and_check_length(
            B1_all, "B1_all", (tdi_type == 0) ? b_xyz : b_diag, 1)),
        reinterpret_cast<cmplx*>(return_pointer_and_check_length(
            B0nc_all, "B0nc_all", (tdi_type == 0) ? b_xyz : b_diag, 1)),
        reinterpret_cast<cmplx*>(return_pointer_and_check_length(
            B1nc_all, "B1nc_all", (tdi_type == 0) ? b_xyz : b_diag, 1)),
        return_pointer_and_check_length(n_sparse_local_arr, "n_sparse_local",
                                         N_sparse_t, 1),
        return_pointer_and_check_length(params_cand_all, "params_cand_all",
                                         nparams, num_bin),
        return_pointer_and_check_length(params_ref_all, "params_ref_all",
                                         nparams, num_data),
        return_pointer_and_check_length(data_index_all, "data_index_all",
                                         num_bin, 1),
        num_bin, num_data,
        n_nodes, nparams, f0_idx, fdot_idx,
        Nf, Nt, Nf_active, Nt_active,
        Nt_layer, N_sparse_t, stride,
        ind_min_t, ind_min_f,
        m_active_half_width,
        layer_df, dt,
        T_obs, t_start,
        nchannels, tdi_type, project_real);
}

void GBComputationGroupWrap::gb_signal_het_v4_get_ll(
    GBTDIonTheFlyWrap *tdi_wrap,
    array_type<double> d_h_out, array_type<double> h_h_out,
    array_type<std::complex<double>> c0_sparse_all,
    array_type<std::complex<double>> A0_all,
    array_type<std::complex<double>> A1_all,
    array_type<std::complex<double>> B0_all,
    array_type<std::complex<double>> B1_all,
    array_type<std::complex<double>> B0nc_all,
    array_type<std::complex<double>> B1nc_all,
    array_type<int> n_sparse_local_arr,
        array_type<double> band_w, array_type<int> band_j0, int band_len,
    array_type<double> params_cand_all,
    array_type<double> params_ref_all,
    array_type<int> data_index_all,
    int num_bin, int num_data,
    int n_nodes, int n_knots, int nparams, int f0_idx, int fdot_idx,
    int Nf, int Nt, int Nf_active, int Nt_active,
    int Nt_layer, int N_sparse_t, int stride,
    int ind_min_t, int ind_min_f,
    int m_active_half_width,
    double layer_df, double dt,
    double T_obs, double t_start,
    int nchannels, int tdi_type, int project_real)
{
    const size_t b_xyz  = (size_t) num_data * nchannels * nchannels
                        * Nf_active * N_sparse_t;
    const size_t b_diag = (size_t) num_data * nchannels * Nf_active * N_sparse_t;

    gb_signal_het_v4_get_ll_wrap(
        tdi_wrap->waveform,
        return_pointer_and_check_length(d_h_out, "d_h_out", num_bin, 1),
        return_pointer_and_check_length(h_h_out, "h_h_out", num_bin, 1),
        reinterpret_cast<cmplx*>(return_pointer_and_check_length(
            c0_sparse_all, "c0_sparse_all",
            (size_t) num_data * nchannels * Nf_active * N_sparse_t, 1)),
        reinterpret_cast<cmplx*>(return_pointer_and_check_length(
            A0_all, "A0_all",
            (size_t) num_data * nchannels * Nf_active * N_sparse_t, 1)),
        reinterpret_cast<cmplx*>(return_pointer_and_check_length(
            A1_all, "A1_all",
            (size_t) num_data * nchannels * Nf_active * N_sparse_t, 1)),
        reinterpret_cast<cmplx*>(return_pointer_and_check_length(
            B0_all, "B0_all", (tdi_type == 0) ? b_xyz : b_diag, 1)),
        reinterpret_cast<cmplx*>(return_pointer_and_check_length(
            B1_all, "B1_all", (tdi_type == 0) ? b_xyz : b_diag, 1)),
        reinterpret_cast<cmplx*>(return_pointer_and_check_length(
            B0nc_all, "B0nc_all", (tdi_type == 0) ? b_xyz : b_diag, 1)),
        reinterpret_cast<cmplx*>(return_pointer_and_check_length(
            B1nc_all, "B1nc_all", (tdi_type == 0) ? b_xyz : b_diag, 1)),
        return_pointer_and_check_length(n_sparse_local_arr, "n_sparse_local",
                                         N_sparse_t, 1),
        band_w.data(), band_j0.data(), band_len,
        return_pointer_and_check_length(params_cand_all, "params_cand_all",
                                         nparams, num_bin),
        return_pointer_and_check_length(params_ref_all, "params_ref_all",
                                         nparams, num_data),
        return_pointer_and_check_length(data_index_all, "data_index_all",
                                         num_bin, 1),
        num_bin, num_data,
        n_nodes, n_knots, nparams, f0_idx, fdot_idx,
        Nf, Nt, Nf_active, Nt_active,
        Nt_layer, N_sparse_t, stride,
        ind_min_t, ind_min_f,
        m_active_half_width,
        layer_df, dt,
        T_obs, t_start,
        nchannels, tdi_type, project_real);
}

// ---- signal-het V5: the v4 body with c0_sparse_all -> c0_mask_all (the
// ---- precomputed row-floor mask) plus the trailing v5_mode ---------------
void GBComputationGroupWrap::gb_signal_het_v5_get_ll(
    GBTDIonTheFlyWrap *tdi_wrap,
    array_type<double> d_h_out, array_type<double> h_h_out,
    array_type<uint64_t> c0_mask_all,
    array_type<std::complex<double>> A0_all,
    array_type<std::complex<double>> A1_all,
    array_type<std::complex<double>> B0_all,
    array_type<std::complex<double>> B1_all,
    array_type<std::complex<double>> B0nc_all,
    array_type<std::complex<double>> B1nc_all,
    array_type<int> n_sparse_local_arr,
    array_type<double> band_w, array_type<int> band_j0, int band_len,
    array_type<double> params_cand_all,
    array_type<double> params_ref_all,
    array_type<int> data_index_all,
    int num_bin, int num_data,
    int n_nodes, int n_knots, int nparams, int f0_idx, int fdot_idx,
    int Nf, int Nt, int Nf_active, int Nt_active,
    int Nt_layer, int N_sparse_t, int stride,
    int ind_min_t, int ind_min_f,
    int m_active_half_width,
    double layer_df, double dt,
    double T_obs, double t_start,
    int nchannels, int tdi_type, int project_real,
    int v5_mode)
{
    const size_t b_xyz  = (size_t) num_data * nchannels * nchannels
                        * Nf_active * N_sparse_t;
    const size_t b_diag = (size_t) num_data * nchannels * Nf_active * N_sparse_t;
    // One bit per (data, channel, active layer, sparse pixel), packed along
    // the pixel axis -- 1/128 the size of c0_sparse_all itself.
    const size_t n_mask = (size_t) num_data * nchannels * Nf_active
                        * (size_t) ((N_sparse_t + 63) / 64);

    gb_signal_het_v5_get_ll_wrap(
        tdi_wrap->waveform,
        return_pointer_and_check_length(d_h_out, "d_h_out", num_bin, 1),
        return_pointer_and_check_length(h_h_out, "h_h_out", num_bin, 1),
        reinterpret_cast<unsigned long long*>(
            return_pointer_and_check_length(c0_mask_all, "c0_mask_all",
                                            n_mask, 1)),
        reinterpret_cast<cmplx*>(return_pointer_and_check_length(
            A0_all, "A0_all",
            (size_t) num_data * nchannels * Nf_active * N_sparse_t, 1)),
        reinterpret_cast<cmplx*>(return_pointer_and_check_length(
            A1_all, "A1_all",
            (size_t) num_data * nchannels * Nf_active * N_sparse_t, 1)),
        reinterpret_cast<cmplx*>(return_pointer_and_check_length(
            B0_all, "B0_all", (tdi_type == 0) ? b_xyz : b_diag, 1)),
        reinterpret_cast<cmplx*>(return_pointer_and_check_length(
            B1_all, "B1_all", (tdi_type == 0) ? b_xyz : b_diag, 1)),
        reinterpret_cast<cmplx*>(return_pointer_and_check_length(
            B0nc_all, "B0nc_all", (tdi_type == 0) ? b_xyz : b_diag, 1)),
        reinterpret_cast<cmplx*>(return_pointer_and_check_length(
            B1nc_all, "B1nc_all", (tdi_type == 0) ? b_xyz : b_diag, 1)),
        return_pointer_and_check_length(n_sparse_local_arr, "n_sparse_local",
                                         N_sparse_t, 1),
        band_w.data(), band_j0.data(), band_len,
        return_pointer_and_check_length(params_cand_all, "params_cand_all",
                                         nparams, num_bin),
        return_pointer_and_check_length(params_ref_all, "params_ref_all",
                                         nparams, num_data),
        return_pointer_and_check_length(data_index_all, "data_index_all",
                                         num_bin, 1),
        num_bin, num_data,
        n_nodes, n_knots, nparams, f0_idx, fdot_idx,
        Nf, Nt, Nf_active, Nt_active,
        Nt_layer, N_sparse_t, stride,
        ind_min_t, ind_min_f,
        m_active_half_width,
        layer_df, dt,
        T_obs, t_start,
        nchannels, tdi_type, project_real,
        v5_mode);
}

// ---- signal-het F-STAT: (N, M) for the 4 basis filters against shared
// ---- references; COMPACT per-reference stash windows (W_slab + w_lo) -----
void GBComputationGroupWrap::gb_signal_het_fstat_get_ll(
    GBTDIonTheFlyWrap *tdi_wrap,
    array_type<double> N_out, array_type<double> M_out,
    array_type<uint64_t> c0_mask_all,
    array_type<std::complex<double>> A0_all,
    array_type<std::complex<double>> A1_all,
    array_type<std::complex<double>> B0_all,
    array_type<std::complex<double>> B1_all,
    array_type<std::complex<double>> B0nc_all,
    array_type<std::complex<double>> B1nc_all,
    array_type<int> n_sparse_local_arr,
    array_type<double> band_w, array_type<int> band_j0, int band_len,
    array_type<double> params_cand_all,
    array_type<double> params_ref_all,
    array_type<int> data_index_all,
    array_type<int> w_lo_arr,
    int num_bin, int num_data,
    int n_nodes, int n_knots, int nparams, int f0_idx, int fdot_idx,
    int Nf, int Nt, int Nf_active, int W_slab, int Nt_active,
    int Nt_layer, int N_sparse_t, int stride,
    int ind_min_t, int ind_min_f,
    int m_active_half_width,
    double layer_df, double dt,
    double T_obs, double t_start,
    int nchannels, int tdi_type, int project_real,
    int fstat_mode)
{
    // COMPACT stash sizes: W_slab wide per reference, absolute window
    // origins in w_lo_arr (full-band = W_slab == Nf_active + zeros).
    const size_t b_xyz  = (size_t) num_data * nchannels * nchannels
                        * W_slab * N_sparse_t;
    const size_t b_diag = (size_t) num_data * nchannels * W_slab * N_sparse_t;
    const size_t n_mask = (size_t) num_data * nchannels * W_slab
                        * (size_t) ((N_sparse_t + 63) / 64);

    gb_signal_het_fstat_get_ll_wrap(
        tdi_wrap->waveform,
        return_pointer_and_check_length(N_out, "N_out",
                                         (size_t) num_bin * 4, 1),
        return_pointer_and_check_length(M_out, "M_out",
                                         (size_t) num_bin * 10, 1),
        reinterpret_cast<unsigned long long*>(
            return_pointer_and_check_length(c0_mask_all, "c0_mask_all",
                                            n_mask, 1)),
        reinterpret_cast<cmplx*>(return_pointer_and_check_length(
            A0_all, "A0_all", b_diag, 1)),
        reinterpret_cast<cmplx*>(return_pointer_and_check_length(
            A1_all, "A1_all", b_diag, 1)),
        reinterpret_cast<cmplx*>(return_pointer_and_check_length(
            B0_all, "B0_all", (tdi_type == 0) ? b_xyz : b_diag, 1)),
        reinterpret_cast<cmplx*>(return_pointer_and_check_length(
            B1_all, "B1_all", (tdi_type == 0) ? b_xyz : b_diag, 1)),
        reinterpret_cast<cmplx*>(return_pointer_and_check_length(
            B0nc_all, "B0nc_all", (tdi_type == 0) ? b_xyz : b_diag, 1)),
        reinterpret_cast<cmplx*>(return_pointer_and_check_length(
            B1nc_all, "B1nc_all", (tdi_type == 0) ? b_xyz : b_diag, 1)),
        return_pointer_and_check_length(n_sparse_local_arr, "n_sparse_local",
                                         N_sparse_t, 1),
        band_w.data(), band_j0.data(), band_len,
        return_pointer_and_check_length(params_cand_all, "params_cand_all",
                                         nparams, num_bin),
        return_pointer_and_check_length(params_ref_all, "params_ref_all",
                                         nparams, num_data),
        return_pointer_and_check_length(data_index_all, "data_index_all",
                                         num_bin, 1),
        return_pointer_and_check_length(w_lo_arr, "w_lo_arr", num_data, 1),
        num_bin, num_data,
        n_nodes, n_knots, nparams, f0_idx, fdot_idx,
        Nf, Nt, Nf_active, W_slab, Nt_active,
        Nt_layer, N_sparse_t, stride,
        ind_min_t, ind_min_f,
        m_active_half_width,
        layer_df, dt,
        T_obs, t_start,
        nchannels, tdi_type, project_real,
        fstat_mode);
}

void GBComputationGroupWrap::gb_signal_het_fill_global_in_kernel(
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
    int N_sparse_fd, double tukey_alpha, double max_r)
{
    (void) Nt_layer;
    gb_signal_het_fill_global_in_kernel_wrap(
        tdi_wrap->waveform,
        return_pointer_and_check_length(template_fill, "template_fill",
            (size_t) num_data * nchannels * Nf * Nt, 1),
        reinterpret_cast<cmplx*>(return_pointer_and_check_length(
            c0_sparse_all, "c0_sparse_all",
            (size_t) num_data * nchannels * Nf_active * N_sparse_t, 1)),
        reinterpret_cast<cmplx*>(return_pointer_and_check_length(
            c0_dense_complex_all, "c0_dense_complex_all",
            (size_t) num_data * nchannels * Nf_active * Nt_active, 1)),
        return_pointer_and_check_length(wdm_window, "wdm_window", Nt, 1),
        return_pointer_and_check_length(n_sparse_local_arr, "n_sparse_local",
                                         N_sparse_t, 1),
        return_pointer_and_check_length(params_cand_all, "params_cand_all",
                                         nparams, num_bin),
        return_pointer_and_check_length(params_ref_all, "params_ref_all",
                                         nparams, num_data),
        return_pointer_and_check_length(factors_all, "factors_all",
                                         num_bin, 1),
        return_pointer_and_check_length(data_index_all, "data_index_all",
                                         num_bin, 1),
        num_bin, num_data,
        nparams, f0_idx, fdot_idx,
        Nf, Nt, Nf_active, Nt_active,
        Nt_layer, N_sparse_t, stride,
        ind_min_t, ind_min_f,
        m_active_half_width,
        layer_df, dt,
        T_obs, t_start,
        nchannels,
        N_sparse_fd, tukey_alpha, max_r);
}

void GBComputationGroupWrap::gb_signal_het_make_reference(
    GBTDIonTheFlyWrap *tdi_wrap,
    array_type<std::complex<double>> c0_sparse_out,
    array_type<std::complex<double>> c0_dense_out,
    array_type<double> wdm_window,
    array_type<int> n_sparse_local_arr,
    array_type<int> w_lo_arr,
    array_type<double> params_ref_all,
    int num_data,
    int nparams, int f0_idx, int fdot_idx,
    int Nf, int Nt, int Nf_active, int Nt_active,
    int Nt_layer, int N_sparse_t, int stride,
    int ind_min_t, int ind_min_f,
    double layer_df, double dt,
    double T_obs, double t_start,
    int nchannels,
    int N_sparse_fd, double tukey_alpha, int n_cp_sig)
{
    gb_signal_het_make_reference_wrap(
        tdi_wrap->waveform,
        reinterpret_cast<cmplx*>(return_pointer_and_check_length(
            c0_sparse_out, "c0_sparse_out",
            (size_t) num_data * nchannels * Nf_active * N_sparse_t, 1)),
        reinterpret_cast<cmplx*>(return_pointer_and_check_length(
            c0_dense_out, "c0_dense_out",
            (size_t) num_data * nchannels * Nf_active * Nt_active, 1)),
        return_pointer_and_check_length(wdm_window, "wdm_window", Nt, 1),
        return_pointer_and_check_length(n_sparse_local_arr, "n_sparse_local",
                                         N_sparse_t, 1),
        return_pointer_and_check_length(w_lo_arr, "w_lo_arr", num_data, 1),
        return_pointer_and_check_length(params_ref_all, "params_ref_all",
                                         nparams, num_data),
        num_data,
        nparams, f0_idx, fdot_idx,
        Nf, Nt, Nf_active, Nt_active,
        Nt_layer, N_sparse_t, stride,
        ind_min_t, ind_min_f,
        layer_df, dt,
        T_obs, t_start,
        nchannels,
        N_sparse_fd, tukey_alpha, n_cp_sig);
}

void GBComputationGroupWrap::gb_signal_het_get_ll_grad_in_kernel(
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
    int N_sparse_fd, double tukey_alpha, double max_r)
{
    (void) Nt_layer;
    const size_t b_xyz  = (size_t) num_data * nchannels * nchannels
                        * Nf_active * N_sparse_t;
    const size_t b_diag = (size_t) num_data * nchannels * Nf_active * N_sparse_t;

    gb_signal_het_get_ll_grad_in_kernel_wrap(
        tdi_wrap->waveform,
        return_pointer_and_check_length(grad_out, "grad_out",
                                         nparams, num_bin),
        return_pointer_and_check_length(d_h_central, "d_h_central", num_bin, 1),
        return_pointer_and_check_length(h_h_central, "h_h_central", num_bin, 1),
        reinterpret_cast<cmplx*>(return_pointer_and_check_length(
            c0_sparse_all, "c0_sparse_all",
            (size_t) num_data * nchannels * Nf_active * N_sparse_t, 1)),
        reinterpret_cast<cmplx*>(return_pointer_and_check_length(
            A0_all, "A0_all",
            (size_t) num_data * nchannels * Nf_active * N_sparse_t, 1)),
        reinterpret_cast<cmplx*>(return_pointer_and_check_length(
            A1_all, "A1_all",
            (size_t) num_data * nchannels * Nf_active * N_sparse_t, 1)),
        reinterpret_cast<cmplx*>(return_pointer_and_check_length(
            B0_all, "B0_all", (tdi_type == 0) ? b_xyz : b_diag, 1)),
        reinterpret_cast<cmplx*>(return_pointer_and_check_length(
            B1_all, "B1_all", (tdi_type == 0) ? b_xyz : b_diag, 1)),
        return_pointer_and_check_length(wdm_window, "wdm_window", Nt, 1),
        return_pointer_and_check_length(n_sparse_local_arr, "n_sparse_local",
                                         N_sparse_t, 1),
        return_pointer_and_check_length(params_cand_all, "params_cand_all",
                                         nparams, num_bin),
        return_pointer_and_check_length(params_ref_all, "params_ref_all",
                                         nparams, num_data),
        return_pointer_and_check_length(data_index_all, "data_index_all",
                                         num_bin, 1),
        return_pointer_and_check_length(param_eps, "param_eps", nparams, 1),
        num_bin, num_data,
        nparams, f0_idx, fdot_idx,
        Nf, Nt, Nf_active, Nt_active,
        Nt_layer, N_sparse_t, stride,
        ind_min_t, ind_min_f,
        m_active_half_width,
        layer_df, dt,
        T_obs, t_start,
        nchannels, tdi_type,
        N_sparse_fd, tukey_alpha, max_r);
}


void gbgpu_part(nb::module_ &m) {
#if defined(__CUDA_COMPILATION__) || defined(__CUDACC__)
    nb::class_<GBGPUComputationWrap>(m, "GBGPUComputationWrapGPU")
#else
    nb::class_<GBGPUComputationWrap>(m, "GBGPUComputationWrapCPU")
#endif
        .def(nb::init<>())
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
        .def("SharedMemoryInfoMatComp_wrap",
             &GBGPUComputationWrap::SharedMemoryInfoMatComp_wrap,
             "Shared-memory GB Fisher/information-matrix evaluation (mojito).")
        ;

    // ========================================================================
    // Phase 3L.7g (2026-06-04): GBTDIonTheFlyWrap + GBComputationGroupWrap
    // pybind11 registrations. Carved out of lisa-on-gpu's binding_tof.cxx.
    // The SOBBH equivalents stay in lisa-on-gpu until Phase 3L.8 lands them in
    // BBHx.
    // ========================================================================

#if defined(__CUDA_COMPILATION__) || defined(__CUDACC__)
    nb::class_<GBTDIonTheFlyWrap>(m, "GBTDIonTheFlyWrapGPU")
#else
    nb::class_<GBTDIonTheFlyWrap>(m, "GBTDIonTheFlyWrapCPU")
#endif
    .def(nb::init<OrbitsWrap *, TDIConfigWrap *, double, double>(),
         nb::arg("orbits"), nb::arg("tdi_config"), nb::arg("Tobs"), nb::arg("t_ref"))
    .def("run_wave_tdi_wrap", &GBTDIonTheFlyWrap::run_wave_tdi_wrap, "Preform TDI combinations.")
    .def("run_fd_wave_tdi_wrap", &GBTDIonTheFlyWrap::run_fd_wave_tdi_wrap,
         "Heterodyne FD GB TDI on a sparse time grid. tukey_alpha applies a "
         "scipy.signal.windows.tukey(N_sparse, alpha) taper to the slow "
         "signal before the in-place FFT; pass the same alpha used on the "
         "dense rfft(Tukey*td) side so the two FD paths agree.",
         nb::arg("X_het"), nb::arg("k_f0_out"), nb::arg("f0_grid_out"),
         nb::arg("params"), nb::arg("t_start"), nb::arg("Tobs"),
         nb::arg("N_sparse"), nb::arg("num_bin"), nb::arg("n_params"),
         nb::arg("nchannels"), nb::arg("tukey_alpha") = 0.0)
    .def("get_buffer_size", &GBTDIonTheFlyWrap::get_buffer_size, "Get needed buffer size.")
    .def("get_fd_buffer_size", &GBTDIonTheFlyWrap::get_fd_buffer_size,
         "Get shared-memory size for the heterodyne FD kernel.")
    .def_rw("orbits", &GBTDIonTheFlyWrap::orbits)
    .def_rw("tdi_config", &GBTDIonTheFlyWrap::tdi_config)
    ;

#if defined(__CUDA_COMPILATION__) || defined(__CUDACC__)
    nb::class_<GBTDIonTheFly>(m, "GBTDIonTheFlyGPU")
#else
    nb::class_<GBTDIonTheFly>(m, "GBTDIonTheFlyCPU")
#endif
    .def(nb::init<Orbits *, TDIConfig*, double, double>(),
         nb::arg("orbits"), nb::arg("tdi_config"), nb::arg("Tobs"), nb::arg("t_ref"))
    ;

#if defined(__CUDA_COMPILATION__) || defined(__CUDACC__)
    nb::class_<GBComputationGroupWrap>(m, "GBComputationGroupWrapGPU")
#else
    nb::class_<GBComputationGroupWrap>(m, "GBComputationGroupWrapCPU")
#endif
    .def(nb::init<>())
    .def("gb_fd_fill_global", &GBComputationGroupWrap::gb_fd_fill_global,
         "FD analog of gb_wdm_fill_global: scatter per-source heterodyne FD onto a "
         "global rfft-grid template (cmplx, shape (num_data, nchannels, n_rfft)).")
    .def("gb_fd_get_ll", &GBComputationGroupWrap::gb_fd_get_ll,
         "FD analog of gb_wdm_get_ll: (d|h) and (h|h) per binary using the "
         "lisatools FD inner product (4 Re sum conj(d) h invC * df).  tdi_type "
         "selects between TDI_XYZ (cross-channel 3x3 invC) and TDI_AET/TDI_AE "
         "(diagonal invC).")
    .def("gb_fd_swap_ll", &GBComputationGroupWrap::gb_fd_swap_ll,
         "FD analog of gb_wdm_swap_ll: returns the five inner products "
         "<d|h_add>, <d|h_remove>, <h_add|h_add>, <h_remove|h_remove>, "
         "<h_add|h_remove> needed for an RJMCMC swap proposal.")
    .def("gb_fd_get_ll_grad", &GBComputationGroupWrap::gb_fd_get_ll_grad,
         "FD analog of gb_wdm_get_ll_grad: chain-rule parameter gradient of "
         "L = -1/2 <d-h|d-h> evaluated in the sparse-FD heterodyne pipeline. "
         "param_eps[k] is the central-FD step for theta_k (pass <= 0 to "
         "freeze).")
    .def("gb_fd_swap_ll_grad", &GBComputationGroupWrap::gb_fd_swap_ll_grad,
         "FD analog of gb_wdm_swap_ll_grad: returns (grad_add, grad_remove), "
         "the per-binary derivatives of ll_diff = L(after swap) - L(before "
         "swap) with respect to theta_add and theta_remove respectively.")
    .def("gb_wdm_het_fill_global", &GBComputationGroupWrap::gb_wdm_het_fill_global,
         "Chunked-heterodyne fill_global. Builds the WDM-domain GB template by "
         "iterating over precomputed chunks (chunk_t_starts / keep_lo / keep_hi / "
         "n_global_offset). Each block (chunk) walks all binaries so per-chunk "
         "PSD/data slabs are reused. grid_dim picks the launch grid (use "
         "chunked_het_grid_dim).")
    .def("gb_wdm_het_get_ll", &GBComputationGroupWrap::gb_wdm_het_get_ll,
         "Chunked-heterodyne get_ll. Returns <d|h> and <h|h> per binary, "
         "matching gb_wdm_get_ll up to numerical precision.")
    .def("gb_wdm_het_swap_ll", &GBComputationGroupWrap::gb_wdm_het_swap_ll,
         "Chunked-heterodyne swap_ll. Returns the same five inner products as "
         "gb_wdm_swap_ll: <d|h_add>, <d|h_remove>, <h_add|h_add>, "
         "<h_remove|h_remove>, <h_add|h_remove>.")
    .def("gb_wdm_het_get_fstat_ll", &GBComputationGroupWrap::gb_wdm_het_get_fstat_ll,
         "Chunked-heterodyne F-stat. Returns per-binary N_arr (4,) = "
         "<d|A_i> and M_mat (10,) = <A_i|A_j> upper-triangle (4 basis "
         "filters per Cornish & Crowder '05). Python computes "
         "F = N^T M^{-1} N / 2 from these. Imag outputs always 0 "
         "(WDM coefficients are real).")
    .def("gb_signal_het_get_ll", &GBComputationGroupWrap::gb_signal_het_get_ll,
         "Signal-heterodyne (v2 polyphase) get_ll. Takes precomputed "
         "rfft(Tukey*td) per binary plus reference c0_sparse/A0/A1/B0/B1; "
         "returns per-binary <d|h>, <h|h> via the bin-folded inner-product "
         "accumulator. Stage 1: CPU only, FD as input. Stage 2 will move FD "
         "generation in-kernel via a sparse-spline absolute-FD source.")
    .def("gb_signal_het_get_ll_sparse", &GBComputationGroupWrap::gb_signal_het_get_ll_sparse,
         "Stage 2a sparse-FD signal-het get_ll. Consumes X_het (length "
         "N_sparse_fd per binary per channel) + per-binary k_f0. Polyphase "
         "fold iterates only the N_sparse_fd nonzero bins. Stage 2b will fill "
         "X_het in-kernel from the source-class heterodyned sparse rfft.")
    .def("gb_signal_het_v3_get_ll", &GBComputationGroupWrap::gb_signal_het_v3_get_ll,
         "Signal-het V3: ratio-spline candidate build straight into the bin-fold")
    .def("gb_signal_het_v4_get_ll", &GBComputationGroupWrap::gb_signal_het_v4_get_ll,
         "Signal-het V3: ratio-spline candidate build straight into the bin-fold")
    .def("gb_signal_het_v5_get_ll", &GBComputationGroupWrap::gb_signal_het_v5_get_ll,
         "Signal-het V5: v4-banded with the per-candidate fold scratch "
         "(r_sparse / dr_sparse) ELIMINATED rather than relocated. They are "
         "an M-fold replication of r_pix modulated by one candidate-"
         "independent bit per (row, pixel), so the bit is precomputed once "
         "per reference build and passed here as c0_mask_all (in place of "
         "c0_sparse_all, which the scorer no longer reads) and r/dr are "
         "rebuilt in registers. Per-pixel shared cost 528 -> 48 B/point; "
         "with the phase-lifetime arena the footprint is 27.6 KB, constant "
         "in N_sparse_t up to N ~ 450. Bit-identical to v4. Trailing "
         "v5_mode: 1 = phase-aliased arena (production; ~5 blocks/SM on an "
         "A100 vs v4's 1), 2 = flat carve at the same arithmetic and "
         "traffic (~3 blocks/SM) -- the A/B that isolates occupancy. "
         "GB_SIGHET_V5_VERBOSE=1 prints registers/thread and CUDA's own "
         "achieved blocks/SM for the launch.")
    .def("gb_signal_het_fstat_get_ll",
         &GBComputationGroupWrap::gb_signal_het_fstat_get_ll,
         // GIL released for the wrap body: the multi-device fstat fan-out
         // runs one lane per GPU on host threads, and holding the GIL here
         // serializes every lane's kernel call (the wrap touches no Python
         // objects after nanobind's argument conversion).
         nb::call_guard<nb::gil_scoped_release>(),
         "Signal-het F-stat: per-candidate N (num_bin, 4) = <d|A_i> and "
         "M_upper (num_bin, 10) = <A_i|A_j> row-major upper triangle for "
         "the 4 Cornish & Crowder basis filters (A=2, iota=pi/2, "
         "psi={0,pi/4,0,pi/4}, phi0={0,pi,3pi/2,pi/2}) at each candidate's "
         "intrinsics, scored against SHARED heterodyne references through "
         "the v5 node stage + fixed-knot resample + generalized bin-fold. "
         "Same (N, M) contract as the chunked-het gb_wdm_het_get_fstat_ll. "
         "Stash arrays are COMPACT per-reference windows (W_slab wide, "
         "absolute origins ind_min_f + w_lo_arr[d]); full-band = W_slab == "
         "Nf_active + all-zero w_lo. fstat_mode 0 = 2 node stages + exact "
         "phi0 rotation (production); 1 = 4 independent stages "
         "(recombination self-check). References must be built in the "
         "CIRCULAR reference frame (A=2, iota=0, psi=0, phi0=0) -- see "
         "setup_fstat_references / the F-STAT section of "
         "gb_tdi_on_the_fly.cu for the dlnA-clamp and ratio-pole "
         "rationale.")
    .def("gb_signal_het_get_ll_in_kernel", &GBComputationGroupWrap::gb_signal_het_get_ll_in_kernel,
         "Stage 2b in-kernel sparse-FD signal-het get_ll. Fuses "
         "gb_run_fd_wave_tdi (sparse heterodyned rfft from the GB source "
         "class) with the polyphase + bin-fold pipeline. Takes a "
         "GBTDIonTheFlyWrap; X_het is held in a transient per-call buffer. "
         "tukey_alpha must be supplied by the caller and match the alpha "
         "used to window the dense rfft(Tukey*td) on the analysis side -- "
         "NO default is provided so the C++ and Python sides cannot fall "
         "out of sync. max_r > 0 caps |r| per channel-cell to prevent "
         "the positive-logL blowup at angle excursions; max_r <= 0 "
         "disables the clip (preserves pre-fix behavior).")
    .def("gb_signal_het_fill_global_in_kernel",
         &GBComputationGroupWrap::gb_signal_het_fill_global_in_kernel,
         "Signal-het fill_global. Same FD + polyphase + r_sparse path as "
         "get_ll_in_kernel, but reconstructs the dense candidate template "
         "via r_dense = interp(r_sparse * e^{-i phi_pred}) * e^{+i phi_pred}, "
         "multiplies by stored c0_dense_complex on the active band, takes "
         "the real part, and scatters factor * Re(c1_dense) into "
         "template_fill at the absolute (m, n_global) WDM positions. "
         "Caller pre-zeroes / accumulates into template_fill. tukey_alpha "
         "must be supplied by the caller (no default).")
    .def("gb_signal_het_make_reference",
         &GBComputationGroupWrap::gb_signal_het_make_reference,
         // Same GIL release as the fstat scorer: each fan-out lane builds
         // its own reference blocks, and concurrent per-device builds only
         // parallelize if the wrap body runs without the GIL.
         nb::call_guard<nb::gil_scoped_release>(),
         "Reference producer: run gb_run_fd_wave_tdi on the REFERENCE params + "
         "the same polyphase as get_ll over ALL Nf_active layers, emitting the "
         "complex WDM c0 at the sparse grid (c0_sparse_out) and full Nt "
         "resolution (c0_dense_out). Replaces the Python polyphase so the sig-het "
         "reference comes from the backend. CPU-only (GPU TODO).")
    .def("gb_signal_het_get_ll_grad_in_kernel",
         &GBComputationGroupWrap::gb_signal_het_get_ll_grad_in_kernel,
         "Signal-het central-difference gradient of logL = d_h - 0.5*h_h. "
         "param_eps[k] is the per-parameter finite-difference step; "
         "eps_k <= 0 freezes dimension k. Returns grad[num_bin, nparams] "
         "alongside d_h_central / h_h_central for the unperturbed point "
         "so callers get logL + grad in one pass. Per binary cost is "
         "1 central + 2*nparams perturbed get_ll_in_kernel calls; reuses "
         "the supplied A0/A1/B0/B1 bin-fold tables for all calls.")
    ;
}


NB_MODULE(cgbgpu, m) {
    m.doc() = "GBGPU pybind11 backend. Hosts the GB-specific waveform + "
              "utils wrappers (gbgpu_utils / SharedMemoryGBGPU) plus the "
              "GBTDIonTheFly + GBComputationGroup machinery carved out of "
              "lisa-on-gpu at Phase 3L.7 (2026-06-04).";
    m.attr("TDI_XYZ") = TDI_XYZ;
    m.attr("TDI_AET") = TDI_AET;
    m.attr("TDI_AE") = TDI_AE;
    gbgpu_part(m);
}
