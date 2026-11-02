// gb_dffr.cu -- the DFFR response of galactic binaries: the gb_dffr_* entry points of gb_dffr.hh,
// which instantiate LAT's DFFR tile (lat_dffr_kernels.hh) and the STFT consumers
// (lat_stft_kernels.hh) with GBDFFRSource of gb_dffr.hh. A translation unit of its own, so that
// the unit of the existing response, gb_tdi_on_the_fly.cu, compiles as before.

#include "gb_dffr.hh"
#include "lat_dffr_kernels.hh"

#include <stdexcept>
#include <string>

namespace {

template <int Rows>
using GBDFFRTile = DFFRTile<GBDFFRSource, Rows>;

// Launch inputs of the galactic-binary tile, with the checks of dffr_tile_inputs and the parameter
// count. Host only.
DFFRTileInputs<GBDFFRSource> gb_dffr_inputs(KinematicsTable *kinematics, STFTFresnel *fresnel,
                                            STFTDomain *stft, int nparams, double t_ref,
                                            int n_side_bins, int tdi_generation,
                                            bool linear_envelope, bool first_order_delays,
                                            STFTDomain **stft_launch)
{
    if (nparams != GBDFFRSource::num_params)
        throw std::invalid_argument("gb_dffr: a galactic binary has 9 parameters, got " +
                                    std::to_string(nparams) + ".");
    GBDFFRSource::Model model;
    model.t_ref = t_ref;
    return dffr_tile_inputs<GBDFFRSource>(fresnel, stft, kinematics, model, n_side_bins,
                                          tdi_generation, linear_envelope, first_order_delays,
                                          stft_launch);
}

// * True for the 16-row tile of the two-template consumers, whose default is 32 rows; on the CPU
// * both are one row.
bool gb_dffr_half_rows(int rows_per_pass)
{
    if (rows_per_pass != 16 && rows_per_pass != 32)
        throw std::invalid_argument("gb_dffr: rows_per_pass must be 16 or 32, got " +
                                    std::to_string(rows_per_pass) + ".");
    return rows_per_pass == 16;
}

// * True for the 8-row Fisher tables, whose default is 4 rows; on the CPU both are one row.
bool gb_dffr_fisher_double_rows(int rows_per_pass)
{
    if (rows_per_pass != 4 && rows_per_pass != 8)
        throw std::invalid_argument("gb_dffr: rows_per_pass of the Fisher matrix must be 4 or 8, "
                                    "got " + std::to_string(rows_per_pass) + ".");
    return rows_per_pass == 8;
}

}  // namespace

// ! Each entry point checks its arguments before it returns on an empty batch, and launches
// ! nothing for zero sources: a grid of zero blocks is an error on the GPU.

void gb_dffr_get_ll_wrap(
    cmplx *d_h_out, cmplx *h_h_out,
    KinematicsTable *kinematics, STFTFresnel *fresnel, STFTDomain *stft,
    double *params_all, int *data_index_all, int *noise_index_all,
    int num_bin, int nparams, double t_ref, int n_side_bins, int tdi_generation,
    bool linear_envelope, bool first_order_delays,
    int *start_freq_inds)
{
    STFTDomain *stft_launch = nullptr;
    const auto inputs = gb_dffr_inputs(kinematics, fresnel, stft, nparams, t_ref, n_side_bins,
                                       tdi_generation, linear_envelope, first_order_delays,
                                       &stft_launch);
    if (num_bin < 1)
        return;
    stft_get_ll_launch<GBDFFRTile<DFFR_ROWS_PER_PASS>>(
        d_h_out, h_h_out, inputs, stft_launch, params_all, data_index_all, noise_index_all,
        num_bin, nparams, n_side_bins, start_freq_inds);
}

void gb_dffr_fill_global_wrap(
    cmplx *template_fill,
    KinematicsTable *kinematics, STFTFresnel *fresnel, STFTDomain *stft,
    double *params_all, int *data_index_all, double *factors_all,
    int num_bin, int nparams, double t_ref, int n_side_bins, int tdi_generation,
    bool linear_envelope, bool first_order_delays,
    int *start_freq_inds)
{
    STFTDomain *stft_launch = nullptr;
    const auto inputs = gb_dffr_inputs(kinematics, fresnel, stft, nparams, t_ref, n_side_bins,
                                       tdi_generation, linear_envelope, first_order_delays,
                                       &stft_launch);
    if (num_bin < 1)
        return;
    // ? The STFT grid is the active band, so the fill writes the active-band layout.
    stft_fill_global_launch<GBDFFRTile<DFFR_ROWS_PER_PASS>>(
        template_fill, inputs, stft_launch, params_all, data_index_all, factors_all, num_bin,
        nparams, n_side_bins, true, start_freq_inds);
}

void gb_dffr_swap_ll_wrap(
    cmplx *d_h_add_out, cmplx *d_h_remove_out,
    cmplx *add_add_out, cmplx *remove_remove_out, cmplx *add_remove_out,
    KinematicsTable *kinematics, STFTFresnel *fresnel, STFTDomain *stft,
    double *params_add_all, double *params_remove_all,
    int *data_index_all, int *noise_index_all,
    int num_bin, int nparams, double t_ref, int n_side_bins, int tdi_generation,
    bool linear_envelope, bool first_order_delays,
    int rows_per_pass, int *start_freq_inds)
{
    STFTDomain *stft_launch = nullptr;
    const auto inputs = gb_dffr_inputs(kinematics, fresnel, stft, nparams, t_ref, n_side_bins,
                                       tdi_generation, linear_envelope, first_order_delays,
                                       &stft_launch);
    const bool half_rows = gb_dffr_half_rows(rows_per_pass);
    if (num_bin < 1)
        return;
    if (half_rows)
        stft_swap_ll_launch<GBDFFRTile<DFFR_ROWS_PER_PASS_HALF>>(
            d_h_add_out, d_h_remove_out, add_add_out, remove_remove_out, add_remove_out, inputs,
            stft_launch, params_add_all, params_remove_all, data_index_all, noise_index_all,
            num_bin, nparams, n_side_bins, start_freq_inds);
    else
        stft_swap_ll_launch<GBDFFRTile<DFFR_ROWS_PER_PASS>>(
            d_h_add_out, d_h_remove_out, add_add_out, remove_remove_out, add_remove_out, inputs,
            stft_launch, params_add_all, params_remove_all, data_index_all, noise_index_all,
            num_bin, nparams, n_side_bins, start_freq_inds);
}

void gb_dffr_get_fstat_ll_wrap(
    double *N_re_out, double *N_im_out,
    double *M_re_out, double *M_im_out,
    KinematicsTable *kinematics, STFTFresnel *fresnel, STFTDomain *stft,
    double *params_all, int *data_index_all, int *noise_index_all,
    int num_bin, int nparams, double t_ref, int n_side_bins, int tdi_generation,
    bool linear_envelope, bool first_order_delays,
    int rows_per_pass, int *start_freq_inds)
{
    STFTDomain *stft_launch = nullptr;
    const auto inputs = gb_dffr_inputs(kinematics, fresnel, stft, nparams, t_ref, n_side_bins,
                                       tdi_generation, linear_envelope, first_order_delays,
                                       &stft_launch);
    const bool half_rows = gb_dffr_half_rows(rows_per_pass);
    if (num_bin < 1)
        return;
    if (half_rows)
        stft_get_fstat_ll_launch<GBDFFRTile<DFFR_ROWS_PER_PASS_HALF>>(
            N_re_out, N_im_out, M_re_out, M_im_out, inputs, stft_launch, params_all,
            data_index_all, noise_index_all, num_bin, nparams, n_side_bins, start_freq_inds);
    else
        stft_get_fstat_ll_launch<GBDFFRTile<DFFR_ROWS_PER_PASS>>(
            N_re_out, N_im_out, M_re_out, M_im_out, inputs, stft_launch, params_all,
            data_index_all, noise_index_all, num_bin, nparams, n_side_bins, start_freq_inds);
}

void gb_dffr_get_ll_grad_wrap(
    double *grad_out,
    KinematicsTable *kinematics, STFTFresnel *fresnel, STFTDomain *stft,
    double *params_all, int *data_index_all, int *noise_index_all,
    double *param_eps,
    int num_bin, int nparams, double t_ref, int n_side_bins, int tdi_generation,
    bool linear_envelope, bool first_order_delays,
    int *start_freq_inds)
{
    STFTDomain *stft_launch = nullptr;
    const auto inputs = gb_dffr_inputs(kinematics, fresnel, stft, nparams, t_ref, n_side_bins,
                                       tdi_generation, linear_envelope, first_order_delays,
                                       &stft_launch);
    if (num_bin < 1)
        return;
    stft_get_ll_grad_launch<GBDFFRTile<DFFR_ROWS_PER_PASS>>(
        grad_out, inputs, stft_launch, params_all, data_index_all, noise_index_all, param_eps,
        num_bin, nparams, n_side_bins, start_freq_inds);
}

void gb_dffr_swap_ll_grad_wrap(
    double *grad_add_out, double *grad_remove_out,
    KinematicsTable *kinematics, STFTFresnel *fresnel, STFTDomain *stft,
    double *params_add_all, double *params_remove_all,
    int *data_index_all, int *noise_index_all,
    double *param_eps_add, double *param_eps_remove,
    int num_bin, int nparams, double t_ref, int n_side_bins, int tdi_generation,
    bool linear_envelope, bool first_order_delays,
    int rows_per_pass, int *start_freq_inds)
{
    STFTDomain *stft_launch = nullptr;
    const auto inputs = gb_dffr_inputs(kinematics, fresnel, stft, nparams, t_ref, n_side_bins,
                                       tdi_generation, linear_envelope, first_order_delays,
                                       &stft_launch);
    const bool half_rows = gb_dffr_half_rows(rows_per_pass);
    if (num_bin < 1)
        return;
    if (half_rows)
        stft_swap_ll_grad_launch<GBDFFRTile<DFFR_ROWS_PER_PASS_HALF>>(
            grad_add_out, grad_remove_out, inputs, stft_launch, params_add_all, params_remove_all,
            data_index_all, noise_index_all, param_eps_add, param_eps_remove, num_bin, nparams,
            n_side_bins, start_freq_inds);
    else
        stft_swap_ll_grad_launch<GBDFFRTile<DFFR_ROWS_PER_PASS>>(
            grad_add_out, grad_remove_out, inputs, stft_launch, params_add_all, params_remove_all,
            data_index_all, noise_index_all, param_eps_add, param_eps_remove, num_bin, nparams,
            n_side_bins, start_freq_inds);
}

void gb_dffr_information_matrix_wrap(
    double *info_out,
    KinematicsTable *kinematics, STFTFresnel *fresnel, STFTDomain *stft,
    double *params_all, int *noise_index_all, int *inds, double *param_eps,
    int num_bin, int nparams, int num_derivs, double t_ref, int n_side_bins, int tdi_generation,
    bool linear_envelope, bool first_order_delays,
    bool easy_central_difference, int rows_per_pass, int *start_freq_inds)
{
    STFTDomain *stft_launch = nullptr;
    const auto inputs = gb_dffr_inputs(kinematics, fresnel, stft, nparams, t_ref, n_side_bins,
                                       tdi_generation, linear_envelope, first_order_delays,
                                       &stft_launch);
    const bool double_rows = gb_dffr_fisher_double_rows(rows_per_pass);
    // ? The launcher checks inds and the counts before it returns on an empty batch.
    if (double_rows)
        dffr_information_matrix<GBDFFRSource, DFFR_FISHER_ROWS_PER_PASS_DOUBLE>(
            info_out, inputs, stft_launch, params_all, noise_index_all, inds, param_eps, num_bin,
            nparams, num_derivs, easy_central_difference, start_freq_inds);
    else
        dffr_information_matrix<GBDFFRSource, DFFR_FISHER_ROWS_PER_PASS>(
            info_out, inputs, stft_launch, params_all, noise_index_all, inds, param_eps, num_bin,
            nparams, num_derivs, easy_central_difference, start_freq_inds);
}

void gb_dffr_fill_template_wrap(
    cmplx *template_out, int *start_out,
    KinematicsTable *kinematics, STFTFresnel *fresnel, STFTDomain *stft,
    double *params_all, int num_bin, int nparams, double t_ref, int n_side_bins,
    int tdi_generation, bool linear_envelope, bool first_order_delays)
{
    STFTDomain *stft_launch = nullptr;
    const auto inputs = gb_dffr_inputs(kinematics, fresnel, stft, nparams, t_ref, n_side_bins,
                                       tdi_generation, linear_envelope, first_order_delays,
                                       &stft_launch);
    dffr_fill_template<GBDFFRSource, DFFR_ROWS_PER_PASS>(template_out, start_out, inputs,
                                                         params_all, num_bin, nparams);
}

void gb_dffr_segment_table_wrap(
    double *effective_frequency, double *chirp_rate, cmplx *reference_phasor,
    cmplx *link_amplitude, cmplx *envelope_slope, cmplx *emission_phasor,
    double *emission_frequency, cmplx *tdi_weights, int *start_freq_ind,
    KinematicsTable *kinematics, STFTFresnel *fresnel, STFTDomain *stft,
    double *params_all, int num_bin, int nparams, double t_ref, int n_side_bins,
    int tdi_generation, bool linear_envelope, bool first_order_delays)
{
    STFTDomain *stft_launch = nullptr;
    const auto inputs = gb_dffr_inputs(kinematics, fresnel, stft, nparams, t_ref, n_side_bins,
                                       tdi_generation, linear_envelope, first_order_delays,
                                       &stft_launch);
    DFFRSegmentTableOutput output;
    output.effective_frequency = effective_frequency;
    output.chirp_rate = chirp_rate;
    output.reference_phasor = reference_phasor;
    output.link_amplitude = link_amplitude;
    output.envelope_slope = envelope_slope;
    output.emission_phasor = emission_phasor;
    output.emission_frequency = emission_frequency;
    output.tdi_weights = tdi_weights;
    output.start_freq_ind = start_freq_ind;
    dffr_segment_table<GBDFFRSource, DFFR_ROWS_PER_PASS>(output, inputs, params_all, num_bin,
                                                         nparams);
}
