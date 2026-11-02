#ifndef __GB_DFFR_HH__
#define __GB_DFFR_HH__

// gb_dffr.hh -- the DFFR response of galactic binaries: GBDFFRSource, the galactic-binary source
// model of LAT's DFFR tile (lat_dffr_kernels.hh), and the host entry points defined in gb_dffr.cu
// on that tile. The binding
// exposes them as the gb_dffr_* methods of GBComputationGroupWrap, next to the gb_stft_* methods of
// the existing response, and the consumers are the same: get_ll, fill_global, swap_ll, the
// F-statistic and both gradients. The response reads the kinematics table of the grid instead of
// the orbits, and the TDI generation (1 or 2) instead of a TDIConfig. rows_per_pass of the
// two-template consumers is 32 or 16 on the GPU; the CPU uses one row.

#include "gbt_global.h"        // cmplx
#include "Detector.hpp"        // KinematicsTable
#include "domains.hpp"         // STFTFresnel, STFTDomain
#include "gb_physics.hh"      // gb_phase, gb_frequency, gb_amplitude_ratio
#include "lat_dffr_kernels.hh" // the source-model contract and the tile

#include <cmath>

// The galactic-binary source model of the DFFR response: the contract of lat_dffr_kernels.hh for
// the nine parameters (amp, f0, fdot, fddot, phi0, iota, psi, lam, beta) of GBTDIonTheFly, with
// one component. The phase and amplitude come from gb_physics.hh, which GBTDIonTheFly evaluates
// too, and the strain is that of LISATDIonTheFly::get_hp_hc:
//   h+ = -A (1 + cos^2 iota) cos(Phi),  hx = -2 A cos(iota) sin(Phi),  rotated by 2 psi.
struct GBDFFRSource {
  static constexpr int num_components = 1;
  static constexpr int num_params = 9;
  // * A, phi0, iota and psi: the amplitude, polarisation weights and the phase offset enter only the
  // * link amplitudes and the phasors, never a frequency or chirp rate.
  static constexpr unsigned reception_invariant_params = (1u << 0) | (1u << 4) | (1u << 5) | (1u << 6);

  // The launch inputs of the model: the epoch of the parameters, in the time frame of the orbits.
  struct Model {
    double t_ref;
  };

  struct Constants {
    DFFRSourceGeometry geometry;
    double amplitude;
    double frequency;
    double frequency_derivative;
    double frequency_second_derivative;
    double initial_phase;
    double plus_weight;
    double cross_weight;
    double time_reference;
  };

  CUDA_DEVICE static void load(Constants *source, const Model &model, const double *params) {
    source->amplitude = params[0];
    source->frequency = params[1];
    source->frequency_derivative = params[2];
    source->frequency_second_derivative = params[3];
    source->initial_phase = params[4];
    dffr_fill_source_geometry(&source->geometry, params[6], params[7], params[8]);
    // * LAT's strain factor -2 folds into the two polarisation weights; it cancels in the
    // * envelope slope, which is a logarithmic derivative.
    const double cos_inclination = std::cos(params[5]);
    source->plus_weight = DFFR_LAT_STRAIN_FACTOR * 0.5 * (1.0 + cos_inclination * cos_inclination);
    source->cross_weight = DFFR_LAT_STRAIN_FACTOR * cos_inclination;
    source->time_reference = model.t_ref;
  }

  CUDA_DEVICE static double time_reference(const Constants &source) {
    return source.time_reference;
  }

  CUDA_DEVICE static double amplitude(const Constants &source) { return source.amplitude; }

  CUDA_DEVICE static double plus_weight(const Constants &source, int component) {
    (void)component;
    return source.plus_weight;
  }

  CUDA_DEVICE static double cross_weight(const Constants &source, int component) {
    (void)component;
    return source.cross_weight;
  }

  CUDA_DEVICE static DFFRQuadPhaseExpansion expand(const Constants &source, int component,
                                                   double retarded_time) {
    (void)component;
    DFFRQuadPhaseExpansion expansion;
    expansion.reference_phase =
        gb_phase(source.frequency, source.frequency_derivative,
                 source.frequency_second_derivative, source.initial_phase, retarded_time);
    expansion.instantaneous_frequency =
        gb_frequency(source.frequency, source.frequency_derivative,
                     source.frequency_second_derivative, retarded_time);
    expansion.frequency_rate = gb_frequency_derivative(
        source.frequency_derivative, source.frequency_second_derivative, retarded_time);
    expansion.amplitude_ratio =
        gb_amplitude_ratio(source.frequency, source.frequency_derivative, retarded_time);
    return expansion;
  }

  CUDA_DEVICE static double amplitude_log_slope(const Constants &source, int component,
                                                double midpoint_offset) {
    (void)component;
    return gb_amplitude_log_slope(source.frequency, source.frequency_derivative, midpoint_offset);
  }
};

void gb_dffr_get_ll_wrap(
    cmplx *d_h_out, cmplx *h_h_out,
    KinematicsTable *kinematics, STFTFresnel *fresnel, STFTDomain *stft,
    double *params_all, int *data_index_all, int *noise_index_all,
    int num_bin, int nparams, double t_ref, int n_side_bins, int tdi_generation,
    bool linear_envelope, bool first_order_delays,
    int *start_freq_inds);

void gb_dffr_fill_global_wrap(
    cmplx *template_fill,
    KinematicsTable *kinematics, STFTFresnel *fresnel, STFTDomain *stft,
    double *params_all, int *data_index_all, double *factors_all,
    int num_bin, int nparams, double t_ref, int n_side_bins, int tdi_generation,
    bool linear_envelope, bool first_order_delays,
    int *start_freq_inds);

void gb_dffr_swap_ll_wrap(
    cmplx *d_h_add_out, cmplx *d_h_remove_out,
    cmplx *add_add_out, cmplx *remove_remove_out, cmplx *add_remove_out,
    KinematicsTable *kinematics, STFTFresnel *fresnel, STFTDomain *stft,
    double *params_add_all, double *params_remove_all,
    int *data_index_all, int *noise_index_all,
    int num_bin, int nparams, double t_ref, int n_side_bins, int tdi_generation,
    bool linear_envelope, bool first_order_delays,
    int rows_per_pass, int *start_freq_inds);

void gb_dffr_get_fstat_ll_wrap(
    double *N_re_out, double *N_im_out,
    double *M_re_out, double *M_im_out,
    KinematicsTable *kinematics, STFTFresnel *fresnel, STFTDomain *stft,
    double *params_all, int *data_index_all, int *noise_index_all,
    int num_bin, int nparams, double t_ref, int n_side_bins, int tdi_generation,
    bool linear_envelope, bool first_order_delays,
    int rows_per_pass, int *start_freq_inds);

void gb_dffr_get_ll_grad_wrap(
    double *grad_out,
    KinematicsTable *kinematics, STFTFresnel *fresnel, STFTDomain *stft,
    double *params_all, int *data_index_all, int *noise_index_all,
    double *param_eps,
    int num_bin, int nparams, double t_ref, int n_side_bins, int tdi_generation,
    bool linear_envelope, bool first_order_delays,
    int *start_freq_inds);

void gb_dffr_swap_ll_grad_wrap(
    double *grad_add_out, double *grad_remove_out,
    KinematicsTable *kinematics, STFTFresnel *fresnel, STFTDomain *stft,
    double *params_add_all, double *params_remove_all,
    int *data_index_all, int *noise_index_all,
    double *param_eps_add, double *param_eps_remove,
    int num_bin, int nparams, double t_ref, int n_side_bins, int tdi_generation,
    bool linear_envelope, bool first_order_delays,
    int rows_per_pass, int *start_freq_inds);

// The Fisher matrices (num_bin, num_derivs, num_derivs) with the estimator of
// gb_stft_information_matrix: steps param_eps[inds[d]], four-point stencil or two-point with
// easy_central_difference. rows_per_pass is 4 or 8 on the GPU; the CPU uses one row.
void gb_dffr_information_matrix_wrap(
    double *info_out,
    KinematicsTable *kinematics, STFTFresnel *fresnel, STFTDomain *stft,
    double *params_all, int *noise_index_all, int *inds, double *param_eps,
    int num_bin, int nparams, int num_derivs, double t_ref, int n_side_bins, int tdi_generation,
    bool linear_envelope, bool first_order_delays,
    bool easy_central_difference, int rows_per_pass, int *start_freq_inds);

// Diagnostics: the compact template (num_bin, 3, NT, 2 n_side_bins + 1) with the band starts
// (num_bin, NT), and every bin-independent quantity of the segment table.
void gb_dffr_fill_template_wrap(
    cmplx *template_out, int *start_out,
    KinematicsTable *kinematics, STFTFresnel *fresnel, STFTDomain *stft,
    double *params_all, int num_bin, int nparams, double t_ref, int n_side_bins,
    int tdi_generation, bool linear_envelope, bool first_order_delays);

void gb_dffr_segment_table_wrap(
    double *effective_frequency, double *chirp_rate, cmplx *reference_phasor,
    cmplx *link_amplitude, cmplx *envelope_slope, cmplx *emission_phasor,
    double *emission_frequency, cmplx *tdi_weights, int *start_freq_ind,
    KinematicsTable *kinematics, STFTFresnel *fresnel, STFTDomain *stft,
    double *params_all, int num_bin, int nparams, double t_ref, int n_side_bins,
    int tdi_generation, bool linear_envelope, bool first_order_delays);

#endif  // __GB_DFFR_HH__
