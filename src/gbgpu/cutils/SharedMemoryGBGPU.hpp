#ifndef __SHAREDMEMORY_GBGPU_HPP__
#define __SHAREDMEMORY_GBGPU_HPP__

#include "global.h"

#ifdef __CUDACC__
#include <curand_kernel.h>

#define CUDA_HOSTDEV __host__ __device__
#define CUDA_DEV __device__
#else
#define CUDA_HOSTDEV
#define CUDA_DEV
#endif

#define TDI_CHANNEL_SETUP_XYZ 1
#define TDI_CHANNEL_SETUP_AET 2
#define TDI_CHANNEL_SETUP_AE 3

#define ARRAY_TYPE_DATA 1
#define ARRAY_TYPE_TEMPLATE 2

void SharedMemoryWaveComp(
    cmplx* tdi_out,
    int* start_inds_out,
    double* amp, 
    double* f0, 
    double* fdot0, 
    double* fddot0, 
    double* phi0, 
    double* iota,
    double* psi, 
    double* lam,
    double* theta, 
    double T,
    double dt,
    int N, 
    int num_bin_all, 
    int tdi_channel_setup,
    double* Ps,
    double L_arm,
    bool tdi2
);

typedef struct InputInfoTag{
    double* amp;
    double* f0;
    double* fdot0;
    double* fddot0;
    double* phi0;
    double* iota;
    double* psi;
    double* lam;
    double* theta;
    double T;
    double dt;
    int N;
    int num_bin_all;
    cmplx *tdi_out;
    cmplx *d_h;
    cmplx *h_h;
    cmplx *data_arr;
    double *noise; // invC
    int *data_index;
    int *noise_index;
    int start_freq_ind;
    int *start_freq_inds;
    int data_length;
    cmplx* d_h_remove;
    cmplx* d_h_add;
    cmplx* remove_remove;
    cmplx* add_add;
    cmplx* add_remove;
    double* amp_add;
    double* f0_add; 
    double* fdot0_add; 
    double* fddot0_add; 
    double* phi0_add; 
    double* iota_add;
    double* psi_add; 
    double* lam_add;
    double* theta_add;
    double* amp_remove; 
    double* f0_remove; 
    double* fdot0_remove; 
    double* fddot0_remove; 
    double* phi0_remove; 
    double* iota_remove;
    double* psi_remove; 
    double* lam_remove;
    double* theta_remove;
    cmplx* h2_h2;
    cmplx* h1_h1;
    cmplx* h1_h2;
    int device;
    bool do_synchronize;
    double* factors;
    cmplx *L_contribution;
    cmplx *p_contribution;
    double *prior_all_curr;
    double *factors_all;
    double *random_val_all;
    int *band_start_bin_ind;
    int *band_num_bins;
    int *band_start_data_ind;
    int *band_data_lengths;
    double *band_inv_temperatures_all;
    bool *accepted_out;
    int num_bands;
    int max_data_store_size;
    bool is_rj;
    double snr_lim;
    int num_swap_setups;
    int min_val;
    int max_val;
    int tdi_channel_setup;
    int num_data; 
    int num_noise;
    cmplx* M_mat;
    cmplx* N_arr;
    int *start_inds_out;
    double *Ps;
    double L_arm;
    bool tdi2;
} InputInfo; 

void SharedMemoryLikeComp(
    cmplx* d_h,
    cmplx* h_h,
    cmplx* data,
    double* noise,
    int* data_index,
    int* noise_index,
    double* amp, 
    double* f0, 
    double* fdot0, 
    double* fddot0, 
    double* phi0, 
    double* iota,
    double* psi, 
    double* lam,
    double* theta,
    double T, 
    double dt,
    int N,
    int num_bin_all,
    int *start_freq_inds,
    int data_length,
    int tdi_channel_setup,
    int device,
    bool do_synchronize,
    int num_data,
    int num_noise,
    double* Ps,
    double L_arm,
    bool tdi2
);


void SharedMemorySwapLikeComp(
    cmplx* d_h_remove,
    cmplx* d_h_add,
    cmplx* remove_remove,
    cmplx* add_add,
    cmplx* add_remove,
    cmplx* data,
    double* noise,
    int* data_index,
    int* noise_index,
    double* amp_add, 
    double* f0_add, 
    double* fdot0_add, 
    double* fddot0_add, 
    double* phi0_add, 
    double* iota_add,
    double* psi_add, 
    double* lam_add,
    double* theta_add,
    double* amp_remove, 
    double* f0_remove, 
    double* fdot0_remove, 
    double* fddot0_remove, 
    double* phi0_remove, 
    double* iota_remove,
    double* psi_remove, 
    double* lam_remove,
    double* theta_remove,
    double T, 
    double dt,
    int N,
    int num_bin_all,
    int *start_freq_inds,
    int data_length,
    int tdi_channel_setup,
    int device,
    bool do_synchronize,
    int num_data,
    int num_noise,
    double* Ps,
    double L_arm,
    bool tdi2
);

void SharedMemoryChiSquaredComp(
    cmplx *h1_h1,
    cmplx *h2_h2,
    cmplx *h1_h2,
    double *noise,
    int *noise_index,
    double *amp,
    double *f0,
    double *fdot0,
    double *fddot0,
    double *phi0,
    double *iota,
    double *psi,
    double *lam,
    double *theta,
    double T,
    double dt,
    int N,
    int num_bin_all,
    int *start_freq_inds,
    int data_length,
    int tdi_channel_setup,
    int device,
    bool do_synchronize,
    int num_data, 
    int num_noise,
    double* Ps,
    double L_arm,
    bool tdi2
);


void SharedMemoryGenerateGlobal(
    cmplx* data,
    int* data_index,
    double* factors,
    double* amp, 
    double* f0, 
    double* fdot0, 
    double* fddot0, 
    double* phi0, 
    double* iota,
    double* psi, 
    double* lam,
    double* theta,
    double T, 
    double dt,
    int N,
    int num_bin_all,
    int *start_freq_inds,
    int data_length,
    int tdi_channel_setup,
    int device,
    bool do_synchronize,
    double* Ps,
    double L_arm,
    bool tdi2
);


void SharedMemoryFstatLikeComp(
    cmplx *M_mat,
    cmplx *N_arr,
    cmplx *data,
    double *noise,
    int *data_index,
    int *noise_index,
    double *f0,
    double *fdot0,
    double *fddot0,
    double *lam,
    double *theta,
    double T,
    double dt,
    int N,
    int num_bin_all,
    int *start_freq_inds,
    int data_length,
    int tdi_channel_setup,
    int device,
    bool do_synchronize,
    int num_data,
    int num_noise,
    double* Ps, 
    double L_arm,
    bool tdi2
);

#endif // __SHAREDMEMORY_GBGPU_HPP__