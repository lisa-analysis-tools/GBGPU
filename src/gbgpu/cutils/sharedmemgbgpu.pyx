import numpy as np
cimport numpy as np
from libc.stdint cimport uintptr_t
from gpubackendtools import wrapper
from libcpp cimport bool
assert sizeof(int) == sizeof(np.int32_t)

cdef extern from "SharedMemoryGBGPU.hpp":
    ctypedef void* cmplx 'cmplx'
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
        bool tdi2,
        int window_type,
        double window_alpha
    ) except+

    void SharedMemoryLikeComp(
        cmplx* d_h,
        cmplx* h_h,
        cmplx* data,
        cmplx* noise,
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
        bool tdi2,
        int window_type,
        double window_alpha
    ) except+

    void SharedMemorySwapLikeComp(
        cmplx* d_h_remove,
        cmplx* d_h_add,
        cmplx* remove_remove,
        cmplx* add_add,
        cmplx* add_remove,
        cmplx* data,
        cmplx* noise,
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
        bool tdi2,
        int window_type,
        double window_alpha
    ) except+

    void SharedMemoryChiSquaredComp(
        cmplx *h1_h1,
        cmplx *h2_h2,
        cmplx *h1_h2,
        cmplx *noise,
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
        bool tdi2,
        int window_type,
        double window_alpha
    ) except+

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
        bool tdi2,
        int window_type,
        double window_alpha
    ) except+

    void SharedMemoryFstatLikeComp(
        cmplx* M_mat,
        cmplx* N_arr,
        cmplx* data,
        cmplx* noise,
        int* data_index,
        int* noise_index,
        double* f0, 
        double* fdot0, 
        double* fddot0, 
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
        bool tdi2,
        int window_type,
        double window_alpha
    ) except+

    void SharedMemoryInfoMatComp(
        double *info_mat,
        cmplx *d_dh_workspace,
        cmplx *noise,
        int *noise_index,
        int *inds,
        double *amp,
        double *f0,
        double *fdot0,
        double *fddot0,
        double *phi0,
        double *iota,
        double *psi,
        double *lam,
        double *theta,
        double *eps_scaled,
        double eps_orig,
        double T,
        double dt,
        int N,
        int num_bin_all,
        int num_derivs,
        int *start_freq_inds,
        int data_length,
        int tdi_channel_setup,
        int device,
        bool do_synchronize,
        int num_noise,
        double *Ps,
        double L_arm,
        bool tdi2,
        bool easy_central_difference,
        int window_type,
        double window_alpha
    ) except+

def SharedMemoryWaveComp_wrap(*args, **kwargs):
    (
        tdi_out,
        start_inds_out,
        amp, 
        f0, 
        fdot0, 
        fddot0, 
        phi0, 
        iota, 
        psi, 
        lam, 
        theta,
        T,
        dt, 
        N,
        num_bin_all, 
        tdi_channel_setup,
        Ps,
        L_arm,
        tdi2,
        window_type,
        window_alpha
    ), tkwargs = wrapper(*args, **kwargs)

    cdef size_t tdi_out_in = tdi_out
    cdef size_t start_inds_out_in = start_inds_out
    cdef size_t amp_in = amp
    cdef size_t f0_in = f0
    cdef size_t fdot0_in = fdot0
    cdef size_t fddot0_in = fddot0
    cdef size_t phi0_in = phi0
    cdef size_t iota_in = iota
    cdef size_t psi_in = psi
    cdef size_t lam_in = lam
    cdef size_t theta_in = theta
    cdef size_t Ps_in = Ps

    SharedMemoryWaveComp(
        <cmplx *> tdi_out_in,
        <int *> start_inds_out_in,
        <double *>amp_in, 
        <double *>f0_in, 
        <double *>fdot0_in, 
        <double *>fddot0_in, 
        <double *>phi0_in, 
        <double *>iota_in,
        <double *>psi_in, 
        <double *>lam_in,
        <double *>theta_in,
        T, 
        dt,
        N,
        num_bin_all,
        tdi_channel_setup,
        <double *>Ps_in,
        L_arm,
        tdi2,
        window_type,
        window_alpha
    )


def SharedMemoryLikeComp_wrap(*args, **kwargs):
    (
        d_h,
        h_h,
        data,
        noise,
        data_index, 
        noise_index,
        amp, 
        f0, 
        fdot0, 
        fddot0, 
        phi0, 
        iota, 
        psi, 
        lam, 
        theta,
        T,
        dt, 
        N,
        num_bin_all,
        start_freq_inds,
        data_length,
        tdi_channel_setup,
        device,
        do_synchronize,
        num_data,
        num_noise,
        Ps,
        L_arm,
        tdi2,
        window_type,
        window_alpha
    ), tkwargs = wrapper(*args, **kwargs)
    

    cdef size_t d_h_in = d_h
    cdef size_t h_h_in = h_h
    cdef size_t data_in = data
    cdef size_t noise_in = noise
    cdef size_t data_index_in = data_index
    cdef size_t noise_index_in = noise_index
    cdef size_t amp_in = amp
    cdef size_t f0_in = f0
    cdef size_t fdot0_in = fdot0
    cdef size_t fddot0_in = fddot0
    cdef size_t phi0_in = phi0
    cdef size_t iota_in = iota
    cdef size_t psi_in = psi
    cdef size_t lam_in = lam
    cdef size_t theta_in = theta
    cdef size_t start_freq_inds_in = start_freq_inds
    cdef size_t Ps_in = Ps

    SharedMemoryLikeComp(
        <cmplx *> d_h_in,
        <cmplx *> h_h_in,
        <cmplx *> data_in,
        <cmplx *> noise_in,
        <int*> data_index_in,
        <int*> noise_index_in,
        <double *>amp_in, 
        <double *>f0_in, 
        <double *>fdot0_in, 
        <double *>fddot0_in, 
        <double *>phi0_in, 
        <double *>iota_in,
        <double *>psi_in, 
        <double *>lam_in,
        <double *>theta_in,
        T, 
        dt,
        N,
        num_bin_all,
        <int*>start_freq_inds_in,
        data_length,
        tdi_channel_setup,
        device,
        do_synchronize,
        num_data,
        num_noise,
        <double *> Ps_in,
        L_arm, 
        tdi2,
        window_type,
        window_alpha
    )


def SharedMemorySwapLikeComp_wrap(*args, **kwargs):
    (
        d_h_remove,
        d_h_add,
        remove_remove,
        add_add,
        add_remove,
        data,
        noise,
        data_index, 
        noise_index,
        amp_add, 
        f0_add, 
        fdot0_add, 
        fddot0_add, 
        phi0_add, 
        iota_add, 
        psi_add, 
        lam_add, 
        theta_add,
        amp_remove, 
        f0_remove, 
        fdot0_remove, 
        fddot0_remove, 
        phi0_remove, 
        iota_remove, 
        psi_remove, 
        lam_remove, 
        theta_remove,
        T,
        dt, 
        N,
        num_bin_all,
        start_freq_inds,
        data_length,
        tdi_channel_setup,
        device,
        do_synchronize,
        num_data,
        num_noise,
        Ps,
        L_arm,
        tdi2,
        window_type,
        window_alpha
    ), tkwargs = wrapper(*args, **kwargs)

    cdef size_t d_h_remove_in = d_h_remove
    cdef size_t d_h_add_in = d_h_add
    cdef size_t remove_remove_in = remove_remove
    cdef size_t add_add_in = add_add
    cdef size_t add_remove_in = add_remove
    cdef size_t data_in = data
    cdef size_t noise_in = noise
    cdef size_t data_index_in = data_index
    cdef size_t noise_index_in = noise_index
    cdef size_t amp_add_in = amp_add
    cdef size_t f0_add_in = f0_add
    cdef size_t fdot0_add_in = fdot0_add
    cdef size_t fddot0_add_in = fddot0_add
    cdef size_t phi0_add_in = phi0_add
    cdef size_t iota_add_in = iota_add
    cdef size_t psi_add_in = psi_add
    cdef size_t lam_add_in = lam_add
    cdef size_t theta_add_in = theta_add
    cdef size_t amp_remove_in = amp_remove
    cdef size_t f0_remove_in = f0_remove
    cdef size_t fdot0_remove_in = fdot0_remove
    cdef size_t fddot0_remove_in = fddot0_remove
    cdef size_t phi0_remove_in = phi0_remove
    cdef size_t iota_remove_in = iota_remove
    cdef size_t psi_remove_in = psi_remove
    cdef size_t lam_remove_in = lam_remove
    cdef size_t theta_remove_in = theta_remove
    cdef size_t start_freq_inds_in = start_freq_inds
    cdef size_t Ps_in = Ps

    SharedMemorySwapLikeComp(
        <cmplx *> d_h_remove_in,
        <cmplx *> d_h_add_in,
        <cmplx *> remove_remove_in,
        <cmplx *> add_add_in,
        <cmplx *> add_remove_in,
        <cmplx *> data_in,
        <cmplx *> noise_in,
        <int*> data_index_in,
        <int*> noise_index_in,
        <double *>amp_add_in, 
        <double *>f0_add_in, 
        <double *>fdot0_add_in, 
        <double *>fddot0_add_in, 
        <double *>phi0_add_in, 
        <double *>iota_add_in,
        <double *>psi_add_in, 
        <double *>lam_add_in,
        <double *>theta_add_in,
        <double *>amp_remove_in, 
        <double *>f0_remove_in, 
        <double *>fdot0_remove_in, 
        <double *>fddot0_remove_in, 
        <double *>phi0_remove_in, 
        <double *>iota_remove_in,
        <double *>psi_remove_in, 
        <double *>lam_remove_in,
        <double *>theta_remove_in,
        T, 
        dt,
        N,
        num_bin_all,
        <int*> start_freq_inds_in,
        data_length,
        tdi_channel_setup,
        device,
        do_synchronize,
        num_data,
        num_noise,
        <double *> Ps_in,
        L_arm, 
        tdi2,
        window_type,
        window_alpha
    )

def SharedMemoryChiSquaredComp_wrap(*args, **kwargs):
    (
        h1_h1,
        h2_h2,
        h1_h2,
        noise,
        noise_index,
        amp, 
        f0, 
        fdot0, 
        fddot0, 
        phi0, 
        iota, 
        psi, 
        lam, 
        theta,
        T,
        dt, 
        N,
        num_bin_all,
        start_freq_inds,
        data_length,
        tdi_channel_setup,
        device,
        do_synchronize,
        num_data, 
        num_noise,
        Ps,
        L_arm,
        tdi2,
        window_type,
        window_alpha
    ), tkwargs = wrapper(*args, **kwargs)

    cdef size_t h1_h1_in = h1_h1
    cdef size_t h2_h2_in = h2_h2
    cdef size_t h1_h2_in = h1_h2
    cdef size_t noise_in = noise
    cdef size_t noise_index_in = noise_index
    cdef size_t amp_in = amp
    cdef size_t f0_in = f0
    cdef size_t fdot0_in = fdot0
    cdef size_t fddot0_in = fddot0
    cdef size_t phi0_in = phi0
    cdef size_t iota_in = iota
    cdef size_t psi_in = psi
    cdef size_t lam_in = lam
    cdef size_t theta_in = theta
    cdef size_t start_freq_inds_in = start_freq_inds
    cdef size_t Ps_in = Ps

    SharedMemoryChiSquaredComp(
        <cmplx *> h1_h1_in,
        <cmplx *> h2_h2_in,
        <cmplx *> h1_h2_in,
        <cmplx *> noise_in,
        <int*> noise_index_in,
        <double *>amp_in, 
        <double *>f0_in, 
        <double *>fdot0_in, 
        <double *>fddot0_in, 
        <double *>phi0_in, 
        <double *>iota_in,
        <double *>psi_in, 
        <double *>lam_in,
        <double *>theta_in,
        T, 
        dt,
        N,
        num_bin_all,
        <int*> start_freq_inds_in,
        data_length,
        tdi_channel_setup,
        device,
        do_synchronize,
        num_data, 
        num_noise,
        <double *> Ps_in,
        L_arm,
        tdi2,
        window_type,
        window_alpha
    )

def SharedMemoryGenerateGlobal_wrap(*args, **kwargs):
    (   
        data,
        data_index,
        factors,
        amp, 
        f0, 
        fdot0, 
        fddot0, 
        phi0, 
        iota, 
        psi, 
        lam, 
        theta,
        T,
        dt, 
        N,
        num_bin_all,
        start_freq_inds,
        data_length,
        tdi_channel_setup,
        device,
        do_synchronize, 
        Ps,
        L_arm,
        tdi2,
        window_type,
        window_alpha
    ), tkwargs = wrapper(*args, **kwargs)

    cdef size_t data_in = data
    cdef size_t data_index_in = data_index
    cdef size_t amp_in = amp
    cdef size_t f0_in = f0
    cdef size_t fdot0_in = fdot0
    cdef size_t fddot0_in = fddot0
    cdef size_t phi0_in = phi0
    cdef size_t iota_in = iota
    cdef size_t psi_in = psi
    cdef size_t lam_in = lam
    cdef size_t theta_in = theta
    cdef size_t factors_in = factors
    cdef size_t start_freq_inds_in = start_freq_inds
    cdef size_t Ps_in = Ps

    SharedMemoryGenerateGlobal(
        <cmplx *> data_in,
        <int*> data_index_in,
        <double *> factors_in,
        <double *>amp_in, 
        <double *>f0_in, 
        <double *>fdot0_in, 
        <double *>fddot0_in, 
        <double *>phi0_in, 
        <double *>iota_in,
        <double *>psi_in, 
        <double *>lam_in,
        <double *>theta_in,
        T, 
        dt,
        N,
        num_bin_all,
        <int*>start_freq_inds_in,
        data_length,
        tdi_channel_setup,
        device,
        do_synchronize,
        <double *> Ps_in,
        L_arm,
        tdi2,
        window_type,
        window_alpha
    )

def SharedMemoryFstatLikeComp_wrap(*args, **kwargs):
    (
        M_mat,
        N_arr,
        data,
        noise,
        data_index, 
        noise_index,
        f0, 
        fdot0, 
        fddot0, 
        lam, 
        theta,
        T,
        dt, 
        N,
        num_bin_all,
        start_freq_inds,
        data_length,
        tdi_channel_setup,
        device,
        do_synchronize,
        num_data,
        num_noise,
        Ps,
        L_arm,
        tdi2,
        window_type,
        window_alpha
    ), tkwargs = wrapper(*args, **kwargs)

    cdef size_t M_mat_in = M_mat
    cdef size_t N_arr_in = N_arr
    cdef size_t data_in = data
    cdef size_t noise_in = noise
    cdef size_t data_index_in = data_index
    cdef size_t noise_index_in = noise_index
    cdef size_t f0_in = f0
    cdef size_t fdot0_in = fdot0
    cdef size_t fddot0_in = fddot0
    cdef size_t lam_in = lam
    cdef size_t theta_in = theta
    cdef size_t start_freq_inds_in = start_freq_inds
    cdef size_t Ps_in = Ps

    SharedMemoryFstatLikeComp(
        <cmplx *> M_mat_in,
        <cmplx *> N_arr_in,
        <cmplx *> data_in,
        <cmplx *> noise_in,
        <int*> data_index_in,
        <int*> noise_index_in,
        <double *>f0_in, 
        <double *>fdot0_in, 
        <double *>fddot0_in, 
        <double *>lam_in,
        <double *>theta_in,
        T, 
        dt,
        N,
        num_bin_all,
        <int*> start_freq_inds_in,
        data_length,
        tdi_channel_setup,
        device,
        do_synchronize,
        num_data,
        num_noise,
        <double *> Ps_in,
        L_arm,
        tdi2,
        window_type,
        window_alpha
    )

def SharedMemoryInfoMatComp_wrap(*args, **kwargs):
    (
        info_mat,
        d_dh_workspace,
        noise,
        noise_index,
        inds,
        amp, 
        f0, 
        fdot0, 
        fddot0, 
        phi0, 
        iota, 
        psi, 
        lam, 
        theta,
        eps_scaled, 
        eps_orig, 
        T, 
        dt, 
        N, 
        num_bin_all, 
        num_derivs, 
        start_freq_inds,
        data_length, 
        tdi_channel_setup, 
        device, 
        do_synchronize,
        num_noise, 
        Ps, 
        L_arm, 
        tdi2, 
        easy_central_difference,
        window_type, 
        window_alpha
    ), tkwargs = wrapper(*args, **kwargs)

    cdef size_t info_mat_in = info_mat
    cdef size_t d_dh_workspace_in = d_dh_workspace
    cdef size_t noise_in = noise
    cdef size_t noise_index_in = noise_index
    cdef size_t inds_in = inds
    cdef size_t amp_in = amp
    cdef size_t f0_in = f0
    cdef size_t fdot0_in = fdot0
    cdef size_t fddot0_in = fddot0
    cdef size_t phi0_in = phi0
    cdef size_t iota_in = iota
    cdef size_t psi_in = psi
    cdef size_t lam_in = lam
    cdef size_t theta_in = theta
    cdef size_t eps_scaled_in = eps_scaled
    cdef size_t start_freq_inds_in = start_freq_inds
    cdef size_t Ps_in = Ps

    SharedMemoryInfoMatComp(
        <double *>info_mat_in,
        <cmplx *>d_dh_workspace_in,
        <cmplx *>noise_in,
        <int *>noise_index_in,
        <int *>inds_in,
        <double *>amp_in, 
        <double *>f0_in, 
        <double *>fdot0_in, 
        <double *>fddot0_in, 
        <double *>phi0_in, 
        <double *>iota_in,
        <double *>psi_in, 
        <double *>lam_in,
        <double *>theta_in,
        <double *>eps_scaled_in, 
        eps_orig, 
        T, 
        dt, 
        N, 
        num_bin_all, 
        num_derivs,
        <int *>start_freq_inds_in,
        data_length, 
        tdi_channel_setup, 
        device, 
        do_synchronize, 
        num_noise,
        <double *>Ps_in, 
        L_arm, 
        tdi2, 
        easy_central_difference,
        window_type, 
        window_alpha
    )