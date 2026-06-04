// Code by Michael Katz. Based on code by Travis Robson, Neil Cornish, Tyson Littenberg, Stas Babak

// imports
#include "stdio.h"

#include "gbgpu_utils.hh"
#include "global.h"
#include "LISA.h"
#include "cuda_complex.hpp"

#define NUM_THREADS 256

// Add functionality for proper summation in the kernel
#ifdef __CUDACC__
CUDA_DEVICE
double atomicAddDouble(double *address, double val)
{
    unsigned long long *address_as_ull =
        (unsigned long long *)address;
    unsigned long long old = *address_as_ull, assumed;

    do
    {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val +
                                             __longlong_as_double(assumed)));

        // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
    } while (assumed != old);

    return __longlong_as_double(old);
}
#endif

// Add functionality for proper summation in the kernel
// `static inline` gives this internal linkage so it doesn't clash with
// the identical definition in SharedMemoryGBGPU.cu when both compile
// into the same cgbgpu .so (Phase GBGPU.pybind.bulk).
CUDA_DEVICE
static inline void atomicAddComplex(cmplx *a, cmplx b)
{
    // transform the addresses of real and imag. parts to double pointers
    double *x = (double *)a;
    double *y = x + 1;
    // use atomicAdd for double variables

#ifdef __CUDACC__
    atomicAddDouble(x, b.real());
    atomicAddDouble(y, b.imag());
#else
    *x += b.real();
    *y += b.imag();
#endif
}

// calculate batched log likelihood
CUDA_KERNEL
void fill_global(cmplx *A_glob, cmplx *E_glob, cmplx *A_template, cmplx *E_template, int *start_ind_all, int M, int num_bin, int *group_index, int data_length)
{
    // prepare loop based on CPU/GPU
    int start, end, increment;
#ifdef __CUDACC__

    start = blockIdx.x * blockDim.x + threadIdx.x;
    end = num_bin;
    increment = blockDim.x * gridDim.x;

#else

    start = 0;
    end = num_bin;
    increment = 1;

#endif
    for (int bin_i = start;
         bin_i < end;
         bin_i += increment)
    {

        // get start index in frequency array
        int start_ind = start_ind_all[bin_i];
        int group_i = group_index[bin_i];

        for (int i = 0;
             i < M;
             i += 1)
        {
            int j = start_ind + i;

            if ((j >= data_length) || (j < 0))
            {
                continue;
            }
            cmplx temp_A = A_template[i * num_bin + bin_i];
            cmplx temp_E = E_template[i * num_bin + bin_i];

            int ind_out = group_i * data_length + j;
            atomicAddComplex(&A_glob[ind_out], temp_A);
            atomicAddComplex(&E_glob[ind_out], temp_E);
            // printf("CHECK: %d %e %e %d %d %d %d %d %d\n", bin_i, A_template[i * num_bin + bin_i], temp_A, group_i, data_length, j, num_groups, per_group, i);
        }
    }
}

// wrapper for log likelihood
void fill_global_wrap(cmplx *A_glob, cmplx *E_glob, cmplx *A_template, cmplx *E_template, int *start_ind_all, int M, int num_bin, int *group_index, int data_length)
{
// GPU / CPU difference
#ifdef __CUDACC__

    int num_blocks = std::ceil((num_bin + NUM_THREADS - 1) / NUM_THREADS);

    if (num_blocks == 0) return;

    fill_global<<<num_blocks, NUM_THREADS>>>(
        A_glob, E_glob, A_template, E_template, start_ind_all, M, num_bin, group_index, data_length);
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());

#else

    fill_global(
        A_glob, E_glob, A_template, E_template, start_ind_all, M, num_bin, group_index, data_length);

#endif
}

// calculate batched log likelihood
CUDA_KERNEL
void get_ll(cmplx *d_h, cmplx *h_h, cmplx *A_template, cmplx *E_template, cmplx *A_data, cmplx *E_data, double *A_psd, double *E_psd, double df, int *start_ind_all, int M, int num_bin, int *data_index, int *noise_index, int data_length)
{
    // prepare loop based on CPU/GPU
    int start, end, increment;
#ifdef __CUDACC__

    start = blockIdx.x * blockDim.x + threadIdx.x;
    end = num_bin;
    increment = blockDim.x * gridDim.x;

#else

    start = 0;
    end = num_bin;
    increment = 1;

#endif
    for (int bin_i = start;
         bin_i < end;
         bin_i += increment)
    {

        // get start index in frequency array
        int start_ind = start_ind_all[bin_i];
        int data_index_bin_i = data_index[bin_i];
        int noise_index_bin_i = noise_index[bin_i];

        // initialize likelihood
        cmplx h_h_temp(0.0, 0.0);
        cmplx d_h_temp(0.0, 0.0);
        for (int i = 0;
             i < M;
             i += 1)
        {
            int j = start_ind + i;

            double A_noise = A_psd[noise_index_bin_i * data_length + j];
            double E_noise = E_psd[noise_index_bin_i * data_length + j];

            // calculate h term
            cmplx h_A = A_template[i * num_bin + bin_i];
            cmplx h_E = E_template[i * num_bin + bin_i];

            cmplx d_A = A_data[data_index_bin_i * data_length + j];
            cmplx d_E = E_data[data_index_bin_i * data_length + j];

            // get <d|h> term
            d_h_temp += gcmplx::conj(d_A) * h_A / A_noise;
            d_h_temp += gcmplx::conj(d_E) * h_E / E_noise;

            // <h|h>
            h_h_temp += gcmplx::conj(h_A) * h_A / A_noise;
            h_h_temp += gcmplx::conj(h_E) * h_E / E_noise;
        }

        // read out
        d_h[bin_i] = 4. * df * d_h_temp;
        h_h[bin_i] = 4. * df * h_h_temp;
    }
}

// wrapper for log likelihood
void get_ll_wrap(cmplx *d_h, cmplx *h_h,
                 cmplx *A_template, cmplx *E_template,
                 cmplx *A_data, cmplx *E_data,
                 double *A_psd, double *E_psd, double df,
                 int *start_ind, int M, int num_bin, int *data_index, int *noise_index, int data_length)
{
// GPU / CPU difference
#ifdef __CUDACC__

    int num_blocks = std::ceil((num_bin + NUM_THREADS - 1) / NUM_THREADS);

    get_ll<<<num_blocks, NUM_THREADS>>>(
        d_h, h_h, A_template, E_template, A_data, E_data, A_psd, E_psd, df, start_ind, M, num_bin, data_index, noise_index, data_length);
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());

#else

    get_ll(
        d_h, h_h, A_template, E_template, A_data, E_data, A_psd, E_psd, df, start_ind, M, num_bin, data_index, noise_index, data_length);

#endif
}

// calculate batched log likelihood
CUDA_KERNEL
void swap_ll_diff(cmplx* d_h_remove, cmplx* d_h_add, cmplx* add_remove, cmplx* remove_remove, cmplx* add_add, cmplx* A_remove, cmplx* E_remove, int* start_ind_all_remove, cmplx* A_add, cmplx* E_add, int* start_ind_all_add, cmplx* A_data, cmplx* E_data, double* A_psd, double* E_psd, double df, int M, int num_bin, int* data_index, int* noise_index, int data_length)
{
    // prepare loop based on CPU/GPU
    int start, end, increment;
    #ifdef __CUDACC__

    start = blockIdx.x * blockDim.x + threadIdx.x;
    end = num_bin;
    increment = blockDim.x * gridDim.x;

    #else

    start = 0;
    end = num_bin;
    increment = 1;

    #pragma omp parallel for
    #endif
	for (int bin_i = start;
			 bin_i < end;
			 bin_i += increment)
    {
        
        // get start index in frequency array
        int start_ind_remove = start_ind_all_remove[bin_i];
        int start_ind_add = start_ind_all_add[bin_i];
        int data_index_bin_i = data_index[bin_i];
        int noise_index_bin_i = noise_index[bin_i];

        // initialize likelihood
        cmplx d_h_remove_temp(0.0, 0.0);
        cmplx d_h_add_temp(0.0, 0.0);
        cmplx add_remove_temp(0.0, 0.0);
        cmplx remove_remove_temp(0.0, 0.0);
        cmplx add_add_temp(0.0, 0.0);

        int lower_start_ind, upper_start_ind, lower_end_ind, upper_end_ind;
        bool is_add_lower;
        if (start_ind_remove <= start_ind_add)
        {
            lower_start_ind = start_ind_remove;
            upper_end_ind = start_ind_add + M;

            upper_start_ind = start_ind_add;
            lower_end_ind = start_ind_remove +M;

            is_add_lower = false;
        }
        else
        {
            lower_start_ind = start_ind_add;
            upper_end_ind = start_ind_remove + M;

            upper_start_ind = start_ind_remove;
            lower_end_ind = start_ind_add + M;

            is_add_lower = true;
        }
        int total_i_vals = upper_end_ind - lower_start_ind;

        double A_noise, E_noise;
        cmplx d_A, d_E;

        cmplx h_A, h_E, h_A_add, h_E_add, h_A_remove, h_E_remove;
        int real_ind, real_ind_add, real_ind_remove; 
        
        if (total_i_vals < 2 * M)
        {
            for (int i = 0;
                    i < total_i_vals;
                    i += 1)
            {
                
                int j = lower_start_ind + i;

                
                A_noise = A_psd[noise_index_bin_i * data_length + j];
                E_noise = E_psd[noise_index_bin_i * data_length + j];

                d_A = A_data[data_index_bin_i * data_length + j];
                d_E = E_data[data_index_bin_i * data_length + j];

                
                
                if (j < upper_start_ind)
                {
                    real_ind = i;
                    if (is_add_lower)
                    {
                        
                        h_A = A_add[real_ind *num_bin + bin_i];
                        h_E = E_add[real_ind *num_bin + bin_i];

                        // get <d|h> term
                        d_h_add_temp += gcmplx::conj(d_A) * h_A / A_noise;
                        d_h_add_temp += gcmplx::conj(d_E) * h_E / E_noise;

                        // <h|h>
                        add_add_temp += gcmplx::conj(h_A) * h_A / A_noise;
                        add_add_temp += gcmplx::conj(h_E) * h_E / E_noise;
                    }
                    else
                    {
                        h_A = A_remove[real_ind *num_bin + bin_i];
                        h_E = E_remove[real_ind *num_bin + bin_i];

                        // get <d|h> term
                        d_h_remove_temp += gcmplx::conj(d_A) * h_A / A_noise;
                        d_h_remove_temp += gcmplx::conj(d_E) * h_E / E_noise;

                        // <h|h>
                        remove_remove_temp += gcmplx::conj(h_A) * h_A / A_noise;
                        remove_remove_temp += gcmplx::conj(h_E) * h_E / E_noise;

                        //if ((bin_i == 0)) printf("%d %d %d \n", i, j, upper_start_ind);
                    }
                }
                else if (j >= lower_end_ind)
                {
                    real_ind = j - upper_start_ind;
                    if (!is_add_lower)
                    {

                        h_A_add = A_add[real_ind *num_bin + bin_i];
                        h_E_add = E_add[real_ind *num_bin + bin_i];

                        // get <d|h> term
                        d_h_add_temp += gcmplx::conj(d_A) * h_A_add / A_noise;
                        d_h_add_temp += gcmplx::conj(d_E) * h_E_add / E_noise;

                        // <h|h>
                        add_add_temp += gcmplx::conj(h_A_add) * h_A_add / A_noise;
                        add_add_temp += gcmplx::conj(h_E_add) * h_E_add / E_noise;
                    }
                    else
                    {
                        h_A_remove = A_remove[real_ind *num_bin + bin_i];
                        h_E_remove = E_remove[real_ind *num_bin + bin_i];

                        // get <d|h> term
                        d_h_remove_temp += gcmplx::conj(d_A) * h_A_remove / A_noise;
                        d_h_remove_temp += gcmplx::conj(d_E) * h_E_remove / E_noise;

                        // <h|h>
                        remove_remove_temp += gcmplx::conj(h_A_remove) * h_A_remove / A_noise;
                        remove_remove_temp += gcmplx::conj(h_E_remove) * h_E_remove / E_noise;
                    }
                }
                else // this is where the signals overlap
                {
                    if (is_add_lower)
                    {
                        real_ind_add = i;
                    }
                    else
                    {
                        real_ind_add = j - upper_start_ind;
                    }

                    h_A_add = A_add[real_ind_add *num_bin + bin_i];
                    h_E_add = E_add[real_ind_add *num_bin + bin_i];

                    // get <d|h> term
                    d_h_add_temp += gcmplx::conj(d_A) * h_A_add / A_noise;
                    d_h_add_temp += gcmplx::conj(d_E) * h_E_add / E_noise;

                    // <h|h>
                    add_add_temp += gcmplx::conj(h_A_add) * h_A_add / A_noise;
                    add_add_temp += gcmplx::conj(h_E_add) * h_E_add / E_noise;

                    if (!is_add_lower)
                    {
                        real_ind_remove = i;
                    }
                    else
                    {
                        real_ind_remove = j - upper_start_ind;
                    }
                    
                    h_A_remove = A_remove[real_ind_remove *num_bin + bin_i];
                    h_E_remove = E_remove[real_ind_remove *num_bin + bin_i];

                    // get <d|h> term
                    d_h_remove_temp += gcmplx::conj(d_A) * h_A_remove / A_noise;
                    d_h_remove_temp += gcmplx::conj(d_E) * h_E_remove / E_noise;

                    // <h|h>
                    remove_remove_temp += gcmplx::conj(h_A_remove) * h_A_remove / A_noise;
                    remove_remove_temp += gcmplx::conj(h_E_remove) * h_E_remove / E_noise;

                    add_remove_temp += gcmplx::conj(h_A_add) * h_A_remove / A_noise;
                    add_remove_temp += gcmplx::conj(h_E_add) * h_E_remove / E_noise;
                }
            }
        }
        else
        {
            for (int i = 0;
                    i < M;
                    i += 1)
            {
                
                int j = start_ind_remove + i;

                
                A_noise = A_psd[noise_index_bin_i * data_length + j];
                E_noise = E_psd[noise_index_bin_i * data_length + j];

                d_A = A_data[data_index_bin_i * data_length + j];
                d_E = E_data[data_index_bin_i * data_length + j];

                //if ((bin_i == num_bin - 1))printf("CHECK remove: %d %e %e  \n", i, A_noise, d_A.real());
                // calculate h term
                h_A = A_remove[i * num_bin + bin_i];
                h_E = E_remove[i * num_bin + bin_i];

                // get <d|h> term
                d_h_remove_temp += gcmplx::conj(d_A) * h_A / A_noise;
                d_h_remove_temp += gcmplx::conj(d_E) * h_E / E_noise;

                // <h|h>
                remove_remove_temp += gcmplx::conj(h_A) * h_A / A_noise;
                remove_remove_temp += gcmplx::conj(h_E) * h_E / E_noise;
                
            }

            for (int i = 0;
                    i < M;
                    i += 1)
            {
                
                int j = start_ind_add + i;

                
                A_noise = A_psd[noise_index_bin_i * data_length + j];
                E_noise = E_psd[noise_index_bin_i * data_length + j];

                d_A = A_data[data_index_bin_i * data_length + j];
                d_E = E_data[data_index_bin_i * data_length + j];

                //if ((bin_i == num_bin - 1))printf("CHECK add: %d %e %e  \n", i, A_noise, d_A.real());
                // calculate h term
                h_A = A_add[i * num_bin + bin_i];
                h_E = E_add[i * num_bin + bin_i];

                // get <d|h> term
                d_h_add_temp += gcmplx::conj(d_A) * h_A / A_noise;
                d_h_add_temp += gcmplx::conj(d_E) * h_E / E_noise;

                // <h|h>
                add_add_temp += gcmplx::conj(h_A) * h_A / A_noise;
                add_add_temp += gcmplx::conj(h_E) * h_E / E_noise;
                
            }
        }
        
        // read out
        d_h_remove[bin_i] =  4. * df * d_h_remove_temp;
        d_h_add[bin_i] =  4. * df * d_h_add_temp;
        add_add[bin_i] =  4. * df * add_add_temp;
        add_remove[bin_i] =  4. * df * add_remove_temp;
        remove_remove[bin_i] =  4. * df * remove_remove_temp;
        
    }
}


// wrapper for log likelihood
void swap_ll_diff_wrap(cmplx* d_h_remove, cmplx* d_h_add, cmplx* add_remove, cmplx* remove_remove, cmplx* add_add, cmplx* A_remove, cmplx* E_remove, int* start_ind_all_remove, cmplx* A_add, cmplx* E_add, int* start_ind_all_add, cmplx* A_data, cmplx* E_data, double* A_psd, double* E_psd, double df, int M, int num_bin, int* data_index, int* noise_index, int data_length)
{
    // GPU / CPU difference
    #ifdef __CUDACC__

    int num_blocks = std::ceil((num_bin + NUM_THREADS -1)/NUM_THREADS);

    swap_ll_diff<<<num_blocks, NUM_THREADS>>>(
       d_h_remove, d_h_add, add_remove, remove_remove, add_add, A_remove, E_remove, start_ind_all_remove, A_add, E_add, start_ind_all_add, A_data, E_data, A_psd, E_psd, df, M, num_bin, data_index, noise_index, data_length
    );
    cudaDeviceSynchronize();
    gpuErrchk(cudaGetLastError());

    #else

    swap_ll_diff(
       d_h_remove, d_h_add, add_remove, remove_remove, add_add, A_remove, E_remove, start_ind_all_remove, A_add, E_add, start_ind_all_add, A_data, E_data, A_psd, E_psd, df, M, num_bin, data_index, noise_index, data_length
    );

    #endif
}
