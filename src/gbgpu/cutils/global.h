/*  This code was edited by Michael Katz. The information in this file is originally from the LAL library.
 *  The original copyright and license is shown below. Michael Katz has edited and added to
 *  the code for his purposes and removed dependencies on the LAL libraries.
 *  This code is distrbuted under the same GNU license it originally came with.

 *  Copyright (C) 2017 Sebastian Khan, Francesco Pannarale, Lionel London
 *
 *  This program is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation; either version 2 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with with program; see the file COPYING. If not, write to the
 *  Free Software Foundation, Inc., 59 Temple Place, Suite 330, Boston,
 *  MA  02111-1307  USA
 */


#ifndef _GLOBAL_HEADER_
#define _GLOBAL_HEADER_

#include <stdlib.h>
#include <stdio.h>
#include <complex>

// gbt_global.h provides cuda_complex.hpp + the cmplx typedef +
// CUDA_DEVICE / CUDA_KERNEL / CUDA_SHARED / gpuErrchk macro set.
// One sprint-wide copy lives in GPUBackendTools.
#include "gbt_global.h"

// gbt_global.h spells the threadsync macro CUDA_SYNC_THREADS. This file
// historically exposed CUDA_SYNCTHREADS (no underscore); keep the old
// spelling as a backwards-compat alias for legacy GBGPU code.
#ifndef CUDA_SYNCTHREADS
#define CUDA_SYNCTHREADS CUDA_SYNC_THREADS
#endif

// gbt_global.h does not provide CUDA_CALLABLE_MEMBER (a host+device
// decorator distinct from the device-only CUDA_DEVICE). Preserve it
// here for the legacy GBGPU kernels.
#ifdef __CUDACC__
#include "cuda_runtime_api.h"
#define CUDA_CALLABLE_MEMBER __host__ __device__
#else
#define CUDA_CALLABLE_MEMBER
#endif

typedef gcmplx::complex<double> cmplx;

#define invsqrt2 0.7071067811865475
#define invsqrt3 0.5773502691896258
#define invsqrt6 0.4082482904638631


#endif
