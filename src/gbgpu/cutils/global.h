#ifndef _GLOBAL_HEADER_
#define _GLOBAL_HEADER_

// gbt_global.h provides cuda_complex.hpp + the cmplx typedef +
// CUDA_KERNEL / CUDA_DEVICE / CUDA_SHARED / CUDA_SYNC_THREADS /
// CUDA_CALLABLE_MEMBER (`__host__ __device__`) macros + the gpuErrchk
// helper. One sprint-wide copy lives in GPUBackendTools; cbbhx pulls
// it in via ${GBT_CUTILS} on the cutils target's include path.
// Phase 3.dedup-followup (2026-06-05) collapsed every other macro and
// header pull that used to live in this file into a single
// gbt_global.h include.
#include "gbt_global.h"

// CUDA_SYNCTHREADS (no underscore) is the spelling used by GBGPU's
// SharedMemoryGBGPU.cu; gbt_global.h ships CUDA_SYNC_THREADS only.
// Keep this backwards-compat alias for legacy GBGPU code.
#ifndef CUDA_SYNCTHREADS
#define CUDA_SYNCTHREADS CUDA_SYNC_THREADS
#endif

#endif
