# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working
with code in this repository.

**LISA Analysis Tools–wide conventions:** [`../LISAanalysistools/docs/conventions.md`](../LISAanalysistools/docs/conventions.md) (canonical). **This repo's map:** [`docs/codebase-map.md`](docs/codebase-map.md).

## LISA Analysis Tools reorg state (post-Phase-3F, 2026-06-02)

GBGPU is the **GB-physics owner** in the LISA Analysis Tools layered architecture.
LISAanalysistools (LAT) owns generic LISA infrastructure; GBGPU owns
GB-specific physics on top of it.

**Build infrastructure (post-2026-06-03):**
- GBGPU's CMakeLists consumes LAT's `Detector.cu`, `Detector.hpp`, `global.hpp`
  via the same `python -c "import lisatools"` shell-out pattern used by
  lisa-on-gpu (`LISATOOLS_CUTILS` variable, then `file(COPY ${LISATOOLS_CUTILS}/...)`).
  This replaced an older `file(DOWNLOAD https://raw.githubusercontent.com/.../refs/heads/main/...)`
  pattern that pulled stale snapshots ignoring the version-pinned
  lisaanalysistools build dep. The copied files live under
  `src/gbgpu/cutils/` and are gitignored — source of truth is LAT.
- Same flow for `${GBT_CUTILS}` (gpubackendtools cutils dir, already
  in place pre-2026-06-03).
- `src/gbgpu/cutils/binding_gbgpu.{hpp,cxx}` -- pybind11 module skeleton
  (Phase GBGPU.pybind, 2026-06-03). Mirror of BBHx's
  binding_bbhx.{hpp,cxx}; built fresh (no prior pybind branch on GBGPU
  to pull forward). Consumes LAT's canonical `ReturnPointerBase` via
  `#include "binding_flr.hpp"` and asserts
  `static_assert(!LISATOOLS_IS_WRAPPER_OWNER, ...)` to enforce the
  single-registrant rule. Produces a new backend module
  `gbgpu_backend_{cpu,cudaXXx}.cgbgpu` alongside the existing Cython
  modules (utils, sharedmem). The skeleton ships an empty
  `GBGPUComputationWrap`; Phase 3L.7 populates it with
  GBTDIonTheFly + GBComputationGroup + gb_wdm_het_* kernels.
- `find_package(pybind11 CONFIG)` added to project-root `CMakeLists.txt`;
  `pybind11` added to `pyproject.toml`'s build-system requires.
- **`GBGPU_WITH_SHAREDMEM` (default `ON`, 2026-07-28)** — compiles the legacy
  `SharedMemoryGBGPU.cu` kernel family into `cgbgpu`. It is 4k lines of
  heavily-templated CUDA and the repo's **only** `#include <cufftdx.hpp>`, so
  `-DGBGPU_WITH_SHAREDMEM=OFF` also removes the configure-time
  `nvidia-mathdx` tarball download + extract + `find_package(mathdx)` — a
  GPU configure with it OFF needs **no network access** and compiles
  substantially faster.

  ```sh
  pip install -e . --config-settings=cmake.define.GBGPU_WITH_SHAREDMEM=OFF
  ```

  The seven `SharedMemory*_wrap` methods stay **bound** (identical Python
  surface) but raise `RuntimeError` with a rebuild instruction — the guard is
  `GBGPU_NO_SHAREDMEM` / `GBGPU_SHAREDMEM_UNAVAILABLE` in `binding_gbgpu.hpp`.
  Unaffected: `GBTDIonTheFly`, the chunked-heterodyne kernels, and the
  signal-heterodyne (`gb_signal_het_*`) family — i.e. an OFF build fully
  supports the sig-het in-model GB path and its parity gates.

  **Still needs ON:** GB F-stat RJ births (`GBGPU.get_fstat_ll`, via
  LAT `gbspecialstretch.para_log_like`), synthetic GB injections and the
  FD-domain global-template subtraction (`GBGPU.generate_global_template`,
  via LAT `recipe.py` / `stock/erebor/injections.py` / `sources/gb/waveform.py`),
  `GBAETWaveform` (`run_wave`), and `gathergalaxy`'s
  `swap_likelihood_difference`. A WDM-domain **search** run therefore needs
  ON; WDM PE runs and the sig-het/chunked-het parity gates do not.

**Already received from lisa-on-gpu (Phase 3F):**
- `gbgpu.jax.sources.ucb` — `JaxUCBSource`.
- `gbgpu.jax.wdm.kernels` — `gb_wdm_get_ll_jax`, `gb_wdm_fill_global_jax`, `gb_wdm_swap_ll_jax`.
- `gbgpu.jax.wdm.heterodyne_kernels` — `gb_wdm_het_get_ll_jax`, `gb_wdm_het_fill_global_jax`, `gb_wdm_het_swap_ll_jax`.
- `gbgpu.jax.wdm.fast_inner_heterodyne` — `fast_wdm_inner_heterodyne_jax`, `gb_chunk_fd_to_wdm_jax`, `ALPHA_AUTO`.

These import generic infrastructure absolutely from `lisatools.jax.{response,wdm,orbits}` — there is NO `gbgpu.jax.base` etc. Use the LAT path.

**Pending arrival (future C++ TDIonTheFly carve-out session):**
- C++: `GBTDIonTheFly` class + `GBComputationGroup` class + `WDMSplineHelpers.hh` + GB chunked-het CUDA kernels. Will land in `GBGPU/src/gbgpu/cutils/` with its own pybind11 module.
- Python: `gbcomps.py` (`GBWDMComputations`, `GBFDComputations`).
- JAX: `computation_group.py`'s `GBComputationGroupWrapJAX` (currently mixed with SOBBH in lisa-on-gpu's `fastlisaresponse.jax.wdm.computation_group`; split during the C++ carve-out).

**Pending arrival (V2 signal-heterodyne port, independent work item):**
- C++: `cutils/GBSignalHet.{hh,cu}` (source-class entry) +
  `cutils/GBAbsoluteFD.{hh,cu}` (`compute_fd_bin(params9, ch, k_global) → cmplx`,
  bin-by-bin, no TD intermediate, no full-N rfft) +
  `cutils/binding_gbsignalhet.cxx`.
- Python: `gbgpu/gbsignalhetcomputations.py` — `GBSignalHetComputations`,
  parallels `GBWDMComputations` (chunked-het) but uses the polyphase
  signal-het kernel suite (`gb_signal_het_{fill_global,get_ll,swap_ll,
  get_ll_grad,hessian,get_fstat_ll}`).
- JAX: `gbgpu/jax/wdm/gb_signal_het_kernels.py` + `gb_signal_het_computation_group.py`.
- The generic polyphase + bin-fold + reconstruct primitives live in LAT
  (`lisatools/cutils/SignalHet*.hh,cu`) — GBGPU only owns the
  GB-specific FD-bin producer.
- Full plan: `~/.claude/plans/yes-find-and-read-sprightly-garden.md`.
- Python prototype + walkthrough live at
  `LISAanalysistools/scripts/gb_chunked_het/gb_signal_het_wdm_v2*.py`
  (mm5 ≈ 1.6e-9 median, ~130× faster than v1 dense path).

**Single-registrant rule (LISA Analysis Tools–wide)**: GBGPU's binding TUs MUST NOT
register `OrbitsWrap`, `LISAResponseWrap`, `TDIConfigWrap`, or
`CubicSplineWrap`. The first three are owned by LAT's `pycppdetector`;
`CubicSplineWrap` is owned by GBT's `interp` module (2026-06-10). When GBGPU receives its tdionthefly
module, add `#include "lisatools_header_abi.hpp"` +
`static_assert(!LISATOOLS_IS_WRAPPER_OWNER, ...)` to its binding source
(see `lisa-on-gpu/src/fastlisaresponse/cutils/binding_tof.cxx` for the
pattern). The umbrella workspace's `tools/check_single_registrant.sh` is the CI
grep complement.

## Backend implementation hierarchy (LISA Analysis Tools–wide rule)

When implementing or modifying an algorithm that exists across multiple
backends (GPU C++ / CPU C++ / JAX), follow this hierarchy:

1. **GPU C++ (CUDA) leads.** This is the canonical performance target
   and reference implementation. New algorithms and optimizations are
   designed for the GPU first; CPU and JAX paths follow.

2. **CPU C++ mirrors GPU C++ as closely as possible.** Same kernel
   structure, same algorithm, same data flow — use `#ifdef __CUDACC__`
   or shared compile-time macros (`CUDA_SHARED`, `THREAD_START`,
   `BLOCK_INCR`, …) to bridge platform differences. The CPU path
   exists primarily for testing and CPU-only environments; it must
   not diverge in algorithm or output beyond floating-point order of
   operations.

3. **CPU C++ must reproduce the overall lisatools computation.**
   Against the lisatools reference (e.g. `FDSignal.transform`,
   `TDSignal.transform`, `XYZ2SensitivityMatrix`), match to machine
   precision (≤ 1e-15 mismatch) in direct modes; cache/approximation
   modes have documented per-feature error budgets.

4. **JAX may diverge internally** — design it to be JAX-efficient.
   JAX-CPU and JAX-GPU compilation targets may even differ. Use
   JAX-native idioms (`jax.lax.scan`, `jax.vmap`, static-shape
   `dynamic_slice` + masks, functional carries) rather than
   mechanically translating CUDA shared memory / register caches.

5. **JAX must match C++ inner-product outputs.** End-to-end
   likelihood quantities (`<d|h>`, `<h|h>`, swap_ll 5 terms) must
   match the C++ to floating-point precision (reldiff ≲ 1e-12) on
   representative test cases. Intermediate quantities (raw templates,
   per-chunk WDM coefficients) may differ at FP precision due to
   summation order — validate at the inner-product level.

**Workflow for a new feature.** GPU C++ → CPU C++ via `#ifdef` → JAX
with JAX-native idioms → cross-backend inner-product validation.
