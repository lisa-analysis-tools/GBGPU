_Last mapped: dc561ce · 2026-07-10 · regenerate when structure changes_

# GBGPU codebase map

## 1. What this is

`GBGPU` (package `gbgpu`) is the LISA Analysis Tools **GB-physics owner**: it computes
Galactic-binary (GB / UCB) gravitational waveforms and their LISA TDI
likelihoods on CPU and CUDA GPUs, plus a JAX mirror. It hosts two largely
independent waveform pipelines — a classic dense "FastGB" fast/slow
decomposition (the original `GBGPU` class), and a newer TDI-on-the-fly +
WDM/FD chunked-heterodyne + polyphase signal-heterodyne stack built on
LISAanalysistools' (LAT) generic response/TDI infrastructure — and supplies
the GB-specific likelihood engines consumed by LAT's global-fit sampler.

## 2. Layout

| Path | Contents |
|---|---|
| `src/gbgpu/gbgpu.py` (3187 lines) | `GBGPUBase`/`GBGPU` — classic dense FastGB waveform + likelihood (fast/slow decomposition, XYZ/AET), driven by SharedMemory CUDA/CPU kernels. |
| `src/gbgpu/thirdbody.py` | `GBGPUThirdBody(GBGPUBase)` — eccentric third-body extension of the classic path. |
| `src/gbgpu/gbcomps.py` | `GBWDMComputations` (chunked-het, thin subclass of LAT's `WDMComputationsBase`) + `GBFDComputations` (FD chunked-het analog, config-only ctor). New-stack Python frontends. |
| `src/gbgpu/gbsignalhetcomputations.py` | `GBSignalHetComputations` — V2 polyphase signal-heterodyne likelihood; CPU-only; has an `for_band_engine()` mode that wraps a `GBWDMComputations` delegate for in-model use. |
| `src/gbgpu/gb_likelihood.py` | Domain-agnostic **band likelihood engines** (`FDBandLikelihoodEngine`, `WDMBandLikelihoodEngine`, `make_band_likelihood_engine`) consumed by LAT's global-fit GB special move. Moved here from `lisatools.globalfit.moves.gb_likelihood` (2026-07); that path is now a deprecation shim. |
| `src/gbgpu/wdm_het.py` | Deprecation shim; real content lives in `lisatools.wdm_het` (moved out so BBHx/SOBBH can share it without depending on GBGPU). |
| `src/gbgpu/parallelbase.py` **and** `src/gbgpu/utils/parallelbase.py` | Two near-duplicate copies of `GBGPUParallelModule` (backend-prefix resolver, prefixes bare backend tags with `gbgpu_`). Both are imported by different modules — see §7. |
| `src/gbgpu/utils/` | `utility.py` (physics helpers: `get_N`, `get_fdot`, `AET`, chirp-mass conversions), `config.py`, `citation.py`/`citations.py`, `exceptions.py`. |
| `src/gbgpu/cutils/` | C++/CUDA source + nanobind bindings (see §5). Copies of LAT's `Detector.{hpp,cu}`/`global.hpp` land here at CMake-configure time (gitignored; LAT is the source of truth). |
| `src/gbgpu/jax/` | JAX mirror: `sources/ucb.py` (`JaxUCBSource`), `wdm/{kernels,heterodyne_kernels,signal_het_kernels,fast_inner_heterodyne,computation_group}.py`, `tdi_on_the_fly.py`, `wrappers.py`. Gated on `import jax`. |
| `tests/` | `test_gbgpu.py` (classic `GBGPU` waveform smoke test), `test_thirdbody.py`. No dedicated tests here for chunked-het/sig-het/C++ TDIonTheFly — those are validated by scripts in `LISAanalysistools/scripts/gb_chunked_het/`. |
| `examples/` | Two notebooks: `GBGPU_tutorial.ipynb`, `fast_likelihood_tutorial.ipynb` — both cover the **classic** dense path, not the WDM/sig-het stack. |
| Cython | **Fully retired.** No `.pyx`/`.pxd` files remain; the old `gbgpu_utils_wrap.pyx` / `sharedmemgbgpu.pyx` CMake rules are commented out in `cutils/CMakeLists.txt`. `pyproject.toml` still lists `Cython` in `build-system.requires` and a `Programming Language :: Cython` classifier — stale (see §7). |

## 3. Core abstractions

Two parallel waveform/likelihood stacks coexist:

```
CLASSIC (dense FastGB)                    NEW STACK (TDI-on-the-fly + WDM/FD/sig-het)
────────────────────────                  ──────────────────────────────────────────
GBGPUBase (abc)                           C++: LISATDIonTheFly (LAT)
 └─ GBGPU                                        └─ GBTDIonTheFly (GBGPU, cutils/gb_tdi_on_the_fly.hh)
     └─ GBGPUThirdBody                                 UCB amp/phase/f/fdot closed forms
run_wave / get_ll / fill_global_template /        GBComputationGroup — gb_fd_*, gb_wdm_het_*, gb_signal_het_* kernels
swap_likelihood_difference / information_matrix
  → self.backend.sharedmem.SharedMemory*_wrap    Python frontends (cutils/__init__.py backend composition):
    (GBGPUComputationWrap methods)                 GBWDMComputations   (chunked-het, WDM domain)
                                                     GBFDComputations   (chunked-het, FD domain)
                                                     GBSignalHetComputations (V2 polyphase; can
                                                       wrap a GBWDMComputations via for_band_engine())

                          gb_likelihood.py: FDBandLikelihoodEngine / WDMBandLikelihoodEngine
                          wrap {GBFDComputations, GBWDMComputations} for LAT's global-fit
                          GB special move (make_band_likelihood_engine() dispatches on the
                          basis_settings type: FDSettings → FD engine, WDMSettings → WDM engine)
```

Naming collision to watch for: `lisatools.response.tdionfly.GBTDIonTheFly` (a
**Python** class, `TDIonTheFly` subclass, lives in LAT) is a *different class*
from GBGPU's C++ `GBTDIonTheFly` (nanobind, subclass of LAT's
`LISATDIonTheFly`). `gbsignalhetcomputations.py` imports the LAT Python one
for its `for_band_engine()` reference-signal construction while
`gb_tdi_on_the_fly.hh` defines the C++ one — grep carefully.

JAX mirrors the new stack's method surface via duck-typed wrapper classes
(`GBComputationGroupWrapJAX` in `jax/wdm/computation_group.py`) so
orchestration code doesn't need per-backend branches.

## 4. Public API / entry points

- **Classic dense waveform + likelihood** (matches the README/tutorials):
  ```python
  from gbgpu.gbgpu import GBGPU
  from lisatools.detector import EqualArmlengthOrbits
  gb = GBGPU(orbits=EqualArmlengthOrbits(...), force_backend="cpu")
  gb.run_wave(amp, f0, fdot, fddot, phi0, iota, psi, lam, beta, N=N, T=T, dt=dt)
  gb.get_ll(params, data_minus_template, psd, ...)
  gb.fill_global_template(...) / gb.swap_likelihood_difference(...) / gb.information_matrix(...)
  ```
- **New-stack chunked-het / FD likelihoods** (used by global-fit GB special move):
  ```python
  from gbgpu.gbcomps import GBWDMComputations, GBFDComputations
  comp = GBWDMComputations(wdm_settings, ..., force_backend="cpu")  # or "cuda12x", "jax"
  comp.fill_global_wdm(...) / comp.get_ll_wdm(...) / comp.get_swap_ll_wdm(...)
  ```
- **V2 polyphase signal-heterodyne** (CPU-only):
  ```python
  from gbgpu.gbsignalhetcomputations import GBSignalHetComputations
  sig = GBSignalHetComputations(data_td, ref_params, Nf=.., Nt=.., ..., force_backend="cpu")
  sig.get_ll(params, data_index=...)
  ```
- **Global-fit band-engine factory** (the layer LAT's sampler actually calls):
  ```python
  from gbgpu.gb_likelihood import make_band_likelihood_engine
  engine = make_band_likelihood_engine(basis_settings, gb_fd_comp=..., gb_wdm_comp=..., ...)
  ```
- **Physics/param helpers**: `gbgpu.utils.utility.{get_N, get_fdot, get_chirp_mass, AET, ...}`.
- **Backend introspection**: `gbgpu.get_backend("cpu"|"cuda12x"|...)`, `gbgpu.has_backend(...)`.

Note: only the classic path's per-backend low-level kernels are reachable via
`self.backend.sharedmem` / `self.backend.get_ll` / `self.backend.fill_global`
(pass-throughs onto a `GBGPUComputationWrap` instance); the new stack reaches
its native symbols via `self.backend.GBTDIonTheFlyWrap` /
`self.backend.GBComputationGroupWrap`.

## 5. Backend structure

- **CPU / CUDA C++**: single nanobind module per backend, `cgbgpu`, shipped as
  `gbgpu_backend_{cpu,cuda11x,cuda12x,cuda13x}.cgbgpu`. Despite comments
  throughout `binding_gbgpu.{hpp,cxx}` still saying "pybind11 module surface"
  (leftover from before the LISA Analysis Tools–wide Phase 3M pybind11→nanobind
  migration), the actual bindings use `#include <nanobind/nanobind.h>` and
  `NB_MODULE(cgbgpu, m)` — this is nanobind, not pybind11. Treat the
  "pybind11" wording in that file's comments as stale.
- One `.so` per backend contains: the GB-specific wrapper class
  (`GBGPUComputationWrap{CPU,GPU}` — hosts the legacy `gbgpu_utils` +
  `SharedMemoryGBGPU` methods), `GBTDIonTheFlyWrap{CPU,GPU}` +
  `GBComputationGroupWrap{CPU,GPU}` (new stack), **and** copy-compiled LAT
  sources (`Detector.cu`, `lat_tdi_on_the_fly.cu`) plus GBT's `Interpolate.cu`
  (for the LAPACKE cubic-spline solver used in orbit/signal splines) so the
  virtual-class typeinfo and LAPACKE-dependent symbols live in GBGPU's own
  `.so` — see `cutils/CMakeLists.txt`.
- CPU/GPU class-name aliasing follows the LISA Analysis Tools–wide rule: `GBTDIonTheFly` →
  `GBTDIonTheFlyGPU`/`CPU`, `GBComputationGroup` →
  `GBComputationGroupGPU`/`CPU`, and their `*Wrap` counterparts, toggled on
  `__CUDA_COMPILATION__`/`__CUDACC__` in both `gb_tdi_on_the_fly.hh` and
  `binding_gbgpu.hpp`.
- **Backend selection**: `gbgpu/__init__.py` registers `GBGPUCpuBackend`,
  `GBGPUCuda{11,12,13}xBackend` (subclasses composing LAT's
  `LISATToolsBackend`-derived methods with GB-specific ones —
  `cutils/__init__.py`) under names `gbgpu_cpu`/`gbgpu_cuda11x`/etc in
  `gpubackendtools`'s `Globals().backends_manager`. `gbgpu.get_backend("cpu")`
  auto-prefixes to `gbgpu_cpu`.
- **JAX**: kernel functions exist (`gbgpu.jax.wdm.*`,
  `gbgpu.jax.tdi_on_the_fly.run_wave_tdi`) and `GBGPUParallelModule` /
  `WDMComputationsBase` check `self.backend.name == "gbgpu_jax"`, but **no
  `GBGPUJaxBackend` class is registered anywhere in `gbgpu/__init__.py` or
  `gbgpu/cutils/__init__.py`** — grepping the whole LISA Analysis Tools repo set for
  `gbgpu_jax` turns up only a docstring reference and a dev script. If
  `force_backend="jax"` doesn't resolve for a GB chunked-het/FD comp, this is
  the first place to look (either a missing `GBGPUJaxBackend` registration,
  or JAX dispatch happens through a different mechanism not found in this
  repo — verify before assuming it's broken).
- `GBSignalHetComputations.supported_backends()` returns `["cpu"]` only —
  the sig-het CUDA kernels are a documented TODO.

## 6. Cross-repo dependencies

**Imports (GBGPU → others):**
- `gpubackendtools` — `ParallelModuleBase`, `Globals`, `Backend`,
  `Cpu/Cuda{11,12,13}xBackend`, exceptions; GBT's `Interpolate.cu`/LAPACKE
  cubic-spline solver and `cuda_complex.hpp`/`gbt_global.h` (via
  `${GBT_CUTILS}` at CMake configure time — GBGPU's own copy of
  `cuda_complex.hpp` was deleted in the Phase 3.dedup pass).
- `lisatools` (LAT) — `Orbits`/`EqualArmlengthOrbits`/`L1Orbits`
  (`lisatools.detector`), `TDIConfig` (`lisatools.response.tdiconfig`),
  `WDMSettings`/`FDSettings`/`TDSettings`/`TDSignal`
  (`lisatools.domains`), `WDMComputationsBase` (`lisatools.chunked_het`),
  `sparse_time_grid`/`bin_fold_real` (`lisatools.signal_het`),
  `XYZ2SensitivityMatrix` (`lisatools.sensitivity`),
  `AnalysisContainer`/`AnalysisContainerArray`
  (`lisatools.analysiscontainer`), the JAX response/orbits/WDM mirrors
  (`lisatools.jax.*`), and at the C++ level: LAT's `Detector.{hpp,cu}`,
  `global.hpp`, `LISAResponse.hh`, `Interpolate.hh`, `fd_domain.hh`,
  `wdm_settings.hh`, `wdm_domain.hh`, `lat_tdi_on_the_fly.hh/.cu`,
  `lat_chunked_het_kernels.hh`, `binding_flr.hpp`,
  `lisatools_header_abi.hpp` — all resolved via `import lisatools` shell-out
  in `cutils/CMakeLists.txt` at configure time, copied into
  `src/gbgpu/cutils/` (gitignored) and compiled into GBGPU's own `.so`.

**Depends on GBGPU (others → GBGPU):**
- `LISAanalysistools` — `lisatools.globalfit.moves.{gb_likelihood,
  gbspecialstretch, gbbands}`, `lisatools.globalfit.stock.erebor.gb`, and
  ~20 dev/validation scripts under `LISAanalysistools/scripts/gb*/` import
  `gbgpu.{gbgpu, gbcomps, gbsignalhetcomputations, gb_likelihood,
  utils.utility}` directly. `lisatools.globalfit.moves.gb_likelihood` is now
  a thin deprecation shim re-exporting `gbgpu.gb_likelihood`.
- `BBHx` — `bbhx/utils/citations.py` reads `gbgpu.__file__`/`_is_editable`
  purely to locate citation metadata (not a functional dependency).
- No code across the LISA Analysis Tools repos was found importing `gbgpu.jax.wdm.computation_group`'s
  `SOBBHComputationGroupWrapJAX` from outside GBGPU (see §7).

## 7. Non-obvious invariants / gotchas

- **`SOBBHComputationGroupWrapJAX` still lives in GBGPU**
  (`src/gbgpu/jax/wdm/computation_group.py:624`), subclassing
  `GBComputationGroupWrapJAX`. This violates the LISA Analysis Tools–wide
  "GB→GBGPU, SOBBH→BBHx" split that the C++ carve-out (Phase 3L.8) already
  completed for the native code. Nothing outside this file imports it —
  looks like leftover from before the JAX split finished. If touching GB/SOBBH
  JAX chunked-het, check whether this should have moved to BBHx already.
- **Duplicate `GBGPUParallelModule`**: `gbgpu/parallelbase.py` and
  `gbgpu/utils/parallelbase.py` define near-identical classes. `gbgpu.py`
  imports the `utils.` one; `gbcomps.py`/`gbsignalhetcomputations.py` import
  the top-level one (aliased as `FastLISAResponseParallelModule`). Keep them
  in sync if editing backend-prefix logic, or better, consolidate.
- **`GBGPU/CLAUDE.md` is significantly stale.** It's dated "post-Phase-3F,
  2026-06-02" and describes `GBTDIonTheFly`/`GBComputationGroup`,
  `gbcomps.py`, and the V2 signal-heterodyne C++/Python/JAX trio as
  "pending arrival" / "future session" work. All of it has since landed:
  `cutils/gb_tdi_on_the_fly.{hh,cu}` (676+3040 lines), `binding_gbgpu.cxx`
  (1043 lines, not an "empty skeleton"), `gbcomps.py` (923 lines),
  `gbsignalhetcomputations.py` (449 lines) all exist and are wired into the
  global-fit sampler. Trust the code, not this file, for current state.
- **README.md is also stale** in tone: it describes GBGPU as "The waveform
  code is entirely Python-based... much simpler in Python for right now"
  with only "fast C-based methods to combine waveforms into global fitting
  templates." That describes only the classic `GBGPU` class; the new-stack
  TDI-on-the-fly/chunked-het/signal-het machinery is C++/CUDA-native with a
  thin Python frontend, the opposite emphasis.
- **`pyproject.toml` still lists `Cython`** in `build-system.requires` and
  carries a `Programming Language :: Cython` classifier, but there is no
  `.pyx`/`.pxd` file anywhere in the repo — Cython is fully retired (see §2).
- **`gbgpu_jax` backend is referenced but not registered** — see §5.
- LISA Analysis Tools–wide rules that apply here and are enforced/checked in this repo's
  binding code: single-registrant rule (`static_assert(!LISATOOLS_IS_WRAPPER_OWNER)`
  in `binding_gbgpu.cxx` — GBGPU must never register `OrbitsWrap`/
  `TDIConfigWrap`/`LISAResponseWrap`/WDM*/FD*/Spline* itself), host→device
  wrapper-struct upload pattern (see the umbrella workspace root `CLAUDE.md`), CPU/GPU
  class-name aliasing (§5), and deepcopy/pickle safety (`self.xp` must stay a
  property derived from `self.backend`, never a cached module reference —
  check `gbgpu.py`/`gbcomps.py`/`gbsignalhetcomputations.py` `xp` properties
  if refactoring).
- `flip_ref_phase` on `GBGPUBase.__init__` silently negates `params[:, 4]`
  (phi0) inside `get_ll`/other methods to match JAX-GB phase convention —
  easy to double-apply if callers also flip phi0 upstream.
- `GBSignalHetComputations.for_band_engine()` constructs itself via
  `cls.__new__(cls)` (bypassing `__init__`) and manually re-implements the
  constructor body against a delegate `GBWDMComputations`' `wdm_settings` —
  if `__init__`'s field set changes, `for_band_engine` must be updated by
  hand (no shared helper).

## 8. Where to look for X

| Change / understand X | Start in |
|---|---|
| Classic dense FastGB waveform generation | `src/gbgpu/gbgpu.py` (`GBGPUBase._construct_slow_part`, `run_wave`) |
| Classic likelihood / global-template fill / swap proposals | `src/gbgpu/gbgpu.py` (`get_ll`, `fill_global_template`, `swap_likelihood_difference`) |
| Eccentric third-body physics | `src/gbgpu/thirdbody.py` |
| GB UCB closed-form amp/phase/freq (new stack) | `src/gbgpu/cutils/gb_tdi_on_the_fly.hh` (`GBTDIonTheFly::ucb_*`) |
| Chunked-heterodyne WDM likelihood (fill/get_ll/swap/fstat) kernels | `src/gbgpu/cutils/gb_tdi_on_the_fly.hh` (`GBComputationGroup::gb_wdm_het_*_wrap`) + `.cu` bodies |
| FD chunked-het likelihood kernels | same file, `gb_fd_*_wrap` methods; Python side `src/gbgpu/gbcomps.py::GBFDComputations` |
| V2 polyphase signal-heterodyne | `src/gbgpu/gbsignalhetcomputations.py` + `gb_tdi_on_the_fly.hh` `gb_signal_het_*_wrap` |
| Global-fit GB special-move likelihood engine | `src/gbgpu/gb_likelihood.py` (`make_band_likelihood_engine`, `FDBandLikelihoodEngine`, `WDMBandLikelihoodEngine`) |
| nanobind module registration / new native method exposure | `src/gbgpu/cutils/binding_gbgpu.{hpp,cxx}` |
| CMake build wiring (LAT/GBT source copy-in, LAPACKE, mathdx/cuFFTDx) | `src/gbgpu/cutils/CMakeLists.txt` |
| Backend composition (which native symbols a backend carries) | `src/gbgpu/cutils/__init__.py` |
| Backend registration / `gbgpu.get_backend` | `src/gbgpu/__init__.py` |
| JAX mirror of chunked-het kernels | `src/gbgpu/jax/wdm/{kernels,heterodyne_kernels}.py` |
| JAX mirror of signal-het kernels | `src/gbgpu/jax/wdm/signal_het_kernels.py` |
| JAX duck-typed native-class wrappers | `src/gbgpu/jax/wdm/computation_group.py`, `src/gbgpu/jax/wrappers.py` |
| Param/physics helper functions (get_N, get_fdot, chirp mass, AET) | `src/gbgpu/utils/utility.py` |
| Backend-prefix / `force_backend` resolution | `src/gbgpu/parallelbase.py` **and** `src/gbgpu/utils/parallelbase.py` (duplicated, see §7) |
| Citations / bibliography metadata | `src/gbgpu/utils/citation.py`, `citations.py` |
