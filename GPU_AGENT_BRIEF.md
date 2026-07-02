# GPU Agent Brief — STFT GB likelihood: FFT-per-column vs Fresnel benchmark + development

**Generated:** 2026-07-02 (from the CPU-side development sessions on macOS)
**Status:** 🟢 Ready — all code CPU-validated and pushed; your machine provides the missing ingredient (a CUDA GPU)
**Branches:** `feat-stft-gb` on BOTH repos —
`lisa-analysis-tools` @ `6d010cf` · `GBGPU` @ `15c2921` (+ a docs commit adding this brief and the design doc)

You are picking up a finished, validated CPU prototype whose **decisive question can only be answered
on your machine**: does the FFT-per-column STFT template generator beat the analytic Fresnel generator
on **GPU wall-clock** at production scale (O(10⁴) galactic binaries)? Everything below is what you need
to build, verify, benchmark, decide, and continue development — without access to the original sessions.

---

## 1. Overview — what this work is

LISA galactic-binary (GB) likelihoods in the **STFT time–frequency domain**: data = per-segment windowed
DFTs of the XYZ TDI streams; lnL uses `⟨a|b⟩ = 4·df·Σ_{t,f} conj(aᵢ)·invC_ij(f)·b_j` (full 3×3 XYZ inverse
covariance). Two template kernels share ONE driver (`stft_eval_block_ll` in LAT
`src/lisatools/cutils/lat_stft_kernels.hh`), differing only in a compile-time **column-producer policy**:

- **`FresnelColumn`** (merged earlier, production feature set): per-segment constant-envelope ×
  quadratic-phase chirp, each pixel a closed-form Fresnel C/S integral; Tukey window folded in exactly
  via a 7-term trig decomposition. ~22 sincos per windowed pixel × (2·n_side+1) pixels/column.
  Full surface: `get_ll` / `fill_global` / `swap` / `grad` / `fstat`.
- **`FFTColumn`** (the new track, `get_ll` only so far): sample the TRUE complex TDI envelope at `n_sub`
  midpoints per segment, window per-sample (exact for ANY window), heterodyne to the carrier bin, then a
  targeted twiddle-recurrence DFT to the 2·n_side+1 band bins. After two restructures — **R1**
  transcendental-free DFT (sincos/column: (2·n_side+1)·n_sub → n_sub+2, i.e. 672→34 at n_side=10,
  n_sub=32) and **R2** per-block orbit spline cache (expensive orbit evals/binary: n_sub·NT·~48 → ~15·N_cp,
  ~1000×) — it realizes the "fewer transcendentals + amortized response" thesis at the op-count level.

**Accuracy verdict (already settled, CPU):** FFTColumn ≥ Fresnel everywhere measured, numerically exact
as n_sub grows, immune to segment length and window/taper choice. Fresnel is faster on CPU (~4–10×) but
that is explicitly a **non-metric** (see §7 rule 1).

**What success looks like:** FFT wall-clock ≤ Fresnel at matched accuracy on production-ish configs
(num_bin ~ 16384, n_side=10, n_sub=32/64, n_cp_orbit=48) → "GO" → FFTColumn becomes the production
kernel and gains swap/grad/fstat + a JAX mirror. If occupancy kills it → "NO-GO" → document why, and
Fresnel (now correct for arbitrary tapers, see §6) remains the workhorse.

Authoritative design + full CPU results: `docs/specs/2026-07-01-stft-gb-fft-per-column-design.md`
(in the GBGPU repo with this brief; §8 decision grid, §12 CPU results, §12.1 Tukey/scaling, §12.2 restructures).

---

## 2. Repos, branches, topology — read before touching git

| repo | clone | branch | tip |
|---|---|---|---|
| LAT | `https://github.com/lisa-analysis-tools/lisa-analysis-tools.git` | `feat-stft-gb` | `6d010cf` |
| GBGPU | `https://github.com/lisa-analysis-tools/GBGPU.git` | `feat-stft-gb` | `15c2921` + docs commit |
| GPUBackendTools | `https://github.com/lisa-analysis-tools/GPUBackendTools.git` | `feat-quintic-spline` | `b07bd18` |

- ⚠️ **GBGPU's origin has NO `l2d-dev` branch** — `feat-stft-gb` carries the entire l2d-dev lineage
  (STFT/Fresnel GB port + FFT work). Do NOT rebase onto `master`/`dev`/`new_dev`; commit directly on
  `feat-stft-gb` and push there.
- LAT's `origin/l2d-dev` is 2 commits behind this branch's base (missing `8f02262` grid-snap fix +
  `fc767ec` STFT kernels — both contained here). Same rule: stay on `feat-stft-gb`.
- Commit ledger (oldest→newest, all on `feat-stft-gb`):
  - LAT: `8f02262`/`fc767ec` (base: WDM grid-snap + STFT/Fresnel kernels) → `133f645`,`3f2a0ba`
    (policy structs + templatized driver, Fresnel byte-identical) → `bf0d3fd` (FFTColumn + `stft_get_ll_fft`
    kernel) → `79024e0` (Tukey per-sample) → `e2dd70a` (R1 twiddle DFT, bit-identical) → `0085355`
    (R2 orbit cache, ==direct to 1.1e-9) → `6d010cf` (windowed-Fresnel off-grid-taper fix, §6).
  - GBGPU: `5fcf9d5` (base: STFT/Fresnel GB computations get_ll/fill_global/swap/grad/fstat) →
    `2ba0a62` (wrap/binding/Python `get_ll_stft_fft`) → `0df0e9f` (limits doc) → `e608ebd` (Tukey test)
    → `b0baa8c` (`n_cp_orbit` knob + cache-vs-direct test) → `15c2921` (the benchmark script).

---

## 3. Machine setup & build (Linux + CUDA)

Requirements: Python ≥3.12, `uv`, CUDA toolkit with `nvcc` in PATH, LAPACK/LAPACKE dev headers
(`liblapacke-dev` / `openblas-devel`; on the mac we needed `PKG_CONFIG_PATH` pointing at lapack+openblas
`.pc` dirs — on Linux system packages are usually found automatically; export it only if CMake configure
fails to find LAPACKE).

**Build ORDER matters and is load-bearing:**

1. `uv venv --python 3.12 .venv`
2. **GPUBackendTools first** (owns `CubicSplineWrap`; GBGPU's CUDA backends import
   `gbt_backend_<flavor>.interp` at load time and there are **no published CUDA plugin wheels** — it must
   be built from source on this box with nvcc visible so the `cuda12x`/`cuda11x` flavor gets built):
   `uv pip install -e ./GPUBackendTools`
   *Branch:* clone with `-b feat-quintic-spline` (on origin at `b07bd18`, "ready to try gpu install" —
   the exact tip the CPU box developed against). After installing, verify
   `python -c "import gbt_backend_cuda12x.interp"` (adjust flavor to your CUDA major).
3. **LAT second**: `uv pip install -e ./lisa-analysis-tools` (plain first-pass install is fine — it pulls
   the runtime deps into the venv). Then pin `uv pip install "matplotlib<3.11"`.
4. **GBGPU last, and NEVER with build isolation**:
   `uv pip install -e ./GBGPU --no-build-isolation --no-deps --no-cache`
   **Why:** GBGPU's CMake shells out to `python -c "import lisatools"` to locate LAT's cutils (headers
   like `lat_stft_kernels.hh` are compiled into GBGPU's translation units). With isolation on, the build
   env would pip-install `lisaanalysistools` from PyPI — the WRONG code (not this branch) — and silently
   build against stale kernels. `--no-build-isolation` makes it see the venv's editable feat-stft-gb LAT.
   Build deps must pre-exist in the venv for this: `grep -A8 '\[build-system\]' GBGPU/pyproject.toml`
   and `uv pip install` whatever it lists (scikit-build-core/pybind11/cmake/ninja etc.) before step 4.
5. **Verify what you built** (a prior session lost hours to a venv importing the wrong tree):
   ```
   python -c "import lisatools, gbgpu; print(lisatools.__file__); print(gbgpu.__file__)"
   ```
   Both must resolve into YOUR feat-stft-gb clones. Also confirm the GPU backend module exists
   (`gbgpu_backend_cuda12x` / `.cgbgpu`), not just `gbgpu_backend_cpu`.

**Rebuild rule:** any edit to a LAT header consumed by GBGPU (`lat_stft_kernels.hh`,
`lat_tdi_on_the_fly.*`, `lat_chunked_het_kernels.hh`, …) requires rebuilding **GBGPU** with
`--no-build-isolation --no-deps --no-cache` (the `--no-cache` busts uv's stale-wheel cache — without it
you can "rebuild" and test the OLD binary). Cold rebuilds took >5 min on the mac; run them in the
background, and when in doubt sabotage-probe (introduce a deliberate compile error, confirm the build
fails, revert) to prove you're compiling what you think.

---

## 4. Correctness gates — run BEFORE any timing

1. **The test suite** (CPU-validated at 30 passed + 2 subtests, ~72 s on the mac; nanobind "leaked"
   atexit warnings are benign — trust the exit code):
   ```
   cd GBGPU && python -m pytest tests/test_stft_gb.py tests/test_stft_gb_accuracy.py \
       tests/test_stft_gb_crossdomain.py tests/test_stft_gb_fft.py -q
   ```
2. **The benchmark's built-in gate on the GPU backend**: every `n_side` block injects a GB and checks
   recovery mismatch vs the TRUE STFT for both paths + their `d_h` agreement before timing — a broken
   GPU build fails loudly instead of producing fast garbage:
   ```
   python scripts/bench_stft_fft_vs_fresnel.py --smoke --backend gpu
   ```
3. **CPU↔GPU cross-check**: run the smoke on `--backend cpu` and `--backend gpu`; gate quantities
   (`d_h`, `h_h`, mismatches) must agree to inner-product precision (sprint rule: ≲1e-12 relative, FP
   summation order aside). If GPU disagrees with CPU, the GPU kernel is wrong — the CPU mirrors the GPU
   structure exactly (`#ifdef __CUDACC__` bridges), so bisect the diff, don't tune thresholds.

Expected magnitudes (from the CPU campaign, smoke config ~8 d): FFT recovery mm ~2.5e-3, Fresnel
~2.2e-2 at that short-Tobs config; full-year configs see §6 scoreboard.

---

## 5. The benchmark — protocol, knobs, go/no-go

Script: `GBGPU/scripts/bench_stft_fft_vs_fresnel.py` (self-contained; docstring has everything).
Times `STFTGBComputations.get_ll_stft_fft` (FFTColumn) vs `get_ll_stft` (Fresnel) on identical data,
sweeping the §8 decision grid: `num_bin × n_side_bins × n_sub × n_cp_orbit`, min-of-`--repeats` reported.

**CLI reference** (defaults in brackets): `--backend` [auto→gpu→cpu] · `--dt` [10 s] · `--stft-hours`
[6] · `--tobs-days` [91] · `--alpha` [0.1 Tukey] · `--num-bins` [1024,4096,16384] · `--n-side` [10] ·
`--n-sub` [24,32,64] · `--n-cp-orbit` [48,0 — 0 = direct `get_tdi`, no cache] · `--spread-bins` [32] ·
`--repeats` [5] · `--seed` [42] · `--csv PATH` · `--smoke`.

**Protocol:**
1. Smoke + cross-check (§4), then the full sweep:
   ```
   python scripts/bench_stft_fft_vs_fresnel.py --csv bench_stft_fft.csv
   ```
2. A production-leaning variant worth adding: `--tobs-days 365 --num-bins 16384,32768 --repeats 10`.
3. **Occupancy** (THE known risk — R2's shared-mem cache is ~23 KB at n_cp=48, and FFTColumn holds
   `slow[3·n_sub]` complex doubles in registers/local — 3 KB/thread at n_sub=64):
   ```
   ncu --kernel-name 'regex:stft_get_ll' --section Occupancy \
       python scripts/bench_stft_fft_vs_fresnel.py --num-bins 4096 --repeats 1
   ```
   Record achieved occupancy + the limiter (registers? shared? spills to local?) for BOTH kernels;
   one config with `--set full` is worth the time. `nsys` for the end-to-end timeline if launch overhead
   looks suspicious at small num_bin.

**Go/no-go:** GO = FFT wall-clock beats Fresnel at num_bin ~ 16384, n_side=10, n_sub=32 (64 as the
stress case), n_cp_orbit=48. Matched accuracy is automatic (FFT ≥ Fresnel accuracy at n_sub=32
everywhere measured). Read the CSV for scaling: both should go ~linear in num_bin once the GPU
saturates; FFT cost ~linear in n_sub (envelope sampling dominates post-R1); `n_cp_orbit=0` shows what
R2 is worth on GPU (expect a large slowdown without the cache).

**If occupancy-bound, knobs in order of cheapness:**
- `--n-sub 24` (rule §7.5 still satisfied at n_side=10) and/or `--n-cp-orbit 24` (~12 KB shared).
- `__launch_bounds__` / `maxrregcount` experiments on `stft_get_ll_fft_kernel` (LAT
  `lat_stft_kernels.hh`); check whether `slow[]` spills and whether chunking it helps.
- Compile-time guards `STFT_FFT_NSUB_MAX=64`, `STFT_ORBIT_NCP_MAX=48` (same header) bound the static
  footprints. NOTE: production 1-day segments have Doppler spread ±37 bins → n_side~40 needs
  n_sub ≥ 81 > NSUB_MAX — raising that guard (+ GBGPU rebuild) is a known future task; the benchmark's
  decision grid stays within today's guards.

---

## 6. Established results you can lean on (all CPU-validated, 1-yr injection unless noted)

Shared error floors (method-independent): band truncation from the analysis-window skirts —
`mm_floor ≈ 1−√(1−f_out)` — plus "observation must sit interior to the orbit product" (edge segments
cost ~1e-3-scale artifacts; keep t_start ≥ 3 segments inside the orbit span). Compare methods on
**in-band** error; whole-grid mismatch mostly measures the window + n_side.

| config (n_side=10) | truncation floor | Fresnel whole/in-band | FFT whole/in-band |
|---|---|---|---|
| Δ=6 h, Tukey α=0.1 | 1.250e-3 | 1.257e-3 / 7e-6 | 1.251e-3 (n_sub≥48) / 2e-5 |
| Δ=1 d, taper 10⁴ s (α≈0.2315) | 1.630e-5 | 9.57e-5 / 7.9e-5 | 1.637e-5 / 4.3e-7 (n_sub=32) |

- **α-sweep floors** (Δ=6 h): rect 1.06e-2 → α=0.1 1.25e-3 → α=0.5 1.9e-6 → Hann 1.1e-7. Fresnel's
  windowed expansion saturates ~5e-6 at α≥0.5; FFTColumn stays exact (its quadrature IMPROVES with
  smoother windows).
- **Fresnel model error grows with Δ** (constant-envelope assumption): in-band 7e-6 @6h → 8e-5 @24h;
  recovery degrades to ~5e-3 @4-day segments while FFT holds ~2e-3. FFTColumn is Δ-immune.
- **`n_sub ≥ 2·n_side+1` is a HARD rule** (below it far bins alias catastrophically); converged by
  ~1.5× that. Midpoint sampling (τ = t + (m+0.5)Δ/n_sub) is load-bearing — 2nd-order quadrature.
- **Fresnel off-grid-taper fix (LAT `6d010cf`, 2026-07-02):** the right-ramp half-cosine of the Tukey
  decomposition is referenced to the segment END; re-expressed as f∓f_taper shifts it carries constant
  phases e^{∓2πi·f_taper·Δ} = e^{∓2πi/α} — unity iff 1/α ∈ ℤ (all historical alphas), previously
  omitted. Off-grid tapers (e.g. 10⁴ s at Δ=1 d) silently degraded Fresnel to mm ~2e-2; fixed →
  9.57e-5 (values in the table are POST-fix). Fresnel now supports arbitrary taper lengths; FFTColumn
  was never affected. If you touch `domains.cu:get_windowed_fourier_value`, this is why `right_rot_p/m`
  exist.
- CPU wall-clock ratio FFT/Fresnel ~4–10× slower across configs — expected, structural, and a
  non-metric.

---

## 7. Criticalities & rules (hard-won — violating these cost real time)

1. **GPU C++ leads; CPU mirrors it; do NOT optimize for CPU wall-clock.** Sprint-wide backend
   hierarchy (see GBGPU/CLAUDE.md): GPU is canonical, CPU must not diverge algorithmically, JAX may
   diverge internally but must match inner products to ≲1e-12.
2. **One live `STFTComputationGroup` at a time.** Building a second corrupts the first's C++ buffers
   (a cached test fixture produced a spurious mismatch=1.0). Build + use sequentially; the benchmark
   already respects this.
3. **`force_backend=None` crashes `XYZSensitivityBackend`** (missing `supported_backends` — pre-existing
   LAT quirk). Always pass a concrete backend string; the bench script resolves "auto"→gpu→cpu itself.
   A proper upstream fix is a welcome small PR on this branch.
4. **Ground truth is the injected brute STFT, never Fresnel** — Fresnel has its own model error that
   FFTColumn converges PAST. Accuracy tests that compare FFT-to-Fresnel with tight thresholds are
   mis-designed (we made that mistake once).
5. **Template window must match the data window** (same Tukey α / taper; `taper_duration = α·Δ/2`,
   `f_taper = 1/(2·taper)`), carrier snap is floor `int((f−f_min)/df)`, and `n_sub ≥ 2·n_side+1`.
6. **Single-registrant rule** if you touch bindings: GBGPU must NOT register `OrbitsWrap`,
   `LISAResponseWrap`, `TDIConfigWrap` (LAT owns them) or `CubicSplineWrap` (GBT owns it).
7. **Verify the venv imports YOUR tree before trusting any test** (§3.5). And after LAT header edits,
   rebuild GBGPU `--no-cache` (§3 rebuild rule).
8. If you render matplotlib from a process that already imported the native modules, glyphs can corrupt
   (freetype clash — on mac at least): compute → dump npz/json → plot in a fresh subprocess.
9. The 1-yr agreement/α-sweep analysis scripts live on the origin machine (not in the repos); the
   in-repo tests + benchmark are self-sufficient for your mission. Ask the user if you need them.

---

## 8. Mission & roadmap

**Phase A — benchmark (the reason you exist):**
1. Build (§3) → gates (§4) → full sweep + occupancy (§5).
2. Commit `bench_stft_fft.csv` + a short `scripts/bench_results_<gpu-name>.md` (wall-clock table,
   occupancy numbers, limiter, verdict + rationale) to `feat-stft-gb`, push. Appending the verdict as
   §13 of the design doc is the preferred long-form home.

**Phase B — if GO (FFT wins or ties at production configs):**
1. Extend `FFTColumn` to the rest of the surface via the SAME `ColumnProducer` policy on the shared
   driver: `fill_global`, `swap`, `grad`, `fstat` (the Fresnel versions in `lat_stft_kernels.hh` /
   GBGPU `gbcomps.py` are the models — the deltas should be small since the driver is shared).
   GPU-first, CPU mirror via the existing `#ifdef` bridges, suite extended per feature.
2. Raise `STFT_FFT_NSUB_MAX` for wide-band (1-day-segment) configs, re-check occupancy at n_sub=81+.
3. JAX mirror (backend hierarchy rules 4–5: JAX-native internals — `lax.scan` over segments, vmap over
   binaries — validated at the inner-product level ≤1e-12 against C++).

**Phase C — if NO-GO:**
1. Record the limiter precisely (occupancy? memory-bound envelope sampling? launch config?). Try §5
   knobs + a thread-per-(column,chunk) split before concluding.
2. Fresnel remains production (now arbitrary-taper-correct); the FFT column stays as the accuracy
   oracle and the whole-span-taper escape hatch (its per-sample windowing accepts ANY window, including
   a global taper scaled per column — a possible future kernel extension: multiply columns by
   w_glob(t_mid)).

**Either way:** the long-Tobs orbit cache question — N_cp=48 over a full year is accurate for the
smooth orbits (verified), but confirm at your production T_obs; a chunked cache is the designed
fallback if density ever bites.

---

## 9. Key code map

- **LAT `src/lisatools/cutils/lat_stft_kernels.hh`** — the heart: `FresnelColumn` / `FFTColumn` policy
  structs; templatized `stft_eval_block_ll<SourceT, ColumnProducer>` driver (block-per-binary,
  thread-per-segment/column); `stft_get_ll_fft_{kernel,impl}` (builds the orbit cache before the binary
  loop); `stft_unit_cpow` twiddle helper; guards `STFT_FFT_NSUB_MAX=64`, `STFT_ORBIT_NCP_MAX=48`.
- **LAT `src/lisatools/cutils/domains.cu`** — STFT domain machinery: `get_freq_index` (floor snap),
  `add_ip_contrib` (3×3 XYZ contraction), `STFTFresnel::get_windowed_fourier_value` (7-term Tukey
  decomposition + the `right_rot_p/m` off-grid-taper phases, `6d010cf`).
- **LAT `src/lisatools/cutils/lat_chunked_het_kernels.hh`** — `populate_orbit_spline_cache` (~:358);
  `lat_tdi_on_the_fly.{hh,cu}` — `OrbitsSplineCache`, `get_tdi_Xf_single_cached` (~:372).
- **GBGPU `src/gbgpu/cutils/gb_tdi_on_the_fly.{hh,cu}`** — `gb_stft_get_ll_fft_wrap` (instantiates the
  impl for `GBTDIonTheFly`); `binding_gbgpu.{hpp,cxx}` — pybind `gb_stft_get_ll_fft`.
- **GBGPU `src/gbgpu/gbcomps.py`** — `STFTGBComputations.get_ll_stft_fft(params, n_sub=32,
  n_cp_orbit=48, …)` and the Fresnel `get_ll_stft` / `fill_global_stft` next to it; `GBFDComputations`
  for the FD cross-checks.
- **GBGPU `tests/test_stft_gb_fft.py`** — 7 FFT tests (smoke, vs-injection, Tukey recovery, n_sub
  convergence, cache-vs-direct, cost proxy); `test_stft_gb{,_accuracy,_crossdomain}.py` — the Fresnel
  port suite.
- **GBGPU `scripts/bench_stft_fft_vs_fresnel.py`** — your Phase A.

Injection used throughout (also hardcoded in the bench): f0=4.23e-3 Hz, fdot=1e-18, amp=1e-23,
dt=10 s — a bright, ordinary mHz GB.
