# STFT/Fresnel GB likelihood — FFT-per-column generation (design)

**Date:** 2026-07-01
**Status:** Design (brainstormed, pending user review → implementation plan)
**Goal metric:** throughput for O(10⁴) distinct galactic binaries (GBs) on GPU.
**Scope of this doc:** a `get_ll` prototype + head-to-head benchmark vs the current
Fresnel kernel. Swap/grad/fstat and the JAX mirror follow only if `get_ll` wins.

---

## 1. Motivation

The just-merged STFT/Fresnel GB likelihood (`lat_stft_kernels.hh`) fills each
time-frequency (TF) pixel **analytically** via a Fresnel integral. The FD GB path
(`gbfd_build_one_source`, `gb_tdi_on_the_fly.cu:348`) instead builds each source by
evaluating the response on a sparse time grid → windowing → **one radix-2 FFT over
Tobs** → extracting the band. The proposal (from the WDM collaborator) is to apply
that same "build slow-part → FFT" idea **per STFT segment** — many tiny FFTs — since
an STFT *is* a sequence of short windowed FFTs.

### Cost rationale (why it should win on throughput)
- **Windowed Fresnel is dominated by the window.** `get_windowed_fourier_value`
  (`domains.cu:756`) expands a Tukey window into **7 Fresnel sub-interval evaluations**
  → **~22 sin/cos per (pixel, channel)**, over `(2·n_side+1)` pixels × 3 channels × NT
  columns. (Rectangular is ~4 sincos/pixel but needs more bins for leakage.)
- **FFT-per-column replaces that with multiplies.** Per column: heterodyne the
  slow-part on ~`N_sub` sub-samples (1 sincos each), **apply the window as a plain
  time-domain multiply (free)**, and a small transcendental-free DFT/FFT yields *all*
  the column's bins at once.
- Dominant transcendental term drops **~5× (rectangular) to ~30× (windowed)**.
- **Bonuses:** numerically exact (removes the ~2e-3 Fresnel-approx floor,
  `domains.cu:659`), and it collapses the STFT generator onto the *same* machinery the
  FD path already uses (one generation core, not two).

The FLOP argument is decisive enough to build; the **GPU wall-clock is the actual
verdict** (this workstation has no `nvcc`).

---

## 2. Reuse boundary — STFT is NOT WDM  ⚠️

STFT and WDM differ fundamentally, so reuse is deliberately narrow:

**REUSE (domain-agnostic):**
- `get_tdi` — LISA response amp/phase over an **arbitrary time array** (base-class UCB
  closed-form physics, `gb_tdi_on_the_fly.hh:85,114`). *The Fresnel STFT kernel already
  calls this.*
- The generic **coarse-spline cache of `get_tdi` outputs** (`response(t)` on a coarse
  grid, evaluate cheaply at any `t`) — the mechanism behind `gb_wdm_spline_eval_inputs`,
  but the operation itself is domain-neutral.
- The **structure** of `gbfd_build_one_source` (slow-part → window → transform → band),
  re-applied per segment.
- The existing STFT accumulation + `stft_block_reduce_cmplx` and `STFTDomain`
  data-coefficient access.

**DO NOT reuse (WDM-transform-specific):** the polyphase fold / m-layer / FD-bin-gather
/ Hermitian / `layer_df` machinery (`gb_tdi_on_the_fly.cu:1777+`). WDM is **defined from
the frequency domain and frequency-folded**; STFT data is `(num_times, num_freqs)`,
time-then-frequency. A per-time-bin FFT produces **exactly one `(num_freqs)` row** of the
STFT array — a direct fit, no folding, no FD origin.

---

## 3. Architecture

Keep the current `stft_get_ll` skeleton: **one block per binary**, grid-stride over
binaries, **thread-per-column** (time bin), **in-register** small transform, block-reduce
to `(d|h),(h|h) → logL`. Only the per-column inner changes.

Per binary:
1. Broadcast params to shared; build one **coarse response spline** over the analysis
   span (cheap per-segment sub-sampling; default path — see §9 for the spline-vs-direct
   `get_tdi` benchmark).
2. Grid-stride over columns (time bins). For each column `time_i`:
   a. Sub-sample the response spline at `N_sub` points (start `N_sub ∈ {16, 32}`, tuned by the §8 convergence test) in `[t0+time_i·stft_dt, +stft_dt]`
      → `amp[n], phase[n]` per channel.
   b. `slow[n] = amp[n]·w[n]·exp(i(phase[n] − 2π·f0_grid·τ_n))` — **window = multiply**.
   c. **Targeted DFT** (compute only the `~2·n_side+1` needed bins; register-friendly, no
      bit-reversal, exact bins) → column spectrum `H[k]`. (Swap to radix-2 FFT only if
      `N_sub` grows; at `N_sub~16` they're a wash and the DFT is simpler.)
   d. Accumulate `(d|h) += conj(d_row[j])·H[j]/S`, `(h|h) += |H[j]|²/S` over those bins.
3. Block-reduce → `(d|h),(h|h)` → logL.

---

## 4. DRY structure — one driver, pluggable column producer  ⚠️

Avoid duplicating the get_ll driver between Fresnel and FFT. Factor:

- **Shared:** the binary/column loop, param broadcast, the per-column `(d|h)/(h|h)`
  accumulation with noise weighting, `stft_block_reduce_cmplx`, and the whole
  wrapper/binding/`gbcomps` plumbing.
- **Pluggable column producer** as a **compile-time policy** (template param / functor —
  zero runtime cost, no divergent branch):
  - `FresnelColumn` — existing per-pixel analytic (behavior unchanged).
  - `FFTColumn` — new: response-spline sub-sample → heterodyne → window → targeted DFT.

Genuinely new code is only `FFTColumn` (+ the response-spline cache if the benchmark
shows it's needed). The Fresnel path is refactored onto the shared driver **only if it
stays bit-identical and is re-validated** — the compile-time policy makes that safe, but
we don't destabilize the just-merged kernel for its own sake. The same policy later gives
swap/grad/fstat their FFT variants without re-deriving drivers.

---

## 5. Data flow

- **Inputs:** params `(num_bin, nparams)`; data STFT coeffs `(num_times, num_freqs)` per
  channel; noise/PSD; `STFTDomain` grid `(t0, stft_dt, df, NT, NF, active band)`;
  `n_side`/`N_sub`; the analysis window.
- **Grid alignment (key):** a length-`N_sub` DFT spanning `stft_dt` has bin spacing
  `1/stft_dt = df` — the STFT frequency grid — so template bins map 1:1 onto the data
  column's bins. Carrier snapped to the grid (mirror the existing `f0_grid` snap).
- **Output:** per binary `(d|h), (h|h)` (re + im), identical surface to `get_ll_stft`.

---

## 6. Correctness invariants
1. **Window match** — the template applies the *same* analysis window the data STFT used
   (else `(d|h)` is biased).
2. **Carrier/grid snap** — output bins align with data bins.
3. **`N_sub` is the accuracy knob** (the analog of `n_side_bins`): sets how many bins are
   reproduced accurately; too small → far-bin error. Requires an `N_sub` convergence test.
4. **Identical noise weighting / normalization** to the Fresnel path (apples-to-apples).

## 7. Edge cases
- NaN scrub on singular response geometry — mirror `gbfd_build_one_source` (`.cu:443`).
- Observation must sit interior to the orbit (unchanged from current kernel).
- Off-grid `min_freq` — already snapped (merged fix); the FFT path inherits it.
- Carrier near the active-band edge — bin extraction clamps to the active band.

---

## 8. Test + benchmark plan
- **Accuracy (CPU here):** `FFTColumn` get_ll vs (a) `FresnelColumn` get_ll at wide
  `n_side`, and (b) brute-STFT of an injected GB. Targets: match brute-STFT to ~1e-4 (as
  Fresnel does — and possibly better, since FFT is exact); `N_sub` convergence curve
  (mirror the existing `n_side_bins` convergence test).
- **Cost (CPU here):** FLOP/instruction proxy + wall-clock at fixed `num_bin` for both
  kernels (relative, not absolute).
- **Cost (GPU box, deferred):** the real throughput number + occupancy vs Fresnel — the
  go/no-go decision. Prototype is structured so this is a drop-in.

## 9. Risks / open questions
- **Response sub-sampling cost** — the main uncertainty; mitigated by the spline cache
  (cheap polynomial evals). Benchmark confirms; if `get_tdi` is called directly per
  sub-sample instead, cost is `N_sub×` higher — measure both.
- **Register pressure** of an in-register DFT at larger `N_sub` → fall back to a
  warp-cooperative FFT (documented alternative).
- **Window spectral tails** may require `N_sub` somewhat larger than `2·n_side+1` for the
  far bins — the `N_sub` convergence test quantifies this.

## 10. Phasing
1. **Prototype:** `FFTColumn` get_ll (CPU C++) + accuracy + `N_sub` convergence + CPU cost
   proxy. Shared driver + policy in place.
2. **GPU benchmark** (GPU box) → go/no-go on throughput.
3. **If go:** swap/grad/fstat via the same column-producer policy; then the JAX mirror
   (discussion #2).

## 11. Non-goals (this iteration)
- Swap/grad/fstat FFT variants (follow after get_ll wins).
- JAX mirror (discussion #2, after the efficiency decision).
- A GPU-tuned final kernel — the prototype targets correctness + a benchmarkable
  structure, not peak occupancy.

---

## 12. Prototype results (CPU, 2026-07-01) — implemented + validated

The `FFTColumn` `get_ll` prototype is implemented (LAT `stft_get_ll_fft_{kernel,impl}`
+ `FFTColumn` policy on the shared driver; GBGPU `gb_stft_get_ll_fft` +
`STFTGBComputations.get_ll_stft_fft(n_sub)`) and validated on CPU. Full suite:
**28 passed + 2 subtests** (the original 23+2 Fresnel suite is **byte-identical** — the
policy refactor did not perturb it).

**✅ Correctness — the FFT is exact and beats Fresnel.** Against the ground-truth brute
STFT of an injected GB (n_side=20, n_sub=64): FFT recovery mismatch **3.94e-3 < Fresnel
4.14e-3**. The FFT is *more* accurate than the analytic Fresnel path, confirming the §1
claim that it removes the ~2e-3 Fresnel-integral floor. Convention/scale
(`0.5·(stft_dt/N)·Σ conj(tdi)·e^{-i2πfτ}`, **midpoint** τ_m for 2nd-order quadrature) is
correct first-time — no tuning needed beyond switching left→midpoint sampling.

**✅ `N_sub` is the accuracy knob, with a resolution floor.** Recovery mismatch vs n_sub
(n_side=20): `8→0.59, 16→0.49, 32→5.77e-3, 64→3.94e-3`. A **hard requirement emerged:
`n_sub ≳ 2·n_side+1`** to resolve the requested band — below it the far bins alias
(catastrophic). Above it, midpoint quadrature converges 2nd-order. (Design §6/§9's
"`N_sub` too small → far-bin error" — now quantified as an aliasing cliff, not a gentle
tail.)

**⚠️ CPU cost — the FFT prototype is ~10× SLOWER than Fresnel, and this is the decisive
open question.** At 64 binaries, n_sub=32: Fresnel **454 ms** vs FFT **4493 ms
(≈9.9×)**. Root cause: this prototype calls `get_tdi_Xf_single` **`N_sub`× per column**
(direct response evaluation) vs Fresnel's 1×, so the **response evaluation dominates** and
swamps the DFT's fewer-transcendentals saving. This is exactly the §9 *main uncertainty*,
now measured: **the response-spline cache is ESSENTIAL (not optional)** for the FFT path to
be competitive — it must amortize `get_tdi` across sub-samples. The §1 "~5–30× fewer
transcendentals" advantage only converts to throughput once (a) the spline removes the
`N_sub×` `get_tdi` cost and (b) the GPU's transcendental-bound regime is measured. Both
remain to prove.

**Go/no-go read:** correctness is **proven** (and better than Fresnel); a CPU **speed win
is not** (expected — no `nvcc` here, and the direct-`get_tdi` path is un-amortized). The
decision needs two more measurements: the **response-spline variant** (amortized `get_tdi`)
on CPU, then the **GPU wall-clock** on a GPU box. Until then this is a validated,
benchmarkable structure — not yet a throughput improvement.

**Implementation notes:** midpoint quadrature; `STFT_FFT_NSUB_MAX=64` bounds the on-stack
sample buffer (GPU register-pressure ceiling). Commits: LAT `133f645`,`3f2a0ba`,`bf0d3fd`;
GBGPU `2ba0a62`,`0df0e9f` (branch `feat-stft-gb-fft`).

### 12.1 Tukey windowing + 1-year segment-length scaling (2026-07-01)

`FFTColumn` now applies the analysis window as a **per-sample multiply** (the design's
"window = free": Tukey when `window_alpha>0`, matching `taper_duration = alpha*dt/2` /
`scipy.signal.tukey`; rectangular scales by `window_factor`). Validated: windowed FFT
recovers the injection (**4.68e-3 < windowed Fresnel 4.94e-3**). Full suite **29 passed + 2
subtests**. Commits: LAT `79024e0`; GBGPU `e608ebd`.

**1-year Tukey scaling** (Tobs ≈ 1 yr, α=0.1, n_side=10, n_sub=32), FFT vs Fresnel `get_ll`
swept over the STFT segment length Δ:

| Δ | #seg | Fresnel mm | FFT mm | FFT-vs-Fresnel | Fresnel ms | FFT ms | ratio |
|----|-----:|-----------:|-------:|---------------:|-----------:|-------:|------:|
| 6h | 1456 | 2.03e-3 | 2.20e-3 | 7.2e-4 | 549 | 3206 | 5.8× |
| 24h | 364 | 1.35e-3 | 1.55e-3 | 9.9e-4 | 129 | 817 | 6.3× |
| 4d | 91 | **5.07e-3** | **2.09e-3** | 4.5e-3 | 31 | 200 | 6.4× |

`n_sub` cliff (Δ=6h, n_side=10 → 21 bins): `8→0.42, 16→7.3e-3, 24→2.44e-3, 32→2.20e-3,
48→2.17e-3, 64→2.18e-3`.

**Findings (measured):**
1. **`n_sub ≳ 2·n_side+1` confirmed and Δ-invariant** — cliff exactly at 21 (n_side=10);
   n_sub=16 marginal, ≥24 clean, then plateaus. `df·Δ = 1` makes the requirement scale-free
   in segment length.
2. **Cost ratio ~Δ-invariant (~6×); absolute cost ∝ 1/Δ** — both methods scale with the
   segment count (Tobs/Δ), so longer Δ is cheaper for both and the ratio holds. The
   response-spline (not longer segments) is still the lever to flip the ratio.
3. **FFT accuracy advantage GROWS with Δ (the headline).** At 6h/24h the two agree tightly
   (Fresnel ≈ FFT). At **4-day segments Fresnel degrades to 5.07e-3 while the FFT holds
   2.09e-3 (~2.4× more accurate)**, and the FFT-vs-Fresnel gap grows 6× (7e-4 → 4.5e-3).
   Fresnel's per-segment model (constant amplitude + linear-chirp phase, anchored at the bin)
   approximates away the within-segment LISA **response modulation** (antenna-pattern
   rotation + Doppler) that becomes non-negligible over a multi-day segment; the FFT samples
   the true response → exact for any Δ. **So the FFT is the enabler for longer STFT segments**
   (fewer segments = cheaper + finer df) without the Fresnel accuracy breakdown. (This refines
   the earlier prediction: the growth is real even for a slow GB — driven by the response
   modulation, not only the astrophysical chirp.)
4. **Tukey helps the FFT.** Its faster sidelobe decay lets n_side=10 (vs ~20 rectangular) →
   better accuracy (~2e-3 vs ~4e-3) AND a smaller `n_sub` (24 vs ~40) → cheaper.

**Infra note found:** the LAT STFT computation groups do not support two live instances (a
second `STFTComputationGroup` build invalidates the first's C++ device buffers) — build + use
sequentially. Exposed by a windowed test reusing a cached rectangular group (spurious
mismatch=1.0); the benchmark and tests now build + use each fixture before the next.

### 12.2 GPU-efficiency restructures (2026-07-01) — the prototype was not the GPU design

The §12 prototype was correct but did MORE work than Fresnel (naive per-(bin,sample)
`sincos` DFT + direct `get_tdi` per sub-sample) → it would lose a GPU benchmark. Per the
backend hierarchy, CPU wall-clock is NOT a target; the two restructures below are the GPU
algorithm, validated on CPU by **correctness + operation counts** (single-source `.cu`,
CPU mirrors via `#ifdef`). Full suite after both: **30 passed + 2 subtests**.

**R1 — transcendental-free DFT (heterodyne + twiddle recurrence).** Demodulate to the
carrier bin (`N_sub` `sincos`, in place), then get the `2·n_side+1` targeted bins from
twiddle powers (roots of unity via short recurrence). Mathematically identical to the
direct DFT (`df·stft_dt=1`), so **all FFT tests reproduce bit-identical numbers**.
**Op-count: `sincos`/column `(2·n_side+1)·N_sub → N_sub+2`** (n_side=10, N_sub=32: **672→34**,
~20×). Realizes the design's fewer-transcendentals thesis. Commit LAT `e2dd70a`.

**R2 — orbit spline cache (amortize `get_tdi`).** Build the LISA orbit spline cache once
per block over `[t0, t0+NT·dt]` (`populate_orbit_spline_cache`; the orbit is
source-independent → shared across all binaries) and use `get_tdi_Xf_single_cached`, so the
expensive per-sub-sample light-travel-time/position lookups become cheap cached cubic
evals. The smooth (yearly) orbit is captured by a coarse cubic: **cache reproduces direct
`get_tdi` to 1.1e-9** (spline precision), recovery unchanged. **Op-count: expensive orbit
evaluations/binary `N_sub·NT·~48 → ~15·N_cp`** (default `N_cp=48`; e.g. NT=256,N_sub=64:
~786k → 720, ~1000×). `n_cp_orbit` knob (0=direct). Threaded `OrbitsSplineCache*` through
the shared driver + policies; Fresnel path untouched. Commits LAT `0085355`, GBGPU `b0baa8c`.

**Together:** the FFTColumn now does ~`N_sub` transcendentals + ~`15·N_cp` (shared) orbit
evals per binary, vs Fresnel's ~`22·(2·n_side+1)` `sincos` + `~3·NT` orbit evals — i.e. the
"fewer transcendentals + amortized response" thesis is realized at the op-count level. The
**GPU wall-clock** (occupancy, SFU throughput) remains the final go/no-go, off-box.

## 13. GPU verdict (A100-SXM4-40GB, 2026-07-03) — NO-GO

Answered on erebor (single A100-40GB). Full report + raw data:
`scripts/bench_results_A100.md`, `scripts/bench_stft_fft_A100.csv`.

**FFTColumn `get_ll` is 3.2–7.1× slower than FresnelColumn** at production scale
(num_bin=16384, n_side=10) at matched accuracy. Best FFT config (n_sub=24, n_cp=48)
= 3.18×. The FFT/Fresnel ratio is dead-stable and linear across num_bin (3.16 →
3.18 over 1024→16384) — the GPU is saturated, so scaling up does not close the gap.

**Root cause** (`cuobjdump -res-usage`; `ncu` achieved-occupancy blocked by
`ERR_NVGPUCTRPERM`): the FFT kernel needs **~30.5 KB shared/block** (the R2 orbit
cache → ~1 block/SM on A100's default carveout) and spills the `slow[3·n_sub]`
buffer (compile-time `STFT_FFT_NSUB_MAX=64`) to **3824 B/thread of local memory**,
vs Fresnel's 4.2 KB shared / 832 B local. The §12.2 op-count win is real but is
defeated on GPU by shared-memory occupancy + local-memory traffic plus the raw
per-segment envelope sampling. The §5 knobs (n_sub floor already 24; `NSUB_MAX`↓;
`n_cp`↓; `__launch_bounds__`) do not plausibly close a 3.2× shared-mem-bound gap.

**Decision:** Fresnel stays the production STFT-GB kernel (now arbitrary-taper-
correct, LAT `6d010cf`); FFTColumn is retained as the accuracy oracle / any-window
escape hatch. **Phase B (FFT swap/grad/fstat + JAX) is a NO.** Correctness: GPU
built-in gate passes; CPU↔GPU cross-check identical to 4 sig figs.
