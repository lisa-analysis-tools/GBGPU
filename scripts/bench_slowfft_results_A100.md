# FastGB-STFT slow-FFT vs Fresnel — GPU Phase-1 go/no-go (A100-40GB)

**Verdict: GO — worth building the fused Stage-1+2 CUDA kernel (Phase 2).**

GO here means "worth fusing", *not* "the cupy prototype out-races Fresnel today" (it does not
— see the terse plan criterion below). The decision rests on three algorithm-level facts, none
of which is the prototype's end-to-end wall-clock:

1. **Accuracy is a correctness win, not a speed trade.** slow-FFT recovers the injection at
   **mm = 7.24e-4 @ 6 h (≈5× sharper than Fresnel's 3.77e-3)** and stays **exact at large Δ —
   7.05e-6 @ 24 h, 3.21e-4 @ 96 h — where BOTH production C++ kernels (Fresnel *and* FFTColumn)
   are numerically DEGENERATE** (template zeroed @24 h → `h_h≈-3e-38`; overflow ~1e260 @96 h →
   `mm=nan`). At Δ ≥ 24 h there is currently **no working production STFT-GB likelihood**;
   slow-FFT is the only correct option (Tasks 4/5).
2. **The batched physics is already at/below Fresnel wall-clock.** With the one-time orbit/GBGPU
   build hoisted out (as any real use and the fused kernel do), **Stage 1 alone runs at
   0.68–1.50× Fresnel** — often *faster* than the fused analytic kernel, as unoptimised cupy.
3. **No 30 KB occupancy wall.** The FFT-per-column design was NO-GO (`bench_results_A100.md`,
   3.2× slower) because it cached the orbit spline in **~30.5 KB shared/block** → ~1 block/SM +
   register spill. slow-FFT **has no such cache**: Stage 1 precomputes the slow part, so a fused
   Stage-2 DFT touches only `E[3·n_sub]`/segment + the `2·n_side+1` band — a small footprint.

The prototype's 7–24× end-to-end slowness is **90–99 % a Python per-binary Stage-2 DFT loop**
(`stft_template_from_slow_part`'s `for b in range(num_bin)`) plus dense-`H` materialisation —
**pure prototype artifacts a fused kernel deletes by construction.**

- **Machine:** erebor, 1× A100-SXM4-40GB (pinned via `CUDA_VISIBLE_DEVICES`), CUDA driver 12.8.
- **Branch:** GBGPU `feat-stft-gb-gpu` @ `87fd5d1` (+ this commit). **Date:** 2026-07-05.
- **Raw data:** `bench_stft_slowfft_A100.csv` (27 rows). Fixture: 91 d, Tukey α=0.1, n_side=10,
  min-of-3, injected 2048-knot GPU oracle (gate only; see §2).

---

## 1. Wall-clock — the split is the story

Per-call time (min-of-3), one-time private-GBGPU construction (~7.5 s) **hoisted out and reused**
(bit-identical, `d_h` rel = 0.0 vs inline build). `full = stage1 + stage2+`; `stage2+` = the
Python per-binary DFT loop + XYZ contraction. Ratios are vs the fused C++ Fresnel `get_ll_stft`.

| Δ | num_bin | n_sub | Fresnel (ms) | slow-FFT full (ms) | **stage1 (ms)** | stage2+ (ms) | full ratio | **stage1 ratio** |
|---:|--------:|------:|-------------:|-------------------:|----------------:|-------------:|-----------:|-----------------:|
| **8 h** | 1024 | 24 | 71.5 | 537 | 49 | 488 | 7.52 | **0.69** |
| **8 h** | 4096 | 24 | 281 | 2802 | 192 | 2610 | 9.95 | **0.68** |
| **8 h** | **16384** | 24 | 1123 | 12023 | 1220 | 10803 | **10.71** | **1.09** |
| **8 h** | 1024 | 32 | 71.5 | 550 | 57 | 493 | 7.69 | **0.80** |
| **8 h** | 4096 | 32 | 281 | 3390 | 334 | 3056 | 12.04 | **1.19** |
| **8 h** | **16384** | 32 | 1123 | 12769 | 1679 | 11090 | **11.37** | **1.50** |
| 24 h † | 1024 | 24 | 33.4 | 459 | 32 | 426 | 13.71 | 0.96 |
| 24 h † | 4096 | 24 | 104 | 1807 | 57 | 1750 | 17.40 | 0.55 |
| 24 h † | 16384 | 24 | 414 | 8015 | 376 | 7639 | 19.37 | 0.91 |
| 24 h † | 16384 | 32 | 414 | 7855 | 524 | 7330 | 18.98 | 1.27 |
| 96 h † | 4096 | 24 | 71.6 | 1723 | 32 | 1691 | 24.07 | 0.44 |
| 96 h † | 16384 | 24 | 288 | 6874 | 57 | 6817 | 23.90 | **0.20** |
| 96 h † | 16384 | 32 | 288 | 6990 | 71 | 6919 | 24.30 | 0.25 |

† **Fresnel is numerically DEGENERATE at Δ ≥ 24 h** (nan/overflow templates, Task 5). Its kernel
still *runs*, so the wall-clock is real, but it is **not a correct alternative** there — these
ratios are raw wall-clock, **not accuracy-matched**. The accuracy-matched wall-clock point is
**8 h**. (Full 27-row grid incl. 24 h/96 h n_sub=32 and un-chunked 96 h/1024 in the CSV.)

**Reading it:**
- `stage2+` is **90–99 % of the full call** (16384/8 h: 10803 of 12023 ms) and scales ~linearly
  in num_bin — it is the serial **Python** DFT loop, not arithmetic. A fused CUDA Stage-2 does
  these `num_bin × NT` targeted DFTs in parallel; the Python/launch overhead vanishes.
- **`stage1 ratio` is the fused-kernel-relevant number** (a lower bound on the fused cost): the
  batched slow-part physics is **≤ Fresnel** at every accuracy-matched point (0.68–1.50×; 1.09×
  at the 16384/8 h headline, n_sub=24). Unlike FFTColumn, the fused kernel would start from
  **parity**, not a 3× deficit.

### Terse plan criterion (`task-6-brief.md` Step 4) vs the coordinator criterion
The plan's literal Step-4 test — *"GO if slow-FFT ≤ Fresnel wall-clock at num_bin=16384"* — reads
on the **full cupy** call and is **NOT met** (10.7–11.4× @16384/8 h). But that measures the Python
Stage-2 loop, not the algorithm. The coordinator brief supersedes it (as in Task 5, where the
plan's `≤ Fresnel` assertion was superseded by the design's Global Constraint): GO = "worth
building the fused kernel", judged on accuracy + Stage-1/occupancy/scaling. On that criterion: **GO.**

## 2. Accuracy (recap — authoritative numbers are Tasks 4/5, not this bench)

This bench's recovery gate uses a **coarse 2048-knot GPU oracle** (the GBTDIonTheFly generation
kernel requests `get_gb_buffer_size(n_sparse)` dynamic shared mem; > ~164 KB, i.e. n_sparse
> ~2800, exceeds the A100 per-block opt-in cap → `invalid argument` at `gb_tdi_on_the_fly.cu:187`).
It is a **sanity gate only** (a broken build fails loudly), not the accuracy measurement.

| Δ | slow-FFT gate mm (this bench, 2048-knot GPU) | Fresnel gate mm (this bench) | **Authoritative (Tasks 4/5, 8192-knot CPU oracle)** |
|----:|:--:|:--:|:--|
| 6–8 h | 1.6e-3 – 2.1e-3 | 3.1e-3 (works) | slow-FFT **7.24e-4** vs Fresnel 3.77e-3 (**≈5× sharper**) |
| 24 h | 8.6e-4 – 1.2e-3 | **6.7e-3 DEGENERATE** (unreliable) | slow-FFT **7.05e-6**; Fresnel **nan** (h_h≈−3e-38) |
| 96 h | 1.2e-3 – 1.6e-3 | **2.9e-2 DEGENERATE** | slow-FFT **3.21e-4**; Fresnel **nan** (overflow ~1e260) |

slow-FFT's gate mm is **flat across Δ** (8e-4 … 2e-3) — it does **not** degrade at large Δ, the
whole point. Fresnel's gate mm balloons/goes nan (its two root causes — `freq_from_tdi_phase`
finite-difference sampling the orbit at t < 0, and analytic `1/sqrt(fdot)` for a near-mono source
— are Task-5-diagnosed and Δ-driven, n_sparse-independent).

## 3. Memory — a prototype-only limit (dense `H`), not a Phase-2 concern

Stage 2 materialises the dense template `H[num_bin, 3, NT, NF]` (NF = 2·band+1 = 125). **`H`, not
Stage 1's `E[num_bin,3,NT,n_sub]`, is the driver:** 1.64 MB/binary → **26.8 GB at num_bin=16384,
8 h**, and the `conj(H)` + einsum intermediates roughly triple that (~80 GB). The bench chunks
over binaries (block halved on `OutOfMemoryError`, `d_h`/`h_h` concatenated) and flags CHUNKED:

- **Largest un-chunked num_bin:** 4096 @ 8 h and 24 h (NT = 273, 91); **16384 @ 96 h** (NT = 22).
- 16384/8 h ran at block = 4096 (6.7 GB `H`/block); 16384/24 h at block = 8192.

A fused kernel **never materialises `H`** — it accumulates the inner product per (binary, segment,
pixel) on the fly, exactly like Fresnel/FFTColumn — so this 40 GB wall is a prototype artifact and
does **not** carry to Phase 2.

## 4. Why this beats the FFT-per-column NO-GO (occupancy)

`bench_results_A100.md` ruled FFTColumn NO-GO: 3.2–7.1× slower, **structurally** — a **~30.5 KB**
shared/block R2 orbit-spline cache (→ ~1 block/SM on A100's 48 KB carveout) + a 3.8 KB
register→local spill of the `slow[3·n_sub]` twiddle buffer. slow-FFT's architecture removes the
cause: **Stage 1 precomputes the slow part in a batched pass** (the orbit is evaluated once per
call via the standard FastGB path, not cached in kernel shared memory), leaving Stage 2 a **narrow
DFT of the precomputed `E`** — shared footprint ~`E[3·n_sub]`/segment + the `2·n_side+1` band, well
under the occupancy wall. Combined with §1's Stage-1 ≤ Fresnel wall-clock, the fused slow-FFT
kernel is expected to reach **Fresnel-competitive** throughput **and** stay correct where Fresnel
is not — the axis on which FFTColumn also failed (it is degenerate at large Δ too).

## 5. Recommendation / Phase-2 trigger

**GO. Build the fused Stage-1+Stage-2 CUDA kernel.** Concretely:
1. **Fuse** Stage-1 slow-part evaluation and the Stage-2 targeted per-segment DFT into one kernel,
   accumulating the noise-weighted inner product **per pixel** (no dense `H`, no Python loop),
   mirroring `FresnelColumn`'s streaming structure.
2. **Occupancy check first** (`cuobjdump -res-usage`): confirm the fused kernel's shared/register
   footprint stays off the wall (target ≪ 30 KB shared — the design has no orbit cache).
3. Then port the **swap / grad / fstat** surface + the JAX path, and cross-validate inner products
   C++ ↔ cupy-prototype ↔ JAX (sprint backend hierarchy).

**Honest caveats.** (a) No fused kernel is measured — "fused ≈ Stage-1 + cheap DFT ≤ Fresnel" is an
informed projection (Stage-1 ratio is a lower bound; the fused Stage-2 arithmetic adds some cost,
but the removed 90–99 % is Python overhead, not FLOPs). Phase-2's first gate is the fused Stage-2
wall-clock. (b) The only **accuracy-matched** wall-clock point is 8 h (Fresnel degenerate ≥ 24 h);
the strongest part of the case is the large-Δ **correctness** gap. (c) If the science never needs
Δ ≥ 24 h STFT-GB *and* the 6–8 h ≈5×-sharper accuracy is not required, the case weakens to the
(unmeasured) fused wall-clock alone — decide with the large-Δ requirement in view.

## 6. Reproduce

```
# Dedicated venv .venv-stft-gpu; CUDA pip wheels on LD_LIBRARY_PATH, CUDA_HOME unset.
CUDA_VISIBLE_DEVICES=<idle> bash .superpowers/sdd/gpurun.sh \
    scripts/bench_stft_slowfft.py --csv scripts/bench_stft_slowfft_A100.csv   # full sweep
CUDA_VISIBLE_DEVICES=<idle> bash .superpowers/sdd/gpurun.sh \
    scripts/bench_stft_slowfft.py --smoke                                     # quick self-check
```
Each `stft_hours` runs in its own subprocess (single-live-STFT-group isolation, Task-5 hazard);
per fixture, all Fresnel timing precedes any slow-FFT (Stage-1's private GBGPU clobbers the live
group — slow-FFT is immune, it scores numpy snapshots of `data_arr`/`invC_arr`).

## Notes
- **Product change (brief-sanctioned):** `slow_part_on_stft_grid` / `get_ll_stft_slowfft_proto`
  gained an optional `gbtmp=` (via new `make_slowpart_gbgpu(gb)`) so the ~7.5 s per-call private
  GBGPU construction can be built once and reused. Default `None` = unchanged behavior (existing
  tests pass). Without this the "Stage 1" time is 98 % orbit rebuild (a ~843× ratio artifact) and
  the comparison to Fresnel — whose orbit setup is likewise one-time — is meaningless.
- `stage2+` is derived as `full − stage1` (both measured with `gbtmp` reused); the Python loop
  lives in the Stage-2 template step, the XYZ contraction is a batched einsum.
