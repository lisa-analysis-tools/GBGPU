# STFT GB likelihood — batched slow-part + narrow per-segment FFT ("FastGB-STFT")

**Date:** 2026-07-04
**Status:** Design (brainstormed; architecture approved)
**Supersedes for the efficient path:** the FFT-per-column producer (NO-GO, see
`2026-07-01-stft-gb-fft-per-column-design.md` §13 / `scripts/bench_results_A100.md`).
**Branch:** `feat-stft-gb-gpu` (off `feat-stft-gb`).

---

## 1. Motivation

STFT is a first-class, permanent domain for GB likelihoods — co-equal with WDM as a
handle on non-stationary noise. WDM is developed separately; **this spec owns making the
STFT GB path both faster and more accurate than today's kernels.**

Two STFT column producers exist on the shared driver (`stft_eval_block_ll`,
`lat_stft_kernels.hh`):

- **FresnelColumn** — closed-form Fresnel C/S per pixel; production; complete surface
  (`get_ll`/`fill_global`/`swap`/`grad`/`fstat`). Weaknesses: **SFU/transcendental-bound**
  (~22 `sincos`/pixel), and a **constant-envelope model error** that grows with segment
  length Δ (in-band ~7e-6 @6 h → ~8e-5 @24 h; recovery ~5e-3 @4-day).
- **FFTColumn** — samples the true envelope per segment + twiddle-DFT. **NO-GO on GPU**
  (A100 bench 2026-07-03): 3.2–7.1× slower than Fresnel at num_bin=16384. Root cause
  (`cuobjdump -res-usage`): it re-evaluates the TDI response **on the fly** at
  `NT·n_sub ≈ 12k` points/binary through a **~30 KB/block shared orbit cache** (occupancy
  wall) plus a `slow[3·n_sub]` **local spill**.

**Operating point (production target):** Δ ∈ [~8 h, few days] (large segments ⇒ Fresnel's
Δ-error is real); Tukey taper fixed at ~1e-4 Hz so α floats down as Δ grows. **Accuracy
target: in-band mismatch ≲ 1e-5** (lower if it can be had for free). **Both** goals at
once: faster than Fresnel *and* exact.

## 2. Core idea — generalize the classic `GBGPU` (FastGB) heterodyne

The `GBGPU` class computes the LISA response **once** at N coarse time points as a
heterodyned "slow part" (`_construct_slow_part` → `_computeXYZ`), reusing
binary-independent orbit geometry across the whole population, then one small FFT → a
compact N-bin FD waveform. Its GPU efficiency comes from (a) heterodyne ⇒ smooth,
narrow-band envelope, and (b) the response evaluated **batched, with orbits amortized
across all binaries**.

Bring exactly that into STFT (no spline, no `tdi_on_the_fly`):

1. **Evaluate the heterodyned slow part `E_c(τ)` (c = X/Y/Z) directly on the STFT
   sub-grid** `τ = t_seg + (m+½)·dt/n_sub` (NT segments × n_sub sub-samples), using the
   `_construct_slow_part` formulas — orbits / `kdotr` / `kdotP` computed once on the shared
   time grid and broadcast over binaries; only per-binary sky projections + transfer
   functions vary.
2. **Narrow per-segment FFT:** window each segment's `n_sub` samples and DFT to just the
   `2·n_side+1` bins around the carrier (reuse the existing transcendental-free twiddle
   recurrence), contract with the per-segment `invC_j`, accumulate `⟨d|h⟩`, `⟨h|h⟩`.

Same pixel math as FFTColumn; the response is evaluated where FastGB already proved it is
GPU-efficient.

## 3. Why it beats both current kernels

- **vs FFTColumn:** identical `NT·n_sub` point count, but **no per-block 30 KB orbit cache**
  (the occupancy wall is gone by construction), orbits are **amortized across all binaries**
  instead of re-cached per source, memory access is coalesced, and the work is FMA-bound.
  The NO-GO's measured root cause is removed.
- **vs Fresnel:** **exact** — `E` carries the within-segment variation, so the
  constant-envelope Δ-error vanishes (accuracy at large Δ, the target regime). **FMA-bound**
  (no per-pixel `sincos`) — the FMA-vs-SFU speed lever.
- **Cross-domain consistent** with the FD/FastGB path by construction (same slow-part
  response model) — the `test_stft_gb_crossdomain` FD checks should pass by design.

## 4. Architecture / components

### Stage 1 — heterodyned slow part on the STFT grid
- **In:** params `[num_bin×9]`, STFT grid (`t_seg[NT]`, `dt`, `n_sub`, `n_side`, `f_min`,
  `df`), orbits, `TDIConfig`, carrier heterodyne convention.
- Reuse the `_construct_slow_part` math (`kdotr`, `kdotP`, transfer `Gs`, phasing) evaluated
  at the `M = NT·n_sub` sub-grid times. Orbit geometry at the M times computed **once** and
  broadcast; per-binary sky/amplitude/phase vectorized.
- Heterodyne against a per-source carrier (global `f0 = q/T`, FastGB-style) so `E` is smooth;
  the small per-segment residual drift is absorbed by Stage 2's DFT (a per-segment carrier
  bin `freq_j`, exactly as FFTColumn already computes).
- **Out:** `E[num_bin, NT, n_sub, 3]` (complex).

### Stage 2 — narrow per-segment DFT + inner product
- Per (binary, segment): apply the analysis window (Tukey, matched to the data STFT), DFT
  the `n_sub` samples to `2·n_side+1` bins via the existing twiddle recurrence
  (`stft_unit_cpow`), scale by `0.5·dts_sub`.
- Contract each pixel with `invC_j` (3×3 XYZ) via `add_ip_contrib`; reduce `⟨d|h⟩`,`⟨h|h⟩`
  per binary.
- **Footprint:** no orbit cache; `slow[]` sized to the actual `n_sub` (~`2·n_side+1`), not
  `STFT_FFT_NSUB_MAX=64` ⇒ the local spill shrinks and occupancy recovers.

### Integration with `STFTGBComputations`
- New Python entry `get_ll_stft_slowfft` in `gbcomps.py`, parallel to `get_ll_stft`
  (Fresnel) / `get_ll_stft_fft`.
- Prototype Stage 1 with the existing cupy `_construct_slow_part` (fastest path to validate
  accuracy + speed); Stage 2 as a light CUDA kernel that consumes `E` and reuses the
  driver's inner (window × DFT × `add_ip_contrib`) loop.
- If it wins on wall-clock: fuse Stage 1 into CUDA (drop the `[bin, M]` intermediates), then
  carry the full surface (`fill_global`/`swap`/`grad`/`fstat`) + a JAX mirror.

## 5. Data flow

```
params ─▶ Stage 1 (batched slow part on STFT grid) ─▶ E[bin, NT, n_sub, 3]
       ─▶ Stage 2 (window + narrow DFT + invC_j contraction) ─▶ ⟨d|h⟩,⟨h|h⟩ ─▶ lnL
```

## 6. Correctness / accuracy plan

- **Ground truth = injected brute STFT**, never Fresnel (Fresnel has its own model error).
- **Recovery gate:** mismatch vs injection ≤ ~1e-5 in-band; must converge as `n_sub` grows,
  and hold the `n_sub ≥ 2·n_side+1` rule.
- **Cross-checks:** (a) CPU↔GPU inner products agree ≲1e-12 (FP order aside); (b) agreement
  with the FD/FastGB path (Parseval / `test_stft_gb_crossdomain`) — expected by construction;
  (c) matches Fresnel where Fresnel is accurate (short Δ) and **beats it at large Δ** (the
  explicit win condition).

## 7. Risks / open questions

- **cupy-prototype memory:** the `[bin, M]` intermediates (~GBs at 6 h / `n_sub`=32) —
  mitigate by chunking over binaries, or move to the fused kernel.
- **Wall-clock is not yet proven.** `M = NT·n_sub` equals FFTColumn's point count; the win
  is cost-per-eval + occupancy, not fewer evals. **Prototype-benchmark against Fresnel at the
  production operating points before committing to the full surface** (this is the Phase-1
  go/no-go, same discipline as the FFTColumn benchmark).
- **Carrier/heterodyne convention** (global `f0` vs per-segment `freq_j`) sets the residual
  drift phase Stage 2 must apply; pick and validate against injection.
- **Short-taper skirts** (small α at large Δ) vs the `2·n_side+1` band-truncation floor —
  window-driven, shared with all methods; may set a minimum `n_side`.

## 8. Phasing

1. **Prototype + benchmark** `get_ll_stft_slowfft` (cupy Stage 1 + light Stage 2): validate
   recovery mm at the operating points (Δ = 8 h … 4 d) and benchmark wall-clock vs Fresnel at
   num_bin up to 16384. **GO/NO-GO gate.**
2. **If GO:** fuse Stage 1 into CUDA — **precompute orbit geometry at the M grid points into
   a global array shared across all binaries (NOT a per-block shared cache)**, so the 30 KB
   occupancy wall cannot return; re-check occupancy.
3. **Full surface:** `fill_global`/`swap`/`grad`/`fstat`; JAX mirror (inner-product parity
   ≤1e-12).

## 9. Alternative kept in reserve

**Polyphase / signal-heterodyne filter bank** (the FFT-native, spline-free way to extract
segments from the compact representation): more complex, and conceptually parallel to the WDM
`signal-het` path. Revisit if Stage 1's M-point response evaluation proves too costly, or as
the ultimate-throughput design.
