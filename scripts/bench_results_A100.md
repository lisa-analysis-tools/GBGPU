# STFT GB: FFT-per-column vs Fresnel — GPU benchmark result (A100-40GB)

**Verdict: NO-GO.** On an NVIDIA A100-SXM4-40GB, the `FFTColumn` `get_ll` kernel is
**3.2×–7.1× slower** than the analytic `FresnelColumn` kernel at production scale
(num_bin = 16384, n_side = 10), at matched accuracy. The gap is **structural** — it
does not shrink with problem size — so scaling up will not change the conclusion.
**Fresnel remains the production STFT-GB likelihood kernel.**

- **Machine:** erebor, 1× A100-SXM4-40GB (pinned via `CUDA_VISIBLE_DEVICES`), CUDA driver 12.8.
- **Branches:** LAT `feat-stft-gb` @ `6d010cf`, GBGPU `feat-stft-gb` @ `2e3a5dd`, GBT `feat-quintic-spline` @ `b07bd18`.
- **Date:** 2026-07-03. Raw data: `bench_stft_fft_A100.csv` (21 rows).

---

## 1. Wall-clock (the go/no-go number)

`get_ll` per call, min-of-5, tobs = 91 d, 6 h segments, Tukey α = 0.1, n_side = 10.

| num_bin | Fresnel (ms) | FFT n_sub=24, n_cp=48 | FFT n_sub=32, n_cp=48 | FFT n_sub=64, n_cp=48 |
|--------:|-------------:|----------------------:|----------------------:|----------------------:|
| 1024    | 78.8         | 249.0  (3.16×)        | 313.4  (3.98×)        | 579.2  (7.35×)        |
| 4096    | 311.7        | 987.6  (3.17×)        | 1229.5 (3.95×)        | 2213.1 (7.10×)        |
| **16384** | **1243.5** | **3955.6 (3.18×)**    | 4924.1 (3.96×)        | 8809.5 (7.08×)        |

Orbit-cache off (`n_cp=0`) is uniformly worse: at num_bin=16384, n_sub=32 → 7620 ms
(6.13×); n_sub=64 → 14581 ms (11.7×). So R2 (the orbit spline cache) is load-bearing
(~1.5–1.7×) but nowhere near enough to reach parity.

**Best FFT config at production scale = n_sub=24, n_cp=48 → still 3.18× slower.**

### Scaling — why "bigger" won't help
Both kernels are GPU-saturated and **linear in num_bin** (Fresnel µs/bin flat at
76.9 → 76.1 → 75.9 across 1024→4096→16384; FFT ratio dead-stable at **3.16 → 3.17 → 3.18**).
FFT cost is **~linear in n_sub** (envelope sampling dominates post-R1): ratio 3.18 (n_sub=24)
→ 3.96 (32) → 7.08 (64). Because the ratio is constant across a 16× range of num_bin, the
production-leaning `--tobs-days 365 --num-bins 32768` point is a safe extrapolation (~3.2×).

## 2. Accuracy (matched)
Whole-grid recovery mismatch vs the injected true STFT (n_side=10, α=0.1, 91 d):
Fresnel `mm = 3.88e-3`; FFT `mm = 4.65e-3 (n_sub=24) → 4.40e-3 (32) → 4.39e-3 (64)` —
same order, FFT converging down with n_sub. Accuracy is matched for the timing comparison
(the CPU campaign's in-band decomposition, not re-measured here, established FFT ≥ Fresnel
in-band; either way timing is the deciding factor). Orbit-cache (`n_cp=48`) is bit-consistent
with direct (`n_cp=0`): identical mm.

### Correctness gates (passed)
- GPU built-in gate (injection recovery) passes for both kernels.
- **CPU↔GPU cross-check: identical to 4 sig figs** (smoke, 8 d): Fresnel mm 2.152e-2 (CPU) = 2.152e-2 (GPU);
  FFT mm 2.548e-3/2.454e-3 (CPU) = (GPU); d_h-vs-Fresnel rel 3.352e-2/3.836e-2 both. GPU kernels are numerically correct.

## 3. Occupancy — why FFT is slower (root cause)

`ncu` achieved-occupancy is **blocked on this cluster** (`ERR_NVGPUCTRPERM`, non-root perf
counters disabled). Static per-kernel footprints from `cuobjdump -res-usage` (authoritative
for the occupancy limiter) on `gbgpu_backend_cuda12x`:

| kernel | registers/thread | local (stack)/thread | **shared/block** |
|---|---:|---:|---:|
| Fresnel `stft_get_ll_kernel`     | 212 | 832 B  | **4304 B (~4.2 KB)** |
| FFT `stft_get_ll_fft_kernel`     | 174 | **3824 B** | **31280 B (~30.5 KB)** |

Both of the brief's feared limiters are confirmed:
1. **Shared-memory occupancy wall.** The FFT kernel needs **~30.5 KB shared/block** (the R2
   orbit spline cache) vs Fresnel's 4.2 KB. On A100 with the default 48 KB shared carveout that
   is ~1 block/SM (Fresnel fits many) — occupancy is starved, hiding little of the memory latency.
2. **Register spill to local memory.** The `slow[3·n_sub]` twiddle-DFT buffer (sized for the
   compile-time `STFT_FFT_NSUB_MAX = 64`) does **not** fit in registers and spills to **3824 B of
   per-thread local memory** (hence FFT uses *fewer* registers, 174 < 212 — the buffer left the
   register file). Every `slow[]` access is then off-chip.

So the FFT "fewer transcendentals" op-count advantage (real, and it wins on op count) is defeated
on the GPU by **memory pressure** (shared-mem-limited occupancy + local-memory traffic) plus the
raw extra work of sampling the envelope at n_sub midpoints/segment and DFT-ing it. This is largely
intrinsic to the algorithm, not a tuning miss.

### Knobs assessed (brief §5) — none reach parity
- `n_sub=24` is **already the best** config (3.18×); it is the floor (rule `n_sub ≥ 2·n_side+1 = 21`).
- Lowering `STFT_FFT_NSUB_MAX` 64→32 (+GBGPU rebuild) would ~halve the 3824 B local spill, but the
  30.5 KB shared cache (the dominant occupancy limiter) is unchanged.
- `n_cp=24` halves the shared cache toward ~15 KB (better occupancy) but trades away cache hit rate;
  `n_cp=48` already beat `n_cp=0`, so this is unlikely to net below ~3×.
- `__launch_bounds__`/`maxrregcount` cannot fix a shared-memory occupancy limit or the extra arithmetic.
Even stacked, these do not plausibly close a 3.2× gap whose dominant cause is shared-memory occupancy.

## 4. Recommendation
- **Keep Fresnel as the production STFT-GB kernel** (now arbitrary-taper-correct, LAT `6d010cf`).
- Retain `FFTColumn` as the **accuracy oracle / any-window escape hatch** (its per-sample windowing is
  exact for any window; immune to segment length and taper).
- Do **not** invest in the FFT swap/grad/fstat + JAX surface for performance — the get_ll wall-clock
  already decides it. (Phase B is a NO per the brief's decision grid.)
- If FFT is ever needed for a wide-band (1-day-segment) config where n_sub must exceed 64, the shared-mem
  wall gets worse, not better — budget accordingly.

## 5. Reproduce
```
# Env: dedicated venv .venv-stft-gpu built from the feat-stft-gb worktrees.
# CUDA runtime: pip CUDA wheels on LD_LIBRARY_PATH + CUDA_HOME UNSET (cupy uses bundled nvrtc).
# See tasks/todo_stft_gpu_bench.md / memory 'stft-gb-gpu-venv-recipe' for the full recipe + gpurun.sh.
CUDA_VISIBLE_DEVICES=<idle> bash gpurun.sh scripts/bench_stft_fft_vs_fresnel.py --csv bench_stft_fft_A100.csv
# static occupancy footprints (no perf-counter perms needed):
cuobjdump -res-usage <site-packages>/gbgpu_backend_cuda12x/cgbgpu*.so | grep -A1 stft_get_ll
```

## Notes
- `bench_stft_fft_vs_fresnel.py` was patched for GPU: it passed `gpus=None` (fine on CPU; GPU asserts
  "device is None") → now `gpus=[0] if on_gpu else None` (pin one GPU via `CUDA_VISIBLE_DEVICES`).
- CPU test suite on this GPU box: failures are the **known orbits-default-to-GPU test-harness gap**
  (`OrbitsWrapCPU` fed `cupy.ndarray`; see repo `tasks/todo.md`), not a build or bench defect.
