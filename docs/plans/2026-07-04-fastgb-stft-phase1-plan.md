# FastGB-STFT — Phase 1 (prototype + go/no-go) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a cupy/numpy prototype of the FastGB-STFT GB likelihood (heterodyned slow part evaluated on the STFT sub-grid → narrow per-segment DFT → time-resolved inner product) and answer one question: **does it beat Fresnel's GPU wall-clock at matched accuracy?**

**Architecture:** Two stages, both in Python for the prototype so no new CUDA is needed. Stage 1 reuses `GBGPU._construct_slow_part` + the `_computeXYZ` time-domain assembly (`XYZsl`) evaluated on the STFT sub-grid (never FFT to FD). Stage 2 windows each segment and does a targeted DFT to the `2·n_side+1` band bins, then the produced template is scored via `STFTComputationGroup.compute_signal_likelihood_terms`. Accuracy is validated on CPU against the injected brute STFT; wall-clock is benchmarked on GPU against Fresnel.

**Tech Stack:** Python 3.12, numpy/cupy (`xp` pattern), lisatools STFT stack (`STFTComputationGroup`, `TDSignal.stft`), `gbgpu.GBGPU`/`STFTGBComputations`, pytest/unittest.

## Global Constraints

- Work in the dedicated venv `/data/asantini/globalfit/erebor_org_setup/.venv-stft-gpu`; GPU runs use the wrapper `$CLAUDE_JOB_DIR/tmp/gpurun.sh` (pip CUDA libs on `LD_LIBRARY_PATH`, `CUDA_HOME` unset). See memory `stft-gb-gpu-venv-recipe`.
- Branch: `feat-stft-gb-gpu` (off `feat-stft-gb`), in worktree `gpu_worktrees/GBGPU` (+ LAT worktree for reference reads).
- **Ground truth is the injected brute STFT, never Fresnel** (Fresnel has its own ~4e-3 model error). Recovery `mm = |1 − d_h.real / sqrt(d_d · h_h.real)|`.
- **`n_sub ≥ 2·n_side+1`** (Nyquist for the band); accuracy converges with `n_sub`.
- One live `STFTComputationGroup` at a time — each test builds its own fixture and fully uses it before building another (see `_build_fixture` docstring, `tests/test_stft_gb_fft.py`).
- Accuracy target: recovery `mm ≲ 1e-5` in-band at the production operating points (Δ ∈ [8 h, 4 d], Tukey taper ~1e-4 Hz), and **≥ Fresnel accuracy at large Δ**.
- `force_backend` is always a concrete string ("cpu"/"gpu"), never `None`.
- CPU C++ / cupy accuracy tests run under pytest; GPU wall-clock runs under `gpurun.sh` on an idle GPU pinned via `CUDA_VISIBLE_DEVICES`.

---

## File Structure

- **Create** `src/gbgpu/stft_slowfft_proto.py` — the prototype: `slow_part_on_stft_grid()` (Stage 1), `stft_template_from_slow_part()` (Stage 2), `get_ll_stft_slowfft_proto()` (assemble + score). One responsibility: the FastGB-STFT prototype numerics. Kept out of `STFTGBComputations` until the go/no-go passes.
- **Create** `tests/test_stft_gb_slowfft.py` — accuracy/convergence/large-Δ tests, reusing the `_build_fixture` pattern from `tests/test_stft_gb_fft.py`.
- **Create** `scripts/bench_stft_slowfft.py` — GPU wall-clock vs Fresnel (adapts `scripts/bench_stft_fft_vs_fresnel.py`).
- **Reference only** (read, do not modify): `LAT gpu_worktrees/.../gbgpu.py` (`_construct_slow_part`:534, `_computeXYZ`:483, `run_wave`:164, `special_get_N`:3104), `src/gbgpu/gbcomps.py` (`get_ll_stft`:564, `get_ll_stft_fft`:593), `tests/test_stft_gb_fft.py` (`_build_fixture`, `_make_gb`).

---

### Task 1: Test harness + prototype stub

**Files:**
- Create: `src/gbgpu/stft_slowfft_proto.py`
- Test: `tests/test_stft_gb_slowfft.py`

**Interfaces:**
- Produces: `get_ll_stft_slowfft_proto(grp, gb, params, n_sub) -> (d_h[num_bin], h_h[num_bin])` complex xp arrays. `grp` is the `STFTComputationGroup`; `gb` is an `STFTGBComputations` (source of orbits/tdi_config/n_side_bins/window/T/t_ref). Prototype stub returns zeros of the right shape/dtype until Task 4 fills it.

- [ ] **Step 1: Write the failing smoke test**

```python
# tests/test_stft_gb_slowfft.py
import unittest
import numpy as np
try:
    from tests.test_stft_gb_fft import _build_fixture, _make_gb   # reuse the reference fixture
    from gbgpu.stft_slowfft_proto import get_ll_stft_slowfft_proto
    HAVE = True
except (ImportError, ModuleNotFoundError) as _e:
    HAVE = False; _ERR = repr(_e)

@unittest.skipUnless(HAVE, "requires GBGPU STFT-GB build + LAT stack")
class SlowFFTSmokeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fx = _build_fixture()          # 6 h segments, alpha=0.0, injected brute STFT = data
    def test_returns_finite_shaped(self):
        gb = _make_gb(self.fx, n_side_bins=5)
        d_h, h_h = get_ll_stft_slowfft_proto(self.fx["grp"], gb, self.fx["params"], n_sub=32)
        self.assertEqual(np.asarray(d_h).shape, (1,))
        self.assertEqual(np.asarray(h_h).shape, (1,))
        self.assertTrue(np.all(np.isfinite(np.asarray(d_h))))
```

If `from tests.test_stft_gb_fft import ...` fails under the runner, copy `_build_fixture`/`_make_gb`/`_tukey` verbatim into this file instead (they are self-contained; ~40 lines).

- [ ] **Step 2: Run test to verify it fails**

Run: `cd gpu_worktrees/GBGPU && ../../.venv-stft-gpu/bin/python -m pytest tests/test_stft_gb_slowfft.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'gbgpu.stft_slowfft_proto'`.

- [ ] **Step 3: Write the stub module**

```python
# src/gbgpu/stft_slowfft_proto.py
"""FastGB-STFT prototype: heterodyned slow part on the STFT sub-grid -> narrow
per-segment DFT -> time-resolved inner product. Phase-1 prototype (design
docs/specs/2026-07-04-stft-gb-slowpart-fft-design.md). CPU/GPU via the `xp` of the
computation group; validated vs the injected brute STFT, benchmarked vs Fresnel."""

def get_ll_stft_slowfft_proto(grp, gb, params, n_sub=32):
    xp = gb.xp
    num_bin = int(params.shape[0])
    d_h = xp.zeros(num_bin, dtype=xp.complex128)
    h_h = xp.zeros(num_bin, dtype=xp.complex128)
    return d_h, h_h  # stub; filled in Tasks 2-4
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd gpu_worktrees/GBGPU && ../../.venv-stft-gpu/bin/python -m pytest tests/test_stft_gb_slowfft.py -q`
Expected: PASS (1 passed).

- [ ] **Step 5: Commit**

```bash
git -C gpu_worktrees/GBGPU add src/gbgpu/stft_slowfft_proto.py tests/test_stft_gb_slowfft.py
git -C gpu_worktrees/GBGPU commit -m "test(stft): FastGB-STFT prototype harness + stub"
```

---

### Task 2: Stage 1 — heterodyned slow part on the STFT sub-grid

**Files:**
- Modify: `src/gbgpu/stft_slowfft_proto.py`
- Test: `tests/test_stft_gb_slowfft.py`

**Interfaces:**
- Produces: `slow_part_on_stft_grid(gb, params, t_seg, dt, n_sub) -> (E, q)` where `E` is xp complex `[num_bin, 3, NT, n_sub]` (heterodyned time-domain slow part, X/Y/Z) and `q[num_bin]` int is the FastGB carrier bin. `t_seg[NT]` are the segment start times, `dt` the STFT segment length (`stft_dt`), `n_sub` the sub-samples/segment.

**Numerics recipe (adapt the FastGB slow part — no FFT):** Build the sub-grid `tm[NT, n_sub] = t_seg[:,None] + (arange(n_sub)+0.5)[None,:]*(dt/n_sub)` (midpoint quadrature, matching `FFTColumn`), flattened to `tm[M]`, `M=NT*n_sub`. Instantiate a `gbgpu.GBGPU(orbits=gb.orbits, ...)` and reproduce `run_wave`'s setup for this `tm`: `Ps = self._spacecraft(tm)` (orbits evaluated **once** on `tm`, reused across binaries), then `Gs, q = self._construct_slow_part(T, arm_length, Ps, tm, f0, fdot, fddot, fstar, phi0, k, DP, DC, eplus, ecross)` exactly as `run_wave` calls it (gbgpu.py:402 area — copy the argument construction verbatim), then the `_computeXYZ` **time-domain** assembly through line 510 (`XYZsl = fctr2[:,None,:] * xp.array([Xsl,Ysl,Zsl]).transpose(1,0,2)`) — **stop before the `xp.fft.fft` on line 513.** Reshape `XYZsl[num_bin,3,M] -> E[num_bin,3,NT,n_sub]`. `E` is heterodyned against `q/T` (the `(om-df)*tm` term in `_construct_slow_part` argS, line 631) and therefore smooth. Do NOT use `GBTDIonTheFly` / `get_tdi_Xf_single`.

- [ ] **Step 1: Write the failing test (Stage 1 reproduces the brute TD signal on the sub-grid)**

```python
class SlowPartStage1Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fx = _build_fixture()
    def test_slow_part_reconstructs_td_signal(self):
        from gbgpu.stft_slowfft_proto import slow_part_on_stft_grid
        gb = _make_gb(self.fx, n_side_bins=20)
        # Rebuild the SAME brute TD signal the fixture STFT'd, to compare against.
        # (fixture uses dt=10, stft_dt=6*3600, n_stft=256; see _build_fixture)
        dt = 10.0; stft_dt = 6*3600.0; n_stft = 256; nps = int(stft_dt/dt)
        t_seg = np.arange(n_stft) * stft_dt
        n_sub = 64
        E, q = slow_part_on_stft_grid(gb, self.fx["params"], t_seg, stft_dt, n_sub)
        self.assertEqual(np.asarray(E).shape, (1, 3, n_stft, n_sub))
        # Reconstruct s(tau) = Re[ E * exp(2pi i (q/T) tau) ] at the sub-grid and
        # compare to the brute GBTDIonTheFly TD signal at the same tau (mid-segment
        # sample). Full parity check lives in Task 3 (template vs brute STFT); here
        # assert E is finite, non-trivial, and smooth across sub-samples.
        E = np.asarray(E)
        self.assertTrue(np.all(np.isfinite(E)))
        self.assertGreater(np.abs(E).max(), 0.0)
        # smoothness: successive sub-sample differences are small vs the values
        d = np.abs(np.diff(E, axis=-1)); v = np.abs(E[..., :-1]) + 1e-300
        self.assertLess(np.median(d / v), 0.5)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd gpu_worktrees/GBGPU && ../../.venv-stft-gpu/bin/python -m pytest tests/test_stft_gb_slowfft.py::SlowPartStage1Test -q`
Expected: FAIL — `ImportError: cannot import name 'slow_part_on_stft_grid'`.

- [ ] **Step 3: Implement `slow_part_on_stft_grid`** following the numerics recipe above. Copy the exact argument construction from `run_wave`/`_computeXYZ` (gbgpu.py:164-513) for the `tm` sub-grid; return `E[num_bin,3,NT,n_sub]` and `q`.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd gpu_worktrees/GBGPU && ../../.venv-stft-gpu/bin/python -m pytest tests/test_stft_gb_slowfft.py::SlowPartStage1Test -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git -C gpu_worktrees/GBGPU add -u && git -C gpu_worktrees/GBGPU commit -m "feat(stft): FastGB-STFT Stage 1 — heterodyned slow part on the STFT sub-grid"
```

---

### Task 3: Stage 2 — narrow per-segment DFT → STFT template columns

**Files:**
- Modify: `src/gbgpu/stft_slowfft_proto.py`
- Test: `tests/test_stft_gb_slowfft.py`

**Interfaces:**
- Produces: `stft_template_from_slow_part(gb, E, q, t_seg, dt, n_sub, n_side, settings) -> H` where `H` is an xp complex STFT template `[num_bin, 3, NT, num_freqs]` laid out on the group's frequency grid (zeros outside the `2·n_side+1` band around each segment's carrier bin). `settings` is the STFT settings from the group (`f_min`, `df`, `num_freqs`).

**Numerics recipe (mirror `FFTColumn` on the precomputed `E`):** For each segment, apply the analysis window per sub-sample (`_tukey`-equivalent, `taper_duration = alpha*dt/2`), then the same targeted twiddle-DFT `FFTColumn` uses (`lat_stft_kernels.hh:322-360`): pick the carrier bin `freq_j = get_freq_index(f0_seg)`, re-heterodyne `E` from the global `q/T` to `freq_j·df` (residual drift, one phase ramp), and DFT the `n_sub` samples to `diff ∈ [-n_side, +n_side]` via `bin = 0.5*dts_sub * base^diff * Σ_m demod[m] (W^diff)^m`, `W=exp(-2πi/n_sub)`, `base=exp(-2πi(df*t_seg + 0.5/n_sub))`. Scatter into `H[:, :, seg, freq_j+diff]`.

- [ ] **Step 1: Write the failing test (template matches the brute STFT per-pixel, converging with n_sub)**

```python
class SlowPartStage2Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fx = _build_fixture()   # data_res holds the brute STFT of the injection
    def test_template_matches_brute_stft(self):
        from gbgpu.stft_slowfft_proto import slow_part_on_stft_grid, stft_template_from_slow_part
        gb = _make_gb(self.fx, n_side_bins=20)
        dt = 10.0; stft_dt = 6*3600.0; n_stft = 256
        t_seg = np.arange(n_stft) * stft_dt
        settings = self.fx["grp"].settings
        def band_mm(n_sub):
            E, q = slow_part_on_stft_grid(gb, self.fx["params"], t_seg, stft_dt, n_sub)
            H = np.asarray(stft_template_from_slow_part(gb, E, q, t_seg, stft_dt, n_sub, 20, settings))[0]
            D = np.asarray(self.fx["grp"].get_stft_data_array())   # brute STFT [3, NT, num_freqs]
            m = np.abs(D) > 0
            num = np.vdot(D[m], H[m]); den = np.sqrt(np.vdot(D[m],D[m]) * np.vdot(H[m],H[m]))
            return abs(1.0 - (num/den).real)
        mm32, mm64 = band_mm(32), band_mm(64)
        print(f"\n[stage2] template-vs-brute mm: n_sub=32 {mm32:.3e}  n_sub=64 {mm64:.3e}")
        self.assertLess(mm64, mm32)      # converges with n_sub
        self.assertLess(mm64, 1e-3)      # matches the brute STFT template
```

Note: if `grp.get_stft_data_array()` is not the exact accessor, read `STFTComputationGroup` in `lisatools/domaincomputation.py` for the data-STFT accessor (`self.data_arr`/`linear_data_arr`) and use it; the assertion (converging band mismatch vs the brute STFT) is the contract.

- [ ] **Step 2: Run test to verify it fails**

Run: `cd gpu_worktrees/GBGPU && ../../.venv-stft-gpu/bin/python -m pytest tests/test_stft_gb_slowfft.py::SlowPartStage2Test -q`
Expected: FAIL — `ImportError: cannot import name 'stft_template_from_slow_part'`.

- [ ] **Step 3: Implement `stft_template_from_slow_part`** per the recipe (port `FFTColumn`'s window + twiddle-DFT, `lat_stft_kernels.hh:258-360`, operating on the precomputed `E` instead of `get_tdi_Xf_single_cached`).

- [ ] **Step 4: Run test to verify it passes**

Run: `cd gpu_worktrees/GBGPU && ../../.venv-stft-gpu/bin/python -m pytest tests/test_stft_gb_slowfft.py::SlowPartStage2Test -q`
Expected: PASS, `n_sub=64` mm `< 1e-3` and `< n_sub=32`.

- [ ] **Step 5: Commit**

```bash
git -C gpu_worktrees/GBGPU add -u && git -C gpu_worktrees/GBGPU commit -m "feat(stft): FastGB-STFT Stage 2 — narrow per-segment DFT template"
```

---

### Task 4: Inner product + recovery (fill the prototype end-to-end)

**Files:**
- Modify: `src/gbgpu/stft_slowfft_proto.py`
- Test: `tests/test_stft_gb_slowfft.py`

**Interfaces:**
- Produces: fills `get_ll_stft_slowfft_proto(grp, gb, params, n_sub)` to return the real `(d_h, h_h)` by scoring `H` (Task 3) through the group's inner-product machinery: `grp.compute_signal_likelihood_terms(H, ...)` (see the LAT note `lat_stft_kernels.hh:648` and `STFTGBComputations.fill_global_stft`:762 for how a produced template is scored). If a direct `compute_signal_likelihood_terms` entry is not exposed, contract manually: `d_h = 4*df*Σ conj(H) · invC · D`, `h_h = 4*df*Σ conj(H) · invC · H`, reading `invC` (`grp.invC_arr`) and the data STFT from the group.

- [ ] **Step 1: Write the failing recovery + convergence test**

```python
class SlowPartRecoveryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fx = _build_fixture()
    def test_recovers_injection_and_converges(self):
        gb = _make_gb(self.fx, n_side_bins=20)
        d_d = self.fx["d_d"]; mms = []
        for n_sub in (32, 64, 96):
            d_h, h_h = get_ll_stft_slowfft_proto(self.fx["grp"], gb, self.fx["params"], n_sub=n_sub)
            d_h = complex(np.asarray(d_h).reshape(-1)[0]); h_h = float(np.asarray(h_h).reshape(-1)[0].real)
            mms.append(abs(1.0 - d_h.real / np.sqrt(d_d * h_h)))
            print(f"[slowfft-recovery] n_sub={n_sub:3d} mm={mms[-1]:.3e}")
        self.assertLess(mms[-1], 1e-2)       # recovers the injection
        self.assertLessEqual(mms[-1], mms[0]) # converges with n_sub
    def test_matches_fresnel_short_segments(self):
        gb = _make_gb(self.fx, n_side_bins=20)
        gb.get_ll_stft(self.fx["params"])
        d_h_f = complex(np.asarray(gb.d_h_out).reshape(-1)[0]); h_h_f = float(np.asarray(gb.h_h_out).reshape(-1)[0].real)
        mm_f = abs(1.0 - d_h_f.real / np.sqrt(self.fx["d_d"] * h_h_f))
        d_h, h_h = get_ll_stft_slowfft_proto(self.fx["grp"], gb, self.fx["params"], n_sub=64)
        d_h = complex(np.asarray(d_h).reshape(-1)[0]); h_h = float(np.asarray(h_h).reshape(-1)[0].real)
        mm_x = abs(1.0 - d_h.real / np.sqrt(self.fx["d_d"] * h_h))
        print(f"[slowfft-vs-fresnel 6h] fresnel {mm_f:.3e}  slowfft {mm_x:.3e}")
        self.assertLessEqual(mm_x, mm_f * 1.20 + 1e-4)   # at least Fresnel accuracy at 6 h
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd gpu_worktrees/GBGPU && ../../.venv-stft-gpu/bin/python -m pytest tests/test_stft_gb_slowfft.py::SlowPartRecoveryTest -q`
Expected: FAIL — stub returns zeros → `mm` is nan/1.0, assertion fails.

- [ ] **Step 3: Implement the inner-product assembly** in `get_ll_stft_slowfft_proto` (call Stage 1 → Stage 2 → score `H`). Handle the `d_d`/normalization exactly as `get_ll_stft` does (gbcomps.py:564-591).

- [ ] **Step 4: Run test to verify it passes**

Run: `cd gpu_worktrees/GBGPU && ../../.venv-stft-gpu/bin/python -m pytest tests/test_stft_gb_slowfft.py::SlowPartRecoveryTest -q`
Expected: PASS — recovery `mm < 1e-2`, converges, ≥ Fresnel at 6 h.

- [ ] **Step 5: Commit**

```bash
git -C gpu_worktrees/GBGPU add -u && git -C gpu_worktrees/GBGPU commit -m "feat(stft): FastGB-STFT prototype end-to-end get_ll (recovers injection)"
```

---

### Task 5: Large-Δ accuracy — the value proposition

**Files:**
- Test: `tests/test_stft_gb_slowfft.py`

**Interfaces:** Consumes Task 4's `get_ll_stft_slowfft_proto` and Fresnel `get_ll_stft`. This task adds only tests + a parametrized fixture builder for Δ ∈ {1 d, 4 d} with the ~1e-4 Hz taper.

- [ ] **Step 1: Write the large-Δ test (slowfft beats Fresnel where the constant-envelope error bites)**

```python
def _build_fixture_dt(stft_hours, taper_s):
    """_build_fixture variant: segment = stft_hours, Tukey alpha set so the taper
    duration ~ taper_s per side (alpha = 2*taper_s/stft_dt), fixed ~1e-4 Hz taper."""
    # copy _build_fixture, set stft_dt = stft_hours*3600, n_stft so Tobs ~ 1 yr,
    # alpha = min(1.0, 2.0*taper_s/stft_dt); window = _tukey(nperseg, alpha).
    ...  # implement by adapting _build_fixture (tests/test_stft_gb_fft.py:45)

class SlowPartLargeDeltaTest(unittest.TestCase):
    def test_beats_fresnel_at_1day_and_4day(self):
        for hours in (24.0, 96.0):
            fx = _build_fixture_dt(hours, taper_s=1e4)
            gb = _make_gb(fx, n_side_bins=20)
            gb.get_ll_stft(fx["params"])
            d_h_f = complex(np.asarray(gb.d_h_out).reshape(-1)[0]); h_h_f = float(np.asarray(gb.h_h_out).reshape(-1)[0].real)
            mm_f = abs(1.0 - d_h_f.real / np.sqrt(fx["d_d"] * h_h_f))
            d_h, h_h = get_ll_stft_slowfft_proto(fx["grp"], gb, fx["params"], n_sub=96)
            d_h = complex(np.asarray(d_h).reshape(-1)[0]); h_h = float(np.asarray(h_h).reshape(-1)[0].real)
            mm_x = abs(1.0 - d_h.real / np.sqrt(fx["d_d"] * h_h))
            print(f"[large-dt {hours:.0f}h] fresnel mm={mm_f:.3e}  slowfft mm={mm_x:.3e}")
            self.assertLessEqual(mm_x, mm_f + 1e-6)   # exact >= Fresnel; expect strictly better at 4 d
```

- [ ] **Step 2: Run to verify it fails** (until `_build_fixture_dt` is implemented)

Run: `cd gpu_worktrees/GBGPU && ../../.venv-stft-gpu/bin/python -m pytest tests/test_stft_gb_slowfft.py::SlowPartLargeDeltaTest -q`
Expected: FAIL (NameError/`_build_fixture_dt` incomplete).

- [ ] **Step 3: Implement `_build_fixture_dt`** by adapting `_build_fixture` (parametrize `stft_dt`, `n_stft` for ~1 yr, and `alpha` from `taper_s`).

- [ ] **Step 4: Run to verify it passes**

Run: `cd gpu_worktrees/GBGPU && ../../.venv-stft-gpu/bin/python -m pytest tests/test_stft_gb_slowfft.py::SlowPartLargeDeltaTest -q`
Expected: PASS — `slowfft mm ≤ Fresnel mm` at 24 h and 96 h (strictly lower at 96 h, where Fresnel's constant-envelope error is largest). If it does NOT beat Fresnel, STOP and re-examine the heterodyne convention (§7 of the spec) before proceeding.

- [ ] **Step 5: Commit**

```bash
git -C gpu_worktrees/GBGPU add -u && git -C gpu_worktrees/GBGPU commit -m "test(stft): FastGB-STFT large-Delta accuracy beats Fresnel (constant-envelope win)"
```

---

### Task 6: GPU wall-clock benchmark — the Phase-1 go/no-go

**Files:**
- Create: `scripts/bench_stft_slowfft.py`

**Interfaces:** Consumes `get_ll_stft_slowfft_proto` and Fresnel `get_ll_stft`; times both on GPU at production configs and prints the FFT-analog ratio (slowfft/fresnel wall-clock).

- [ ] **Step 1: Write the benchmark script** by adapting `scripts/bench_stft_fft_vs_fresnel.py` (same fixture builder + backend resolution + `time_call` min-of-repeats). Sweep `num_bin ∈ {1024, 4096, 16384}`, `stft_hours ∈ {8, 24, 96}`, `n_side=10`, `n_sub ∈ {24, 32}`; for each, time `get_ll_stft(params)` (Fresnel) vs `get_ll_stft_slowfft_proto(grp, gb, params, n_sub)`; print `ratio = t_slowfft / t_fresnel` and write CSV. Gate on recovery `mm` before timing (mirror the FFT bench's built-in gate).

- [ ] **Step 2: Smoke it on GPU**

Run: `CUDA_VISIBLE_DEVICES=<idle> bash $CLAUDE_JOB_DIR/tmp/gpurun.sh scripts/bench_stft_slowfft.py --smoke`
Expected: prints backend `gpu`, gate `mm` ~1e-3 or better, and a ratio line; completes without error.

- [ ] **Step 3: Full sweep on an idle GPU**

Run: `CUDA_VISIBLE_DEVICES=<idle> bash $CLAUDE_JOB_DIR/tmp/gpurun.sh scripts/bench_stft_slowfft.py --csv scripts/bench_stft_slowfft_A100.csv`
Expected: 21+ rows; a printed `slowfft/fresnel ratio: best … / worst …`.

- [ ] **Step 4: Record the verdict**

Write `scripts/bench_slowfft_results_A100.md` (mirror `bench_results_A100.md`): ratio table vs Fresnel, accuracy (recovery mm at each Δ), the go/no-go call (GO if slowfft ≤ Fresnel wall-clock at num_bin=16384 while `mm ≤ Fresnel`), and — if GO — the Phase-2 trigger (fuse Stage 1 into CUDA, occupancy check, then the full surface).

- [ ] **Step 5: Commit**

```bash
git -C gpu_worktrees/GBGPU add scripts/bench_stft_slowfft.py scripts/bench_stft_slowfft_A100.csv scripts/bench_slowfft_results_A100.md
git -C gpu_worktrees/GBGPU commit -m "bench(stft): FastGB-STFT Phase-1 go/no-go vs Fresnel (A100)"
```

---

## Self-Review

**Spec coverage:** §2 core idea → Tasks 2–4; §3 "beats FFTColumn" (no orbit cache) → prototype uses `_construct_slow_part` not `get_tdi_Xf_single_cached` (Task 2 recipe); §3 "beats Fresnel / exact" → Task 5 (large-Δ) + Task 6 (wall-clock); §4 Stage 1/2 → Tasks 2/3; §4 integration → Task 4; §6 accuracy plan → Tasks 4/5 (recovery mm, convergence, cross-checks); §7 memory risk → cupy prototype chunk note (Task 6 can add `--chunk`); §7 wall-clock hypothesis → Task 6 go/no-go; §7 heterodyne convention → Task 5 Step 4 stop-condition; §8 phasing → Task 6 records the Phase-2 trigger. Phases 2–3 are out of scope (separate plans, gated on Task 6).

**Placeholder scan:** the numerics in Tasks 2–4 are given as precise source-pointer recipes (exact functions + line ranges to adapt) plus full test code that defines correctness (TDD); the `_build_fixture_dt`/`get_stft_data_array` accessors are flagged with the exact reference to confirm during execution. No vague "add error handling".

**Type consistency:** `get_ll_stft_slowfft_proto(grp, gb, params, n_sub) -> (d_h, h_h)` used identically in Tasks 1, 4, 5, 6; `slow_part_on_stft_grid(...) -> (E[num_bin,3,NT,n_sub], q)` consumed by `stft_template_from_slow_part(...) -> H[num_bin,3,NT,num_freqs]` consumed by Task 4 — consistent shapes throughout.
