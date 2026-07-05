"""GPU Phase-1 go/no-go: FastGB-STFT slow-FFT prototype vs analytic Fresnel STFT.

Times the ``get_ll_stft_slowfft_proto`` prototype (batched Stage-1 heterodyned slow
part on the STFT sub-grid + a Python Stage-2 per-segment DFT loop + XYZ contraction)
against the fused C++ analytic Fresnel ``STFTGBComputations.get_ll_stft`` on the same
injected data, sweeping ``num_bin`` x ``stft_hours`` x ``n_sub`` at ``n_side=10``.

This is the Phase-1 decision benchmark for the design
``docs/specs/2026-07-04-stft-gb-slowpart-fft-design.md``. Accuracy already decisively
favors slow-FFT (Tasks 4/5: recovers the injection to mm=7.24e-4 @6h, and stays EXACT
at large Delta -- 7.05e-6 @24h, 3.21e-4 @96h -- where the production Fresnel/FFT C++
kernels are numerically DEGENERATE, nan/overflow). The only open question here is
WALL-CLOCK, and specifically how much of the prototype's time is the removable Python
Stage-2 DFT loop (a fused CUDA kernel eliminates it) vs the batched Stage-1 physics.

So each slow-FFT time is reported as a SPLIT:
  * ``stage1``      -- ``slow_part_on_stft_grid`` (batched cupy; the fused-kernel-relevant
                       work; includes the per-call private-GBGPU construction, a prototype
                       artifact amortized at large num_bin);
  * ``stage2+``     -- everything after Stage 1 (the Python per-binary DFT template loop +
                       XYZ inner-product contraction) = ``full - stage1``.

Realities handled honestly:
  * The prototype is cupy, NOT a fused kernel -- it is expected to be slower than the fused
    Fresnel; that alone is NOT a NO-GO for the Phase-2 fused CUDA kernel (hence the split).
  * Memory: Stage-2 materializes ``H[num_bin, 3, NT, NF]`` (NF ~ 125 -> the real driver,
    bigger than Stage-1's ``E[num_bin, 3, NT, n_sub]``). If a config OOMs, binaries are
    processed in chunks (block halved until it fits) and the config is flagged CHUNKED.
    The largest num_bin that fits un-chunked is reported.
  * Fresnel is numerically DEGENERATE at Delta >= 24 h (Task 5): its kernel still RUNS (so
    wall-clock is measurable) but it is NOT a correct alternative there. Rows/summary at
    Delta >= 24 h are flagged ``fresnel_degenerate`` -- the ratio is a raw wall-clock ratio,
    NOT an accuracy-matched comparison.
  * "Two live STFT groups per process" is UB (Task 5). Each ``stft_hours`` is therefore run
    in its OWN subprocess (this script re-execs itself once per Delta) so every fixture is
    single-live-instance; the parent aggregates the child CSVs.
  * Per fixture, Fresnel (gate + all timing) runs BEFORE any slow-FFT, because Stage 1
    builds a private ``GBGPU`` that clobbers the live group's C++ STFTDomain (slow-FFT itself
    is immune -- it scores numpy snapshots of ``data_arr``/``invC_arr``).

Usage:
    # GPU box, full sweep (auto backend -> CUDA), one subprocess per stft_hours:
    CUDA_VISIBLE_DEVICES=<idle> bash .superpowers/sdd/gpurun.sh \
        scripts/bench_stft_slowfft.py --csv scripts/bench_stft_slowfft_A100.csv
    # quick self-check (single fixture, tiny, still GPU):
    CUDA_VISIBLE_DEVICES=<idle> bash .superpowers/sdd/gpurun.sh \
        scripts/bench_stft_slowfft.py --smoke
"""
import argparse
import csv
import os
import subprocess
import sys
import tempfile
import time

import numpy as np

from lisatools.detector import DefaultOrbits
from lisatools.response.tdiconfig import TDIConfig
from lisatools.response.tdionfly import GBTDIonTheFly
from lisatools.domains import TDSignal, TDSettings, get_stft_settings
from lisatools.domaincomputation import STFTComputationGroup
from lisatools.sensitivity import XYZSensitivityBackend
from lisatools.analysiscontainer import AnalysisContainer, AnalysisContainerArray
from gbgpu.gbcomps import STFTGBComputations
from gbgpu.stft_slowfft_proto import (
    get_ll_stft_slowfft_proto, slow_part_on_stft_grid, make_slowpart_gbgpu)

AMP, F0, FDOT, FDDOT = 1e-23, 4.2300812341e-3, 1e-18, 0.0
PHI0, INC, PSI, LAM, BETA = 0.892342342342, 1.2309804223, 3.00908098, 4.827342308, -0.50923423

N_SUB_MAX = 64      # STFT_FFT_NSUB_MAX (lat_stft_kernels.hh); slow-FFT has no such cap but
                    # we keep the comparison inside the Fresnel-relevant sub-grid regime.

CSV_FIELDS = ["stft_hours", "num_bin", "n_side", "n_sub", "kernel", "ms", "stage1_ms",
              "stage2_ms", "us_per_bin", "ratio", "stage1_ratio", "mm",
              "fresnel_degenerate", "chunked", "block", "construct_ms"]


def _tukey(n, alpha):
    """scipy.signal.windows.tukey(n, alpha), numpy-only (matches the C++ taper)."""
    if alpha <= 0.0:
        return np.ones(n)
    w = np.ones(n)
    t = int(np.floor(alpha * (n - 1) / 2.0))
    k = np.arange(0, t + 1)
    ramp = 0.5 * (1.0 + np.cos(np.pi * (2.0 * k / (alpha * (n - 1)) - 1.0)))
    w[:t + 1] = ramp
    w[-(t + 1):] = ramp[::-1]
    return w


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--backend", default="auto",
                   help="force_backend name (cpu, cuda11x/12x/13x, cuda, gpu); "
                        "'auto' lets the framework pick the best available")
    p.add_argument("--dt", type=float, default=10.0)
    p.add_argument("--stft-hours", default="8,24,96",
                   help="comma list of STFT segment lengths (hours) to sweep; each runs "
                        "in its own subprocess (single-live-group isolation)")
    p.add_argument("--tobs-days", type=float, default=91.0,
                   help="observation span (rounded down to whole segments)")
    p.add_argument("--alpha", type=float, default=0.1, help="Tukey analysis-window alpha")
    p.add_argument("--num-bins", default="1024,4096,16384",
                   help="comma list of binary counts to time")
    p.add_argument("--n-side", default="10", help="comma list of n_side_bins")
    p.add_argument("--n-sub", default="24,32", help="comma list of slow-FFT n_sub")
    p.add_argument("--spread-bins", type=float, default=32.0,
                   help="benchmark sources drawn uniformly in f0 = kc +- spread bins")
    p.add_argument("--repeats", type=int, default=3, help="timed calls per config (min reported)")
    p.add_argument("--n-sparse", type=int, default=2048,
                   help="ToF-spline knots for the injected oracle (the recovery gate ONLY -- "
                        "does NOT affect any timed kernel). Capped at ~2800 on GPU: the "
                        "GBTDIonTheFly generation kernel requests get_gb_buffer_size(n_sparse) "
                        "dynamic shared mem, and >~164KB (n_sparse>~2800) exceeds the A100 "
                        "per-block opt-in cap -> 'invalid argument' at gb_tdi_on_the_fly.cu:187. "
                        "Authoritative accuracy is Tasks 4/5 (8192-knot CPU oracle).")
    p.add_argument("--min-block", type=int, default=256,
                   help="smallest binary chunk before giving up on OOM")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--csv", default=None, help="write result rows to this CSV path")
    p.add_argument("--smoke", action="store_true",
                   help="tiny single-fixture config to validate the script on GPU")
    p.add_argument("--_child", action="store_true",
                   help="internal: run a single stft_hours fixture (subprocess worker)")
    a = p.parse_args(argv)
    if a.smoke:
        a.tobs_days, a.num_bins, a.n_sub = 8.0, "256", "24,32"
        a.stft_hours, a.repeats, a.n_sparse = "8", 2, 2048
    a.num_bins = [int(x) for x in str(a.num_bins).split(",")]
    a.n_side = [int(x) for x in str(a.n_side).split(",")]
    a.n_sub = [int(x) for x in str(a.n_sub).split(",")]
    a.stft_hours = [float(x) for x in str(a.stft_hours).split(",")]
    return a


# ---------------------------------------------------------------------------
# backend + small helpers
# ---------------------------------------------------------------------------
def resolve_backend(name):
    if name == "auto":
        for fb in ("gpu", "cpu"):
            try:
                orbits = DefaultOrbits(force_backend=fb)
                break
            except Exception:
                continue
        else:
            raise RuntimeError("no usable backend (tried 'gpu', 'cpu')")
    else:
        fb = name
        orbits = DefaultOrbits(force_backend=fb)
    orbits.configure(linear_interp_setup=True)
    return fb, orbits


def _scalar(xp, arr):
    return complex(xp.asarray(arr).reshape(-1)[0])


def _mism(xp, d_h, h_h, d_d):
    """Recovery mismatch |1 - Re<d|h>/sqrt(<d|d><h|h>)|; nan-guarded (Fresnel degenerate)."""
    hr = _scalar(xp, h_h).real
    if not np.isfinite(hr) or hr <= 0.0:
        return float("nan")
    return abs(1.0 - _scalar(xp, d_h).real / np.sqrt(d_d * hr))


def _as_float(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


# ---------------------------------------------------------------------------
# fixture (adapted verbatim from bench_stft_fft_vs_fresnel.py, + N_sparse + stft_hours)
# ---------------------------------------------------------------------------
def build_fixture(a, stft_hours, fb, orbits, xp, on_gpu):
    """Inject one GB, STFT it with the Tukey analysis window, build the group + d_d."""
    tdi_config = TDIConfig("2nd generation", force_backend=fb)
    stft_dt = stft_hours * 3600.0
    nperseg = int(round(stft_dt / a.dt))
    n_stft = int(a.tobs_days * 86400.0 // stft_dt)
    nobs = n_stft * nperseg
    Tobs = nobs * a.dt
    df = 1.0 / stft_dt
    kc = int(round(F0 / df))
    band = int(np.ceil(a.spread_bins)) + max(a.n_side) + 20

    data_t = xp.arange(nobs) * a.dt
    t_tdi = xp.linspace(0.0, Tobs, a.n_sparse + 1)[1:-1]          # sharp oracle
    settings = get_stft_settings(data_t, stft_dt, min_freq=(kc - band) * df,
                                 max_freq=(kc + band) * df, force_backend=fb)
    gen = GBTDIonTheFly(t_tdi, Tobs, 0.0, 1.0 / a.dt, 1, tdi_config=tdi_config,
                        orbits=orbits, tdi_chan="XYZ", force_backend=fb)
    out = gen(AMP, F0, FDOT, FDDOT, PHI0, INC, PSI, LAM, BETA,
              convert_to_ra_dec=False, return_spline=True)
    keep = (data_t > t_tdi[0]) & (data_t < t_tdi[-1])
    tdi_ts = xp.zeros((1, 3, nobs))
    tdi_ts[:, :, keep] = out.eval_tdi(data_t[keep])
    td_signal = TDSignal(tdi_ts[0], settings=TDSettings(nobs, a.dt, 0.0, force_backend=fb))
    stft_signal = td_signal.stft(window=xp.asarray(_tukey(nperseg, a.alpha)), settings=settings)
    sens = XYZSensitivityBackend(orbits=orbits, settings=settings, force_backend=fb)
    sens.sens_mat = sens.compute_sensitivity_matrix(sens.basis_settings.f_arr, 15e-12, 3e-15)
    ac = AnalysisContainer(stft_signal, sens)
    grp = STFTComputationGroup(AnalysisContainerArray([ac], gpus=[0] if on_gpu else None),
                               split_index=0, window_alpha=a.alpha, force_backend=fb)
    grp.compute_d_d_term()
    d_d = float(xp.asarray(grp.d_d).reshape(-1)[0].real)
    return dict(tdi_config=tdi_config, grp=grp, d_d=d_d, Tobs=Tobs, settings=settings,
                stft_dt=stft_dt, df=df, kc=kc, band=band, NT=int(settings.NT),
                NF=int(settings.NF_active), nperseg=nperseg, nobs=nobs, n_stft=n_stft)


def draw_params(a, kc, df, nb):
    rng = np.random.default_rng(a.seed + nb)     # host numpy; slow-FFT wants host params
    f0 = (kc + rng.uniform(-a.spread_bins, a.spread_bins, nb)) * df
    return np.column_stack([
        np.full(nb, AMP), f0, FDOT * rng.uniform(0.5, 2.0, nb), np.zeros(nb),
        rng.uniform(0.0, 2 * np.pi, nb), np.arccos(rng.uniform(-1.0, 1.0, nb)),
        rng.uniform(0.0, np.pi, nb), rng.uniform(0.0, 2 * np.pi, nb),
        np.arcsin(rng.uniform(-1.0, 1.0, nb)),
    ])


# ---------------------------------------------------------------------------
# chunked slow-FFT (memory guard) + Stage-1-only helpers
# ---------------------------------------------------------------------------
def run_slowfft_chunked(grp, gb, params, n_sub, block, xp, on_gpu, gbtmp):
    """Full prototype get_ll over binaries in blocks of ``block``; concat per-binary d_h/h_h.

    d_h/h_h are per-binary (each source is an independent template vs the SHARED data), so
    blocks CONCATENATE (not sum). block >= num_bin -> a single un-chunked call. ``gbtmp`` is
    the prebuilt private GBGPU reused across calls (one-time construction hoisted out).
    """
    nb = params.shape[0]
    dhs, hhs = [], []
    for lo in range(0, nb, block):
        d_h, h_h = get_ll_stft_slowfft_proto(grp, gb, params[lo:lo + block], n_sub, gbtmp=gbtmp)
        dhs.append(d_h)
        hhs.append(h_h)
        if on_gpu:
            xp.get_default_memory_pool().free_all_blocks()
    return xp.concatenate(dhs), xp.concatenate(hhs)


def run_stage1_chunked(gb, params, t_seg, stft_dt, n_sub, block, xp, on_gpu, gbtmp):
    """Stage 1 (slow_part_on_stft_grid) only, in blocks; discard E (timing only).

    ``gbtmp`` reused (construction hoisted) so this measures Stage-1 PHYSICS, not the
    ~seconds per-call orbit/GBGPU build.
    """
    nb = params.shape[0]
    for lo in range(0, nb, block):
        E, q = slow_part_on_stft_grid(gb, params[lo:lo + block], t_seg, stft_dt, n_sub, gbtmp=gbtmp)
        del E, q
        if on_gpu:
            xp.get_default_memory_pool().free_all_blocks()


def resolve_block(make_fn, nb, min_block, oom, sync, free):
    """Find the largest binary chunk that runs without OOM (also warms the path).

    Returns the working block. block == nb -> the config fits un-chunked. On OOM the block
    is halved (down to min_block, then re-raise). LOUDLY logs any chunking.
    """
    block = nb
    while True:
        try:
            out = make_fn(block)()
            del out
            free()
            sync()
            return block
        except oom:
            free()
            if block <= min_block:
                print(f"   !! OOM even at block={block} (num_bin={nb}); giving up on this config",
                      flush=True)
                raise
            nblock = max(min_block, block // 2)
            print(f"   !! OOM at block={block} -> retry block={nblock} (num_bin={nb}) "
                  f"[CHUNKED -- prototype-only memory limit]", flush=True)
            block = nblock


def time_call(fn, repeats, sync, warmup=True):
    if warmup:
        fn(); sync()
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter(); fn(); sync(); ts.append(time.perf_counter() - t0)
    return min(ts)


# ---------------------------------------------------------------------------
# single-fixture worker (one stft_hours)
# ---------------------------------------------------------------------------
def run_single(a, stft_hours):
    fb, orbits = resolve_backend(a.backend)
    xp = orbits.xp
    on_gpu = xp.__name__ == "cupy"
    oom = (MemoryError,) + ((xp.cuda.memory.OutOfMemoryError,) if on_gpu else ())

    def sync():
        if on_gpu:
            xp.cuda.runtime.deviceSynchronize()

    def free():
        if on_gpu:
            xp.get_default_memory_pool().free_all_blocks()

    dev = (f" [{xp.cuda.runtime.getDeviceProperties(0)['name'].decode()}]" if on_gpu else "")
    print(f"backend: {fb} -> xp={xp.__name__}{dev}   stft_hours={stft_hours:g}", flush=True)

    fx = build_fixture(a, stft_hours, fb, orbits, xp, on_gpu)
    grp, d_d, Tobs, settings = fx["grp"], fx["d_d"], fx["Tobs"], fx["settings"]
    NT, NF, stft_dt, df = fx["NT"], fx["NF"], fx["stft_dt"], fx["df"]
    t0 = float(settings.t0)
    t_seg = t0 + xp.arange(NT) * stft_dt                    # matches the prototype's grid
    p_inj = np.array([[AMP, F0, FDOT, FDDOT, PHI0, INC, PSI, LAM, BETA]])
    degenerate = stft_hours >= 24.0                         # Fresnel unusable there (Task 5)

    e_mb = 3 * NT * max(a.n_sub) * 16 / 1e6                 # per-binary E (n_sub) [MB]
    h_mb = 3 * NT * NF * 16 / 1e6                           # per-binary H (NF)   [MB]
    print(f"Tobs={Tobs:.3e}s ({Tobs/86400:.1f}d)  NT={NT} x {stft_hours:g}h  NF={NF}  "
          f"band=kc={fx['kc']}+-{fx['band']}  alpha={a.alpha}  N_sparse={a.n_sparse}", flush=True)
    print(f"mem/binary: E~{e_mb:.2f}MB(n_sub={max(a.n_sub)}) H~{h_mb:.2f}MB(NF={NF}); "
          f"H is the driver -> num_bin=16384 needs ~{16384*h_mb/1e3:.1f}GB just for H "
          f"(x2-3 for conj/einsum) => chunking expected at 8h/16384.", flush=True)
    if degenerate:
        print(f"NOTE: Fresnel is numerically DEGENERATE at {stft_hours:g}h (Task 5) -- its "
              f"kernel still runs (wall-clock valid) but is NOT accuracy-matched here.", flush=True)

    bench_params = {nb: draw_params(a, fx["kc"], df, nb) for nb in a.num_bins}
    valid_n_sub = [n for n in a.n_sub if (2 * max(a.n_side) + 1) <= n <= N_SUB_MAX]

    rows = []
    # One fixture/group is reused across n_side, but Stage-1's private GBGPU clobbers the
    # live C++ group after the first pass -- a 2nd n_side would then score Fresnel on a
    # clobbered group. Rebuild the fixture per n_side to lift this (Phase-2 item).
    assert len(a.n_side) == 1, (
        "bench reuses one group; multiple n_side would score Fresnel on a clobbered "
        "group -- rebuild the fixture per n_side to support >1")
    for ns in a.n_side:
        gb = STFTGBComputations(stft_comps=grp, T=Tobs, t_ref=0.0, orbits=orbits,
                                tdi_config=fx["tdi_config"], force_backend=fb, n_side_bins=ns,
                                window_factor=1.0, freq_from_tdi_phase=True)

        # ---- Fresnel gate + ALL Fresnel timing FIRST (clean group, before any slow-FFT) ----
        gb.get_ll_stft(p_inj)
        mm_f = _mism(xp, gb.d_h_out, gb.h_h_out, d_d)
        tag_f = "(DEGENERATE -- evidence only)" if degenerate else ""
        print(f"\n== n_side={ns} (need n_sub>={2*ns+1})  fresnel gate mm={mm_f:.3e} {tag_f} ==",
              flush=True)
        t_fres = {}
        for nb in a.num_bins:
            t_fres[nb] = time_call(lambda: gb.get_ll_stft(bench_params[nb]), a.repeats, sync)
            print(f"   fresnel num_bin={nb:6d}: {1e3*t_fres[nb]:9.2f} ms", flush=True)
            rows.append(dict(stft_hours=stft_hours, num_bin=nb, n_side=ns, n_sub="",
                             kernel="fresnel", ms=1e3 * t_fres[nb], stage1_ms="", stage2_ms="",
                             us_per_bin=1e6 * t_fres[nb] / nb, ratio=1.0, stage1_ratio="",
                             mm=mm_f, fresnel_degenerate=degenerate, chunked=False, block=nb,
                             construct_ms=""))

        # ---- one-time private-GBGPU construction (deepcopy orbits + re-configure) ----
        # This is the dominant per-CALL cost if left inline (~seconds; measured below), but
        # it is orbit/param/n_sub-independent -> a one-time SETUP cost in any real use (and
        # the fused Phase-2 kernel builds the orbit once too). Fresnel's baseline likewise
        # excludes its one-time gb/orbit setup. So we HOIST it: build gbtmp once here (this
        # is what clobbers the live group -- done AFTER all Fresnel timing) and REUSE it for
        # every slow-FFT call, timing the fair per-call physics. The inline-rebuild cost is
        # reported separately (construct_ms) as context, never folded into the ratio.
        make_slowpart_gbgpu(gb); sync()                        # warmup (JIT)
        _t0 = time.perf_counter(); gbtmp = make_slowpart_gbgpu(gb); sync()
        t_construct = time.perf_counter() - _t0
        print(f"\n[one-time] private-GBGPU construction (hoisted out of per-call): "
              f"{1e3*t_construct:.1f} ms  (num_bin-independent; excluded from ratios)", flush=True)

        # Verify reusing gbtmp is bit-identical to constructing inline (gbtmp=None): GBGPU is
        # built for repeated calls, but prove it so the hoist is not silently altering physics.
        dh_re, hh_re = get_ll_stft_slowfft_proto(grp, gb, p_inj, valid_n_sub[0], gbtmp=gbtmp)
        dh_fr, hh_fr = get_ll_stft_slowfft_proto(grp, gb, p_inj, valid_n_sub[0], gbtmp=None)
        reuse_rel = abs(_scalar(xp, dh_re) - _scalar(xp, dh_fr)) / (abs(_scalar(xp, dh_fr)) + 1e-300)
        assert reuse_rel < 1e-12, (
            f"reused gbtmp disagrees with inline build (rel={reuse_rel:.2e}) -- hoist unsafe")
        print(f"[verify] reused-gbtmp vs inline-build d_h rel={reuse_rel:.2e} (< 1e-12: hoist sound)",
              flush=True)

        # ---- slow-FFT gate + per-call timing (gbtmp reused; group already clobbered above) ----
        for n_sub in valid_n_sub:
            d_h, h_h = get_ll_stft_slowfft_proto(grp, gb, p_inj, n_sub, gbtmp=gbtmp)
            mm_s = _mism(xp, d_h, h_h, d_d)
            flag = "" if (np.isfinite(mm_s) and mm_s < 5e-2) else "  <-- CHECK (bad build?)"
            print(f"\n-- slow-FFT n_sub={n_sub} gate mm={mm_s:.3e}{flag} --", flush=True)
            for nb in a.num_bins:
                pr = bench_params[nb]

                def full_fn(blk, pr=pr, n_sub=n_sub):
                    return lambda: run_slowfft_chunked(grp, gb, pr, n_sub, blk, xp, on_gpu, gbtmp)

                def s1_fn(blk, pr=pr, n_sub=n_sub):
                    return lambda: run_stage1_chunked(gb, pr, t_seg, stft_dt, n_sub, blk, xp, on_gpu, gbtmp)

                block = resolve_block(full_fn, nb, a.min_block, oom, sync, free)
                t_full = time_call(full_fn(block), a.repeats, sync, warmup=False)
                t_s1 = time_call(s1_fn(block), a.repeats, sync, warmup=True)
                t_s2 = max(0.0, t_full - t_s1)
                chunked = block < nb
                mark = "  [CHUNKED]" if chunked else ""
                print(f"   num_bin={nb:6d} n_sub={n_sub:3d}: slowfft {1e3*t_full:9.2f} ms "
                      f"(stage1 {1e3*t_s1:8.2f} + stage2+ {1e3*t_s2:8.2f}) vs fresnel "
                      f"{1e3*t_fres[nb]:8.2f} ms  ratio={t_full/t_fres[nb]:7.2f}  "
                      f"stage1_ratio={t_s1/t_fres[nb]:6.2f}{mark}", flush=True)
                rows.append(dict(stft_hours=stft_hours, num_bin=nb, n_side=ns, n_sub=n_sub,
                                 kernel="slowfft", ms=1e3 * t_full, stage1_ms=1e3 * t_s1,
                                 stage2_ms=1e3 * t_s2, us_per_bin=1e6 * t_full / nb,
                                 ratio=t_full / t_fres[nb], stage1_ratio=t_s1 / t_fres[nb],
                                 mm=mm_s, fresnel_degenerate=degenerate, chunked=chunked,
                                 block=block, construct_ms=1e3 * t_construct))
        del gb, gbtmp

    if a.csv:
        _write_csv(a.csv, rows)
        print(f"\nwrote {len(rows)} rows -> {a.csv}", flush=True)
    print_summary(rows, header=f"stft_hours={stft_hours:g}")
    return rows


# ---------------------------------------------------------------------------
# parent orchestrator (one subprocess per stft_hours) + aggregation
# ---------------------------------------------------------------------------
def run_parent(a):
    rows = []
    tmpdir = tempfile.mkdtemp(prefix="bench_slowfft_")
    for h in a.stft_hours:
        child_csv = os.path.join(tmpdir, f"h{h:g}.csv")
        cmd = [sys.executable, os.path.abspath(__file__), "--_child",
               "--stft-hours", f"{h:g}", "--backend", a.backend, "--dt", str(a.dt),
               "--tobs-days", str(a.tobs_days), "--alpha", str(a.alpha),
               "--num-bins", ",".join(map(str, a.num_bins)),
               "--n-side", ",".join(map(str, a.n_side)),
               "--n-sub", ",".join(map(str, a.n_sub)),
               "--spread-bins", str(a.spread_bins), "--repeats", str(a.repeats),
               "--n-sparse", str(a.n_sparse), "--min-block", str(a.min_block),
               "--seed", str(a.seed), "--csv", child_csv]
        print(f"\n{'#'*70}\n# stft_hours={h:g}  (isolated subprocess: single live STFT group)\n"
              f"{'#'*70}", flush=True)
        r = subprocess.run(cmd)                      # inherit stdout/stderr + CUDA env
        if r.returncode != 0:
            print(f"!! child stft_hours={h:g} FAILED (rc={r.returncode}); its rows are omitted",
                  flush=True)
            continue
        if os.path.exists(child_csv):
            with open(child_csv, newline="") as fh:
                rows.extend(list(csv.DictReader(fh)))

    if a.csv and rows:
        _write_csv(a.csv, rows)
        print(f"\nwrote {len(rows)} aggregated rows -> {a.csv}", flush=True)
    print_summary(rows, header="ALL stft_hours")
    return 0


def _write_csv(path, rows):
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        w.writeheader()
        w.writerows(rows)


def print_summary(rows, header=""):
    sf = [r for r in rows if str(r.get("kernel")) == "slowfft"]
    print(f"\n===== summary [{header}] =====", flush=True)
    if not sf:
        print("  (no slow-FFT rows)", flush=True)
        return
    ratios = [_as_float(r["ratio"]) for r in sf]
    s1r = [_as_float(r["stage1_ratio"]) for r in sf]
    print(f"  slowfft/fresnel full ratio : best {np.nanmin(ratios):.2f} / worst {np.nanmax(ratios):.2f}",
          flush=True)
    print(f"  stage1-only/fresnel ratio  : best {np.nanmin(s1r):.2f} / worst {np.nanmax(s1r):.2f}  "
          f"(the fused-kernel-relevant lower bound)", flush=True)
    # headline config the brief asks for
    for r in sf:
        if int(_as_float(r["num_bin"])) == 16384 and abs(_as_float(r["stft_hours"]) - 8.0) < 1e-9:
            deg = str(r.get("fresnel_degenerate"))
            print(f"  [HEADLINE num_bin=16384 stft_hours=8 n_sub={r['n_sub']}] full ratio="
                  f"{_as_float(r['ratio']):.2f}  stage1 ratio={_as_float(r['stage1_ratio']):.2f}"
                  f"  chunked={r.get('chunked')}", flush=True)
    # largest un-chunked num_bin per (stft_hours, n_sub)
    fits = {}
    for r in sf:
        if str(r.get("chunked")) in ("False", "false", "0"):
            key = (_as_float(r["stft_hours"]), str(r["n_sub"]))
            fits[key] = max(fits.get(key, 0), int(_as_float(r["num_bin"])))
    if fits:
        parts = ", ".join(f"{h:g}h/n_sub={ns}:{nb}" for (h, ns), nb in sorted(fits.items()))
        print(f"  largest un-chunked num_bin: {parts}", flush=True)


def main(argv=None):
    a = parse_args(argv)
    if a._child or len(a.stft_hours) == 1:
        run_single(a, a.stft_hours[0])
        return 0
    return run_parent(a)


if __name__ == "__main__":
    sys.exit(main())
