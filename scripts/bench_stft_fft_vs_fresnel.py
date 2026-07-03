"""GPU go/no-go benchmark: FFT-per-column vs Fresnel STFT GB ``get_ll``.

Times ``STFTGBComputations.get_ll_stft_fft`` (FFTColumn: heterodyne+twiddle DFT +
orbit spline cache) against ``get_ll_stft`` (analytic Fresnel) on the same data,
sweeping the decision grid from the design doc (docs/specs/2026-07-01-stft-gb-fft-
per-column-design.md, §8 + §12.2): ``num_bin`` x ``n_side_bins`` x ``n_sub`` x
``n_cp_orbit``. Go = FFT wall-clock beats Fresnel at matched accuracy on the
production-ish configs (num_bin ~ O(10^4), n_side=10, n_sub=32/64, n_cp_orbit=48).

Before timing, each ``n_side`` block runs a correctness gate on an injected GB:
recovery mismatch vs the true STFT for both paths + their d_h agreement — a broken
GPU build fails loudly here instead of producing fast garbage.

Usage:
    # GPU box, full sweep (backend auto-picks the best available, i.e. CUDA):
    python scripts/bench_stft_fft_vs_fresnel.py --csv bench_stft_fft.csv
    # CPU validation of the script itself (small everything):
    python scripts/bench_stft_fft_vs_fresnel.py --smoke

Occupancy (the shared-mem/register ceiling question, R2 cache ~23 KB at n_cp=48):
    ncu --kernel-name 'regex:stft_get_ll' --section Occupancy \
        python scripts/bench_stft_fft_vs_fresnel.py --num-bins 4096 --repeats 1
"""
import argparse
import csv
import sys
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

AMP, F0, FDOT, FDDOT = 1e-23, 4.2300812341e-3, 1e-18, 0.0
PHI0, INC, PSI, LAM, BETA = 0.892342342342, 1.2309804223, 3.00908098, 4.827342308, -0.50923423

N_SUB_MAX = 64      # STFT_FFT_NSUB_MAX (lat_stft_kernels.hh)
N_CP_MAX = 48       # STFT_ORBIT_NCP_MAX


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
    p.add_argument("--stft-hours", type=float, default=6.0, help="STFT segment length")
    p.add_argument("--tobs-days", type=float, default=91.0,
                   help="observation span (rounded down to whole segments)")
    p.add_argument("--alpha", type=float, default=0.1, help="Tukey analysis-window alpha")
    p.add_argument("--num-bins", default="1024,4096,16384",
                   help="comma list of binary counts to time")
    p.add_argument("--n-side", default="10", help="comma list of n_side_bins")
    p.add_argument("--n-sub", default="24,32,64", help="comma list of FFT n_sub")
    p.add_argument("--n-cp-orbit", default="48,0",
                   help="comma list of orbit spline-cache densities (0 = direct get_tdi)")
    p.add_argument("--spread-bins", type=float, default=32.0,
                   help="benchmark sources drawn uniformly in f0 = kc +- spread bins")
    p.add_argument("--repeats", type=int, default=5, help="timed calls per config (min reported)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--csv", default=None, help="write result rows to this CSV path")
    p.add_argument("--smoke", action="store_true",
                   help="tiny config to validate the script (CPU-friendly)")
    a = p.parse_args(argv)
    if a.smoke:
        a.tobs_days, a.num_bins, a.n_sub, a.repeats = 8.0, "8", "24,32", 2
    a.num_bins = [int(x) for x in a.num_bins.split(",")]
    a.n_side = [int(x) for x in a.n_side.split(",")]
    a.n_sub = [int(x) for x in a.n_sub.split(",")]
    a.n_cp_orbit = [int(x) for x in a.n_cp_orbit.split(",")]
    return a


def main(argv=None):
    a = parse_args(argv)
    # Resolve "auto" to a concrete backend name ourselves: passing force_backend=None
    # crashes in classes without supported_backends() (e.g. XYZSensitivityBackend).
    if a.backend == "auto":
        for fb in ("gpu", "cpu"):
            try:
                orbits = DefaultOrbits(force_backend=fb)
                break
            except Exception:
                continue
        else:
            raise RuntimeError("no usable backend (tried 'gpu', 'cpu')")
    else:
        fb = a.backend
        orbits = DefaultOrbits(force_backend=fb)
    orbits.configure(linear_interp_setup=True)
    xp = orbits.xp
    on_gpu = xp.__name__ == "cupy"

    def sync():
        if on_gpu:
            xp.cuda.runtime.deviceSynchronize()

    def scalar(arr):
        return complex(xp.asarray(arr).reshape(-1)[0])

    def mism(d_h, h_h, d_d):
        return abs(1.0 - scalar(d_h).real / np.sqrt(d_d * scalar(h_h).real))

    print(f"backend: {fb} -> xp={xp.__name__}"
          + (f" [{xp.cuda.runtime.getDeviceProperties(0)['name'].decode()}]" if on_gpu else ""))

    # --- data: inject one GB, STFT it with the Tukey analysis window -------------
    tdi_config = TDIConfig("2nd generation", force_backend=fb)
    stft_dt = a.stft_hours * 3600.0
    nperseg = int(round(stft_dt / a.dt))
    n_stft = int(a.tobs_days * 86400.0 // stft_dt)
    nobs = n_stft * nperseg
    Tobs = nobs * a.dt
    df = 1.0 / stft_dt
    kc = int(round(F0 / df))
    band = int(np.ceil(a.spread_bins)) + max(a.n_side) + 20
    print(f"Tobs = {Tobs:.3e} s ({Tobs/86400:.1f} d)  segments: NT={n_stft} x {a.stft_hours}h  "
          f"nperseg={nperseg}  band: kc={kc} +- {band} bins  alpha={a.alpha}")

    data_t = xp.arange(nobs) * a.dt
    t_tdi = xp.linspace(0.0, Tobs, 513)[1:-1]
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
    p_inj = np.array([[AMP, F0, FDOT, FDDOT, PHI0, INC, PSI, LAM, BETA]])

    # --- benchmark parameter sets: num_bin distinct sources around the carrier ---
    rng = np.random.default_rng(a.seed)
    def draw(nb):
        f0 = (kc + rng.uniform(-a.spread_bins, a.spread_bins, nb)) * df
        return np.column_stack([
            np.full(nb, AMP), f0, FDOT * rng.uniform(0.5, 2.0, nb), np.zeros(nb),
            rng.uniform(0.0, 2 * np.pi, nb), np.arccos(rng.uniform(-1.0, 1.0, nb)),
            rng.uniform(0.0, np.pi, nb), rng.uniform(0.0, 2 * np.pi, nb),
            np.arcsin(rng.uniform(-1.0, 1.0, nb)),
        ])
    bench_params = {nb: draw(nb) for nb in a.num_bins}

    def time_call(fn):
        fn(); sync()                       # warmup (also JIT/caches)
        ts = []
        for _ in range(a.repeats):
            t0 = time.perf_counter(); fn(); sync(); ts.append(time.perf_counter() - t0)
        return min(ts)

    rows = []
    for ns in a.n_side:
        gb = STFTGBComputations(stft_comps=grp, T=Tobs, t_ref=0.0, orbits=orbits,
                                tdi_config=tdi_config, force_backend=fb, n_side_bins=ns,
                                window_factor=1.0, freq_from_tdi_phase=True)
        # correctness gate (injected source): both paths must recover the injection
        gb.get_ll_stft(p_inj)
        mm_f = mism(gb.d_h_out, gb.h_h_out, d_d)
        dh_f = scalar(gb.d_h_out)
        print(f"\n== n_side={ns}  (need n_sub >= {2*ns+1})  fresnel recovery mm={mm_f:.3e} ==")
        gate = {}
        for n_sub in a.n_sub:
            if n_sub < 2 * ns + 1 or n_sub > N_SUB_MAX:
                print(f"   skip n_sub={n_sub}: outside [{2*ns+1}, {N_SUB_MAX}] (aliases/over guard)")
                continue
            for n_cp in a.n_cp_orbit:
                if n_cp > N_CP_MAX:
                    print(f"   skip n_cp_orbit={n_cp} > {N_CP_MAX}")
                    continue
                gb.get_ll_stft_fft(p_inj, n_sub=n_sub, n_cp_orbit=n_cp)
                mm_x = mism(gb.d_h_out_fft, gb.h_h_out_fft, d_d)
                dh_rel = abs(scalar(gb.d_h_out_fft) - dh_f) / abs(dh_f)
                gate[(n_sub, n_cp)] = (mm_x, dh_rel)
                print(f"   n_sub={n_sub:3d} n_cp={n_cp:2d}: fft recovery mm={mm_x:.3e}  "
                      f"dh vs fresnel rel={dh_rel:.3e}")
        # timing
        for nb in a.num_bins:
            pr = bench_params[nb]
            t_f = time_call(lambda: gb.get_ll_stft(pr))
            rows.append(dict(kernel="fresnel", num_bin=nb, n_side=ns, n_sub="", n_cp="",
                             ms=1e3 * t_f, us_per_bin=1e6 * t_f / nb, ratio=1.0,
                             mm=mm_f, dh_rel=0.0))
            for (n_sub, n_cp), (mm_x, dh_rel) in gate.items():
                t_x = time_call(lambda: gb.get_ll_stft_fft(pr, n_sub=n_sub, n_cp_orbit=n_cp))
                rows.append(dict(kernel="fft", num_bin=nb, n_side=ns, n_sub=n_sub, n_cp=n_cp,
                                 ms=1e3 * t_x, us_per_bin=1e6 * t_x / nb, ratio=t_x / t_f,
                                 mm=mm_x, dh_rel=dh_rel))
                print(f"   num_bin={nb:6d} n_sub={n_sub:3d} n_cp={n_cp:2d}: "
                      f"fft {1e3*t_x:9.2f} ms vs fresnel {1e3*t_f:9.2f} ms  "
                      f"ratio={t_x/t_f:6.3f}")
        del gb

    print(f"\n{'kernel':>8} {'num_bin':>8} {'n_side':>6} {'n_sub':>5} {'n_cp':>4} "
          f"{'ms/call':>10} {'us/bin':>8} {'vs_fresnel':>10} {'recovery_mm':>11}")
    for r in rows:
        print(f"{r['kernel']:>8} {r['num_bin']:>8} {r['n_side']:>6} {str(r['n_sub']):>5} "
              f"{str(r['n_cp']):>4} {r['ms']:>10.2f} {r['us_per_bin']:>8.2f} "
              f"{r['ratio']:>10.3f} {r['mm']:>11.3e}")

    if a.csv:
        with open(a.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        print(f"\nwrote {len(rows)} rows -> {a.csv}")

    worst = max((r["ratio"] for r in rows if r["kernel"] == "fft"), default=float("nan"))
    best = min((r["ratio"] for r in rows if r["kernel"] == "fft"), default=float("nan"))
    print(f"\nFFT/Fresnel wall-clock ratio: best {best:.3f} / worst {worst:.3f} "
          f"(<1 = FFT faster; the go/no-go number)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
