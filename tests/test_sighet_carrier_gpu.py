"""GPU == CPU for the carrier sig-het path on UNEQUAL-ARM noise (cluster-only).

The carrier fold layouts (sym = the default, full = the fallback / 653594e's
unequal-arm path) and the one-channel carrier reference producer with its c1
pass run the same CUDA_DEVICE code on both backends; this pins that the GPU
build reproduces the CPU numbers on the small 15-day grid with a channel-
symmetric but NOT a*I + b*J inverse covariance (the production noise class).

Skipped unless a CUDA backend is available. Cluster recipe:
``SIGHET_GPU_TEST_BACKEND=cuda12x python -m unittest tests.test_sighet_carrier_gpu``
(add ``GB_SIGHET_V5_VERBOSE=1`` to print the v5 kernel's registers / blocks per SM).
"""

import os
import unittest

import numpy as np

from lisatools.detector import ESAOrbits
from lisatools.domains import WDMSettings
from lisatools.utils.constants import YRSID_SI

from gbgpu.gbcomps import GBWDMComputations
from gbgpu.gbsignalhetcomputations import (
    GBSignalHetComputations,
    _n_cp_kernel_arg,
    _window_dj,
)

V5_KNOBS = dict(v3_n_nodes=32, v4_knots=64, v4_band=16, v5=1)

#: GPU vs CPU tolerance, the repo convention (test_phase_max_fused): the two builds
#: differ only in summation order / FMA contraction, so compare at rtol 1e-9 with an
#: absolute floor at 1e-9 of the batch's LARGEST accumulation (a row whose value is a
#: deep cancellation -- d_h_im here -- carries the rounding of the big terms). Measured
#: on the first cluster run (cuda13x): d_h / h_h within 1e-10; d_h_im 2.2e-9 relative on
#: one cancelling row, identically on the pre-existing full layout and the new sym one;
#: the reference producer's sparse c0 9.5e-12 relative.
RTOL = 1e-9


def _dyn_atol(*arrays):
    m = max(float(np.max(np.abs(np.asarray(a)))) for a in arrays)
    return 1e-9 * max(m, 1e-300)


def _gpu_backend():
    want = os.environ.get("SIGHET_GPU_TEST_BACKEND")
    try:
        import lisatools
        import cupy  # noqa: F401
    except Exception:
        return None
    names = [want] if want else ["cuda12x", "cuda13x", "cuda11x"]
    for name in names:
        try:
            if lisatools.has_backend(name):
                return name
        except Exception:
            continue
    return None


GPU = _gpu_backend()


class _Holder:
    def __init__(self, data, invc, xp):
        self.linear_data_arr = [xp.ascontiguousarray(xp.asarray(data)).ravel()]
        self.linear_psd_arr = [xp.ascontiguousarray(xp.asarray(invc)).ravel()]

    def __len__(self):
        return 1


class _Env:
    def __init__(self, **kw):
        self.kw = kw

    def __enter__(self):
        self.saved = {k: os.environ.get(k) for k in self.kw}
        os.environ.update(self.kw)

    def __exit__(self, *a):
        for k, v in self.saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


@unittest.skipIf(GPU is None, "no CUDA backend (set SIGHET_GPU_TEST_BACKEND on the cluster)")
class CarrierGpuCpuParityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        dt, Nf, Nt, edge = 10.0, 256, 512, 40
        t0 = int(0.5 * YRSID_SI / dt) * dt
        layer_df = 1.0 / (2.0 * Nf * dt)
        rng = np.random.default_rng(17)
        cls.comps, cls.wdms = {}, {}
        for be in ("cpu", GPU):
            orbits = ESAOrbits(force_backend=be)
            wdm = WDMSettings(Nf, Nt, dt, t0=t0, min_freq=1e-4, max_freq=2e-2,
                              min_time=edge * Nf * dt, max_time=(Nt - edge) * Nf * dt,
                              force_backend=be)
            ch = GBWDMComputations(wdm, t_ref=t0, Nt_sub=128, n_pad=16, N_sparse=256,
                                   N_cp_sig=0, N_cp_orbit=0, orbits=orbits,
                                   tdi_config="2nd generation", force_backend=be, d_d=0.0,
                                   tdi_type="XYZ")
            ch.convert_to_ra_dec = False
            cls.comps[be] = ch
            cls.wdms[be] = wdm
        A = np.array([1e-21, (int(1.3e-3 / layer_df) + 0.41) * layer_df, 1e-17, 0.0, 1.2, 1.5,
                      0.4, 2.0, 0.5])
        C = np.array([8e-22, (int(5e-3 / layer_df) + 0.62) * layer_df, 2e-17, 0.0, 0.4, 0.7,
                      0.9, 4.0, -0.3])
        cls.params_ref = np.stack([A, C])
        wdm = cls.wdms["cpu"]
        ilo, ihi = wdm.ind_min_f, wdm.ind_max_f + 1
        slabs, invCs = [], []
        for p in (A, C):
            h = np.zeros((3, Nf, Nt))
            cls.comps["cpu"].fill_global_wdm(p[None, :], h, convert_to_ra_dec=False)
            h_act = np.ascontiguousarray(h[:, ilo:ihi, wdm.active_slice_t])
            slabs.append(h_act)
            nfa, nta = h_act.shape[1:]
            iC = np.empty((3, 3, nfa, nta))
            for c in range(3):
                iC[c, c] = rng.uniform(1.0, 2.0, (nfa, nta))
            for c, d in ((0, 1), (0, 2), (1, 2)):
                iC[c, d] = iC[d, c] = rng.uniform(-0.3, 0.3, (nfa, nta))
            invCs.append(iC)
        cls.slabs, cls.invCs = np.stack(slabs), np.stack(invCs)
        cls.slots = np.array([0, 1], dtype=np.int32)
        rows = [A, C]
        for p in (A, C):
            q = p.copy()
            q[0] *= 1.1
            q[1] += 0.02 * layer_df
            q[4] += 0.3
            rows.append(q)
        cls.params = np.stack(rows)
        cls.di = np.array([0, 1, 0, 1], dtype=np.int32)

    def _scores(self, be, **env):
        import cupy as cp
        xp = np if be == "cpu" else cp
        comp = GBSignalHetComputations.for_band_engine(self.comps[be], cp_repr="carrier",
                                                       **V5_KNOBS)
        holder = _Holder(self.slabs, self.invCs, xp)
        with _Env(**env):
            comp.setup_in_model(holder, self.params_ref, self.slots)
        try:
            comp.get_ll(self.params, data_index=self.di)
            out = dict(layout=comp._stash_layout,
                       d_h=np.asarray(cp.asnumpy(xp.asarray(comp.last_d_h))),
                       h_h=np.asarray(cp.asnumpy(xp.asarray(comp.last_h_h))),
                       d_h_im=np.asarray(cp.asnumpy(xp.asarray(comp.last_d_h_im))))
        finally:
            comp.clear_in_model()
        return out

    def _check(self, **env):
        cpu, gpu = self._scores("cpu", **env), self._scores(GPU, **env)
        self.assertEqual(cpu["layout"], gpu["layout"])
        atol = _dyn_atol(cpu["d_h"], cpu["h_h"], cpu["d_h_im"])
        for k in ("d_h", "h_h", "d_h_im"):
            np.testing.assert_allclose(gpu[k], cpu[k], rtol=RTOL, atol=atol,
                                       err_msg=f"{k} GPU vs CPU ({env})")
        return cpu["layout"]

    def test_sym_c1_default(self):
        self.assertEqual(self._check(SIGHET_ANCHOR_CORRECT="0"), "sym")

    def test_full_layout_fallback(self):
        self.assertEqual(self._check(SIGHET_CARRIER_LAYOUT="full", SIGHET_ANCHOR_CORRECT="0"),
                         "full")

    def test_carrier_fstat(self):
        """The carrier sig-het F-stat (carrier reference + SYM Gram fold), GPU == CPU."""
        import cupy as cp
        out = {}
        for be in ("cpu", GPU):
            xp = np if be == "cpu" else cp
            comp = GBSignalHetComputations.for_band_engine(self.comps[be], cp_repr="carrier",
                                                           **V5_KNOBS)
            holder = _Holder(self.slabs, self.invCs, xp)
            with _Env(SIGHET_FSTAT_CARRIER="1"):
                comp.setup_fstat_references(self.params_ref, holder, data_index=0,
                                            noise_index=0)
            self.assertTrue(comp._fstat["carrier"])
            N, M = comp.get_fstat_ll_wdm(xp.asarray(self.params),
                                         data_index=xp.zeros(len(self.params), dtype=xp.int32))
            out[be] = (np.asarray(cp.asnumpy(xp.asarray(N))), np.asarray(cp.asnumpy(xp.asarray(M))))
            comp.clear_fstat_references()
        for a, b, name in zip(out[GPU], out["cpu"], ("N", "M")):
            np.testing.assert_allclose(a, b, rtol=RTOL, atol=_dyn_atol(b),
                                       err_msg=f"F-stat {name} GPU vs CPU")

    def test_carrier_reference_producer(self):
        import cupy as cp
        out = {}
        for be in ("cpu", GPU):
            xp = np if be == "cpu" else cp
            comp = GBSignalHetComputations.for_band_engine(self.comps[be], cp_repr="carrier",
                                                           **V5_KNOBS)
            g = comp._g
            sp = xp.zeros((2, 1, g["Nf_active"], g["N_sparse_t"]), dtype=xp.complex128)
            de = xp.zeros((2, 1, g["Nf_active"], g["Nt_active"]), dtype=xp.complex128)
            c1 = xp.zeros_like(de)
            comp.cpp.gb_signal_het_make_reference_carrier(
                comp.tdi_wrap, sp, de, c1, comp.window_full, _window_dj(comp.window_full, xp),
                comp.n_sparse_local, xp.zeros(2, dtype=xp.int32),
                xp.ascontiguousarray(xp.asarray(self.params_ref)), 2, 9, 1, 2,
                g["Nf"], g["Nt"], g["Nf_active"], g["Nt_active"], g["nt_layer"],
                g["N_sparse_t"], g["stride"], g["ind_min_t"], g["ind_min_f"],
                g["layer_df"], g["dt"], g["Tobs"], g["t0"], 3, g["n_sparse_fd"],
                g["tukey_alpha"], _n_cp_kernel_arg(g, allow_carrier=True))
            out[be] = [np.asarray(cp.asnumpy(xp.asarray(a))) for a in (sp, de, c1)]
        for a, b, name in zip(out[GPU], out["cpu"], ("c0 sparse", "c0 dense", "c1 dense")):
            np.testing.assert_allclose(a, b, rtol=RTOL, atol=_dyn_atol(b),
                                       err_msg=f"{name} GPU vs CPU")


if __name__ == "__main__":
    unittest.main()
