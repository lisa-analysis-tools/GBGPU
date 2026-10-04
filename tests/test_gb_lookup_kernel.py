"""The fused GB direct-to-WDM lookup scorer (``gb_lookup_get_ll`` / GBLookupComputations).

Pinned against the exact chunked scorer on a short grid with a channel-symmetric
but NOT a*I + b*J inverse covariance (the unequal-arm production noise class: the
analytic XYZ covariance with per-channel diagonal offsets, so the near-singular
low-f X+Y+Z direction the K1 term protects is kept):

* d_h, h_h and the phase quadrature d_h_im match chunked (to the small table's
  ~1e-5 norm bias) -- d_h_im in chunked's own convention (= d_h at phi0 + pi/2);
* the coarse table path (read at the control points, interpolated in time --
  Mike 10-04) equals the per-pixel reads;
* narrow per-slot slabs == the full band;
* the amplitude-slope K1 term is load-bearing (dropping it multiplies the error).

CPU. Builds the GB lookup table (128-layer record; a minute or two) once into
``$GB_LOOKUP_TABLE_DIR`` (default ``~/.cache/gb_lookup_tables``), shared with the
LAT GB speed harness.
"""

import os
import unittest

import numpy as np

from lisatools.detector import ESAOrbits
from lisatools.domains import WDMSettings
from lisatools.utils.constants import YRSID_SI

from gbgpu.gbcomps import GBWDMComputations
from gbgpu.gblookupcomputations import GBLookupComputations

NF, DT, NT, EDGE, W = 180, 20.0, 720, 60, 5
#: the GB table recipe (LAT scripts/gb/_gb_testbox.py::GB_TABLE_RECIPE): a 128-layer build
#: record (the 32-layer one carries a ~2e-5 norm bias that would mask the K1 check)
RECIPE = dict(prefix="wdm_lookup_gb_cx", fdot_max_factor=0.1, time_layers=128, max_freq=2.5e-2)


def _table():
    from lisatools.wdm_lookup_store import ensure_lookup_table, lookup_table_path

    d = os.environ.get("GB_LOOKUP_TABLE_DIR", os.path.expanduser("~/.cache/gb_lookup_tables"))
    path = lookup_table_path(None, d, NF, DT, recipe=RECIPE)
    ensure_lookup_table(path, Nf=NF, dt=DT, recipe=RECIPE)
    return path


class _Holder:
    def __init__(self, data, invc, W=None, slab_lo=None):
        self.linear_data_arr = [np.ascontiguousarray(data).ravel()]
        self.linear_psd_arr = [np.ascontiguousarray(invc).ravel()]
        if W is not None:
            self.band_slab_Nf = int(W)
            self.slab_min_f = np.asarray(slab_lo, dtype=np.int32)
        self._n = len(data)

    def __len__(self):
        return self._n


class GBLookupKernelTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        t0 = int(0.4 * YRSID_SI / DT) * DT
        cls.wdm = wdm = WDMSettings(NF, NT, DT, t0=t0, min_freq=1e-4, max_freq=2.5e-2,
                                    min_time=EDGE * NF * DT, max_time=(NT - EDGE) * NF * DT,
                                    force_backend="cpu")
        orbits = ESAOrbits(force_backend="cpu")
        cls.chunked = ch = GBWDMComputations(
            wdm, t_ref=t0, Nt_sub=256, n_pad=32, N_sparse=256, N_cp_sig=48, N_cp_orbit=32,
            orbits=orbits, tdi_config="2nd generation", force_backend="cpu", d_d=0.0,
            tdi_type="XYZ")
        ch.convert_to_ra_dec = False
        cls.table = _table()
        rng = np.random.default_rng(5)
        df = wdm.layer_df
        # carrier in-layer offsets across the Meyer flat top and transition band
        f0 = np.array([(9 + 0.45) * df, (17 + 0.2) * df, (31 + 0.55) * df, (58 + 0.8) * df])
        R = len(f0)
        cls.p = np.column_stack([
            np.full(R, 1e-22), f0, 1e-18 * (f0 / 1e-3) ** (11 / 3), np.zeros(R),
            rng.uniform(0, 2 * np.pi, R), np.arccos(np.array([0.02, 0.4, 0.8, -0.3])),
            rng.uniform(0, np.pi, R), rng.uniform(0, 2 * np.pi, R),
            np.arcsin(rng.uniform(-1, 1, R))])
        T = int(wdm.Nt_active)
        Nfa = int(wdm.ind_max_f - wdm.ind_min_f + 1)
        cls.slab_lo = (np.floor(f0 / df).astype(int) - W // 2).astype(np.int32)
        from lisatools.sensitivity import X2TDISens, XY2TDISens

        fl = np.maximum((wdm.ind_min_f + np.arange(Nfa)) * df, 1e-5)
        sxx = np.asarray(X2TDISens.get_Sn(fl, model="scirdv1"), dtype=float)
        sxy = np.asarray(XY2TDISens.get_Sn(fl, model="scirdv1"), dtype=float)
        C = np.empty((Nfa, 3, 3))
        for c in range(3):
            for d in range(3):
                C[:, c, d] = sxx * (1.0 + 0.01 * (c - 1)) if c == d else sxy
        iC_l = np.linalg.inv(C).transpose(1, 2, 0)                  # (3, 3, Nfa)
        iC_full = np.broadcast_to(iC_l[None, :, :, :, None], (R, 3, 3, Nfa, T)).copy()
        idx = np.arange(R, dtype=np.int32)
        # data: each slot holds its own source at a shifted phase / amplitude
        q = cls.p.copy()
        q[:, 0] *= 1.3
        q[:, 4] += 0.7
        full = np.zeros((R, 3, Nfa, T))
        ch.fill_global_wdm(q, full.reshape(-1), data_index=idx, factors=np.ones(R))
        lo = cls.slab_lo - int(wdm.ind_min_f)
        narrow = np.stack([full[i][:, lo[i]:lo[i] + W] for i in range(R)])
        iC_n = np.stack([iC_full[i][:, :, lo[i]:lo[i] + W] for i in range(R)])
        cls.full = _Holder(full, iC_full)
        cls.narrow = _Holder(narrow, iC_n, W, cls.slab_lo)
        cls.idx = idx
        ch.get_ll_wdm(cls.p, cls.narrow, data_index=idx, noise_index=idx)
        cls.ref = dict(d_h=np.asarray(ch.d_h_out).copy(), h_h=np.asarray(ch.h_h_out).copy(),
                       d_h_im=np.asarray(ch.d_h_im_out).copy())

    def _lk(self, holder=None, **kw):
        lk = GBLookupComputations(self.chunked, self.table, **kw)
        lk.get_ll_wdm(self.p, self.narrow if holder is None else holder,
                      data_index=self.idx, noise_index=self.idx)
        return dict(d_h=np.asarray(lk.d_h_out).copy(), h_h=np.asarray(lk.h_h_out).copy(),
                    d_h_im=np.asarray(lk.d_h_im_out).copy())

    def _rel(self, out):
        hh = np.abs(self.ref["h_h"])
        return {k: np.abs(out[k] - self.ref[k]) / hh for k in ("d_h", "h_h", "d_h_im")}

    def test_matches_chunked(self):
        for k, e in self._rel(self._lk()).items():
            self.assertLess(e.max(), 2e-4, f"{k} vs chunked: {e}")

    def test_coarse_table_equals_per_pixel(self):
        a, b = self._lk(k_coarse=True), self._lk(k_coarse=False)
        for k in ("d_h", "h_h", "d_h_im"):
            np.testing.assert_allclose(a[k], b[k], rtol=0,
                                       atol=1e-8 * np.abs(self.ref["h_h"]).max(), err_msg=k)

    def test_narrow_equals_full_band(self):
        a, b = self._lk(), self._lk(holder=self.full)
        for k in ("d_h", "h_h", "d_h_im"):
            np.testing.assert_allclose(a[k], b[k], rtol=1e-12,
                                       atol=1e-12 * np.abs(self.ref["h_h"]).max(), err_msg=k)

    def test_chunked_subclass_integration(self):
        """GBLookupWDMComputations: a chunked comp whose get_ll_wdm is the lookup (same
        numbers as GBLookupComputations), fills inherited, and sig-het builds on it."""
        from gbgpu.gblookupcomputations import GBLookupWDMComputations
        from gbgpu.gbsignalhetcomputations import GBSignalHetComputations

        ch = self.chunked
        sub = GBLookupWDMComputations(
            self.wdm, t_ref=ch.t_ref, Nt_sub=256, n_pad=32, N_sparse=256, N_cp_sig=48,
            N_cp_orbit=32, orbits=ch.orbits, tdi_config="2nd generation", force_backend="cpu",
            d_d=0.0, tdi_type="XYZ", lookup_table=self.table)
        sub.convert_to_ra_dec = False
        ll = np.asarray(sub.get_ll_wdm(self.p, self.narrow, data_index=self.idx,
                                       noise_index=self.idx))
        ref = self._lk()
        np.testing.assert_allclose(np.asarray(sub.d_h_out), ref["d_h"], rtol=1e-13)
        np.testing.assert_allclose(np.asarray(sub.h_h_out), ref["h_h"], rtol=1e-13)
        np.testing.assert_allclose(ll, ref["d_h"] - 0.5 * ref["h_h"], rtol=1e-12)
        a = np.zeros(self.narrow.linear_data_arr[0].size)
        b = np.zeros_like(a)
        for comp, buf in ((sub, a), (ch, b)):
            comp.fill_global_wdm(self.p, buf, data_index=self.idx, factors=np.ones(len(self.p)),
                                 band_slab_Nf=W, slab_min_f=self.slab_lo)
        np.testing.assert_array_equal(a, b)
        sig = GBSignalHetComputations.for_band_engine(sub, cp_repr="carrier", v3_n_nodes=32,
                                                      v4_knots=64, v4_band=16, v5=1)
        self.assertIs(sig.chunked, sub)

    def test_k1_term_is_load_bearing(self):
        with_k1 = self._rel(self._lk())["h_h"]
        without = self._rel(self._lk(k1=False))["h_h"]
        self.assertGreater(without.max(), 10.0 * with_k1.max(),
                           f"no-K1 {without} vs K1 {with_k1}")


def _gpu_backend():
    want = os.environ.get("SIGHET_GPU_TEST_BACKEND")
    try:
        import lisatools
        import cupy  # noqa: F401
    except Exception:
        return None
    for name in ([want] if want else ["cuda12x", "cuda13x", "cuda11x"]):
        try:
            if lisatools.has_backend(name):
                return name
        except Exception:
            continue
    return None


GPU = _gpu_backend()


@unittest.skipIf(GPU is None, "no CUDA backend (set SIGHET_GPU_TEST_BACKEND on the cluster)")
class GBLookupGpuParityTest(GBLookupKernelTest):
    """GPU == CPU for the lookup kernel (both table paths), cluster-only."""

    def test_gpu_matches_cpu(self):
        import cupy as cp

        wdm = self.wdm
        g_wdm = WDMSettings(NF, NT, DT, t0=wdm.t0, min_freq=1e-4, max_freq=2.5e-2,
                            min_time=EDGE * NF * DT, max_time=(NT - EDGE) * NF * DT,
                            force_backend=GPU)
        g_ch = GBWDMComputations(
            g_wdm, t_ref=self.chunked.t_ref, Nt_sub=256, n_pad=32, N_sparse=256, N_cp_sig=48,
            N_cp_orbit=32, orbits=ESAOrbits(force_backend=GPU), tdi_config="2nd generation",
            force_backend=GPU, d_d=0.0, tdi_type="XYZ")
        h = self.narrow

        class _G:
            linear_data_arr = [cp.asarray(h.linear_data_arr[0])]
            linear_psd_arr = [cp.asarray(h.linear_psd_arr[0])]
            band_slab_Nf = h.band_slab_Nf
            slab_min_f = cp.asarray(h.slab_min_f)

            def __len__(self):
                return len(h)

        for kc in (True, False):
            cpu = self._lk(k_coarse=kc)
            lk = GBLookupComputations(g_ch, self.table, k_coarse=kc)
            lk.get_ll_wdm(cp.asarray(self.p), _G(), data_index=cp.asarray(self.idx),
                          noise_index=cp.asarray(self.idx))
            for k, v in (("d_h", lk.d_h_out), ("h_h", lk.h_h_out), ("d_h_im", lk.d_h_im_out)):
                np.testing.assert_allclose(cp.asnumpy(v), cpu[k], rtol=1e-10,
                                           atol=1e-10 * np.abs(cpu["h_h"]).max(),
                                           err_msg=f"{k} GPU vs CPU (k_coarse={kc})")


if __name__ == "__main__":
    unittest.main()
