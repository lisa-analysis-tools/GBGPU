"""Tests of the DFFR response of galactic binaries: ``STFTGBComputations(response="dffr")``.

The response runs on the consumers of the existing STFT response (lat_stft_kernels.hh) with the
DFFR tile (lat_dffr_kernels.hh) and GBDFFRSource, through the gb_dffr_* entry points of the
backend. ``_DFFRCase`` calls the comp's methods; it calls the entry points directly only to
compare them with the comp, to check their own arguments, and for the 16-row tiles.

* Wiring: every comp method equals its entry point called with the expected arguments, bit for
  bit, so a misplaced or ignored setting cannot pass unseen.
* Self-consistency, as ``test_stft_gb.py`` checks the existing response: fill_global then the
  domain inner product equals get_ll; swap(A, A) equals get_ll(A) and relabelling add and remove
  swaps the terms; the F-statistic's N and M equal the get_ll and swap terms; both gradients equal
  host central differences, and eps <= 0 freezes a component.
* The compact template equals the fill_global grid bit for bit, and swap on a union band of several
  bins equals the inner products of compact templates on that band.
* The envelope of the gradients, device replicas, and DFFR against the existing response on the
  same fixture, both TDI generations: a sign, a conjugation or a factor 2 in either would show in
  the overlap or the norm ratio.
* The Fisher matrix against the reference Fisher of ``gbgpu.testing``, whose fill_global templates
  each stencil template of the kernel equals bit for bit; its exact amplitude and phase entries;
  subsets, window starts and frozen steps; a carrier crossing and a chirp that changes sign; and
  the parameters the source model declares free of the reception states.
* Argument checks, and the CUDA backend against the CPU backend (skipped without a usable GPU).
"""

import unittest

import numpy as np

from lisatools.domaincomputation import KINEMATICS_TABLE_FIELDS
from lisatools.domains import get_stft_settings
from lisatools.response.tdiconfig import TDIConfig

try:
    from gbgpu.gbcomps import STFTGBComputations
    from gbgpu.testing import stft_information_matrix_reference

    HAVE_STFT_GB = True
except (ImportError, ModuleNotFoundError):
    HAVE_STFT_GB = False

# GB params (amp, f0, fdot, fddot, phi0, iota, psi, lam, beta), those of test_stft_gb.py.
GB_PARAMS = np.array(
    [1e-23, 4.2300812341e-3, 1e-18, 0.0, 0.892342, 1.230980, 3.009081, 4.827342, -0.509234]
)
GB_PARAMS_B = np.array(
    [2e-23, 4.2800123000e-3, 2e-18, 0.0, 1.100000, 0.700000, 2.200000, 3.500000, 0.300000]
)
# * 2.6 bins of the six-hour segments above GB_PARAMS: its band overlaps that of GB_PARAMS in part,
# * so swap sums both over a union band two to three bins wider than either.
GB_PARAMS_C = np.array(
    [1.5e-23, 4.2300812341e-3 + 2.6 / 21600.0, -3e-18, 0.0, 0.3, 2.0, 0.4, 1.9, 0.9]
)
GRAD_EPS = np.array([1e-26, 1e-11, 1e-21, 1e-28, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6])

#: 24 six-hour segments of 10 s samples, the band 3.9 to 4.6 mHz.
SAMPLE_DT = 10.0
SEGMENT_DT = 6 * 3600.0
NUM_SAMPLES = int(24 * SEGMENT_DT / SAMPLE_DT)
N_SIDE_BINS = 3
# ? Bounds of the comparison with the existing response, set before measuring: a sign or a
# ? conjugation gives an overlap near zero or negative, and the norm ratio catches a factor 2 (a
# ? ratio of 4 or 1/4); the two responses' own approximations stay far inside both.
OVERLAP_BOUND = 0.99
NORM_RATIO_BOUNDS = (0.9, 1.1)
# ? CUDA against CPU, provisional until a first GPU run measures the differences.
GPU_RELATIVE_BOUND = 1e-12
#: The parameters of the global fit's GB proposal: fddot is filled, not sampled.
TEST_INDS = [0, 1, 2, 4, 5, 6, 7, 8]
# ? The kernel and the reference Fisher difference the same templates, bit for bit, so only the
# ? order of the contraction separates them; the existing kernel measured 2.9e-15 on its fixture.
FISHER_BOUND = 1e-13
# ? F_AA A^2 and F_phi0phi0 against (h|h). The stencil is exact for an amplitude that enters
# ? linearly, but the phase step of 1e-6 rad is added to phases of up to 1.4e4 rad on this fixture,
# ? whose rounding (half an ulp, 9e-13 rad) leaves the realised step off by up to 9e-7 of itself:
# ? measured 2.1e-7 here and 1.4e-6 in the existing response (2026-10-19).
EXACT_ENTRY_BOUND = 1e-5
# ? The A and phi0 gradients at the default steps against (Re(d|h) - (h|h)) / A and Im(d|h). A
# ? enters linearly, so its difference is exact to rounding (3e-14 of the component measured).
EXACT_AMPLITUDE_GRADIENT_BOUND = 1e-5
# ? The phase step is realised to 9e-7 of itself per evaluation (phases up to 1.4e4 rad), and the
# ? segment terms of Im(d|h) cancel on the random data (their absolute sum is 3.7 times the total
# ? for GB_PARAMS): measured 1.1e-5 of the component for GB_PARAMS, 1.5e-6 for GB_PARAMS_B
# ? (2026-10-24). 1e-4 of the component is the error at which the kernel's phase would be fixed.
EXACT_PHASE_GRADIENT_BOUND = 1e-4
# ? Direction of the Taylor remainder: 1e2 x GRAD_EPS, calibrated once (2026-10-24). The remainder
# ? at the smallest step stays 40 times above the linear term of the gradient's own step error;
# ? 1e1 x pulled the last ratio of GB_PARAMS to 3.63, and 1e3 x let the cubic term push the first
# ? of GB_PARAMS_B to 4.29.
TAYLOR_DIRECTION_SCALE = 1e2
TAYLOR_STEPS = (1.0, 0.5, 0.25, 0.125)
TAYLOR_RATIO_BOUNDS = (3.5, 4.5)
# ? CUDA Fisher against the CPU and the reference, provisional: a template that rounds differently
# ? on the device enters the differences divided by the step, about 1e-9 in fdot on 8 parameters.
GPU_FISHER_BOUND = 1e-7


def _host(array):
    return array.get() if hasattr(array, "get") else np.asarray(array)


def _arrays(values):
    """The output arrays of one method, whether it returns one array or several."""
    return list(values) if isinstance(values, (tuple, list)) else [values]


def _settings(backend_name):
    return get_stft_settings(
        np.arange(NUM_SAMPLES) * SAMPLE_DT,
        SEGMENT_DT,
        min_freq=3.9e-3,
        max_freq=4.6e-3,
        force_backend=backend_name,
    )


class _StftCompsShim:
    """Stand-in for STFTComputationGroup: ``cpp_fresnel``, ``cpp_domain``, ``d_d``, the group's
    ``linear_envelope``, ``tdi_type`` and ``num_noise``, and ``cpp_kinematics``, which returns a
    table built by the test and records the orbits it was asked for."""

    def __init__(
        self, cpp_fresnel, cpp_domain, table, linear_envelope, tdi_type, d_d=None, num_noise=1
    ):
        self.cpp_fresnel = cpp_fresnel
        self.cpp_domain = cpp_domain
        self.table = table
        self.linear_envelope = linear_envelope
        self.tdi_type = tdi_type
        self.d_d = d_d
        self.num_noise = num_noise
        self.orbits_requested = None

    def cpp_kinematics(self, cpp_orbits):
        self.orbits_requested = cpp_orbits
        return self.table


class _DFFRCase:
    """One fixture on one backend: the DFFR comp bound to a domain and Fresnel shim with the
    kinematics table of the grid, the comp's methods, and the gb_dffr_* entry points with the
    arguments the comp must pass (``knobs``)."""

    def __init__(
        self,
        backend_name,
        window_alpha=0.0,
        use_midpoint=False,
        tdi="1st generation",
        tdi_type="XYZ",
        first_midpoint=None,
        n_side_bins=N_SIDE_BINS,
        linear_envelope=True,
        first_order_delays=True,
        t_ref=0.0,
    ):
        self.settings = _settings(backend_name)
        # * Bound after construction: the shim needs the comp's backend and orbits.
        self.comp = STFTGBComputations(
            stft_comps=None,
            T=NUM_SAMPLES * SAMPLE_DT,
            t_ref=t_ref,
            orbits=None,
            tdi_config=TDIConfig(tdi, force_backend=backend_name),
            force_backend=backend_name,
            n_side_bins=n_side_bins,
            linear_envelope=linear_envelope,
            response="dffr",
            first_order_delays=first_order_delays,
        )
        self.xp = self.comp.xp
        self.table = self._table(first_midpoint)
        self.shim = self._shim(window_alpha, use_midpoint, tdi_type, linear_envelope)
        self.comp.stft_comps = self.shim
        generation = {"1st generation": 1, "2nd generation": 2}[tdi]
        self.knobs = (
            t_ref,
            n_side_bins,
            generation,
            linear_envelope,
            first_order_delays,
        )
        self.n_side_bins = n_side_bins
        self.wrap = self.comp.backend.GBComputationGroupWrap()

    def _shim(self, window_alpha, use_midpoint, tdi_type, linear_envelope):
        """Domain (seeded data, identity inverse covariance) and Fresnel object of the grid."""
        settings, backend, xp = self.settings, self.comp.backend, self.xp
        nt, nf, nch = settings.NT, settings.NF_active, 3
        rng = np.random.default_rng(1234)
        data = rng.standard_normal((1, nch, nt, nf)) + 1j * rng.standard_normal((1, nch, nt, nf))
        if tdi_type == "XYZ":
            inv_c = np.zeros((1, nch, nch, nt, nf), dtype=np.complex128)
            for channel in range(nch):
                inv_c[:, channel, channel] = 1.0
        else:
            inv_c = np.ones((1, nch, nt, nf), dtype=np.complex128)
        data = xp.asarray(np.ascontiguousarray(data.astype(np.complex128)))
        inv_c = xp.asarray(np.ascontiguousarray(inv_c))
        domain = backend.STFTDomainWrap(
            nt,
            nf,
            nch,
            settings.t0,
            settings.min_freq,
            settings.max_freq,
            settings.dt,
            settings.df,
            data.reshape(-1),
            inv_c.reshape(-1),
            1,
            1,
            backend.TDITypeDict[tdi_type],
        )
        fresnel = backend.STFTFresnelWrap(
            nt,
            nf,
            nch,
            settings.t0,
            settings.min_freq,
            settings.max_freq,
            settings.dt,
            settings.df,
            window_alpha=window_alpha,
            use_midpoint=use_midpoint,
        )
        shim = _StftCompsShim(fresnel, domain, self.table, linear_envelope, tdi_type)
        shim.keepalive = (data, inv_c)
        shim.data = data
        return shim

    def _table(self, first_midpoint):
        """The kinematics table of the grid from the comp's equal-arm orbits."""
        settings = self.settings
        self.table_arrays = {
            name: self.xp.zeros(int(np.prod(shape)) * settings.NT)
            for name, shape in KINEMATICS_TABLE_FIELDS
        }
        midpoint = settings.t0 + 0.5 * settings.dt if first_midpoint is None else first_midpoint
        table = self.comp.backend.KinematicsTableWrap(
            midpoint, settings.dt, settings.NT, *self.table_arrays.values()
        )
        table.build(self.comp.cpp_orbits)
        return table

    def _grid(self):
        return self.table, self.shim.cpp_fresnel, self.shim.cpp_domain

    def _params(self, params):
        return self.xp.ascontiguousarray(self.xp.asarray(np.atleast_2d(params), dtype=np.float64))

    def _indices(self, num_bin):
        return (self.xp.zeros(num_bin, dtype=np.int32), self.xp.zeros(num_bin, dtype=np.int32))

    def _cells(self, starts):
        """Window starts per cell for the entry points; None is the full grid (empty array)."""
        if starts is None:
            return self.xp.zeros(0, dtype=np.int32)
        return self.xp.asarray(np.asarray(starts, dtype=np.int32))

    # ---- the comp ----------------------------------------------------------------------------

    def get_ll(self, params, starts=None):
        self.comp.get_ll_stft(params, start_freq_inds=starts)
        return _host(self.comp.d_h_out), _host(self.comp.h_h_out)

    def swap(self, params_add, params_remove, starts=None):
        terms = self.comp.get_swap_ll_stft(params_add, params_remove, start_freq_inds=starts)[2:]
        return [_host(term) for term in terms]

    def fstat(self, params, starts=None):
        self.comp.get_fstat_ll_stft(params, start_freq_inds=starts)
        return _host(self.comp.N_arr_cmplx), _host(self.comp.M_mat_cmplx)

    def fill_global(self, params, starts=None):
        """The template grid (1, 3, NT, NF_active) of the sources, all with factor one."""
        settings = self.settings
        templates = self.xp.zeros((1, 3, settings.NT, settings.NF_active), dtype=np.complex128)
        self.comp.fill_global_stft(params, templates, start_freq_inds=starts)
        return templates

    def get_ll_grad(self, params, eps, starts=None, **options):
        return _host(
            self.comp.get_ll_grad_stft(params, param_eps=eps, start_freq_inds=starts, **options)
        )

    def swap_ll_grad(self, params_add, params_remove, eps_add, eps_remove, starts=None, **options):
        grads = self.comp.get_swap_ll_grad_stft(
            params_add,
            params_remove,
            param_eps_add=eps_add,
            param_eps_remove=eps_remove,
            start_freq_inds=starts,
            **options,
        )
        return tuple(_host(grad) for grad in grads)

    # ---- the entry points, with the arguments the comp must pass -------------------------------

    def entry_get_ll(self, params, starts=None, knobs=None):
        batch = self._params(params)
        num_bin = batch.shape[0]
        d_h = self.xp.zeros(num_bin, dtype=np.complex128)
        h_h = self.xp.zeros(num_bin, dtype=np.complex128)
        self.wrap.gb_dffr_get_ll(
            d_h,
            h_h,
            *self._grid(),
            batch.reshape(-1),
            *self._indices(num_bin),
            num_bin,
            9,
            *(self.knobs if knobs is None else knobs),
            self._cells(starts),
        )
        return _host(d_h), _host(h_h)

    def entry_swap(self, params_add, params_remove, rows_per_pass=32, starts=None):
        batch_add, batch_remove = self._params(params_add), self._params(params_remove)
        num_bin = batch_add.shape[0]
        terms = [self.xp.zeros(num_bin, dtype=np.complex128) for _ in range(5)]
        self.wrap.gb_dffr_swap_ll(
            *terms,
            *self._grid(),
            batch_add.reshape(-1),
            batch_remove.reshape(-1),
            *self._indices(num_bin),
            num_bin,
            9,
            *self.knobs,
            rows_per_pass,
            self._cells(starts),
        )
        return [_host(term) for term in terms]

    def entry_fstat(self, params, rows_per_pass=32, starts=None):
        batch = self._params(params)
        num_bin = batch.shape[0]
        outs = [self.xp.zeros(num_bin * size) for size in (4, 4, 10, 10)]
        self.wrap.gb_dffr_get_fstat_ll(
            *outs,
            *self._grid(),
            batch.reshape(-1),
            *self._indices(num_bin),
            num_bin,
            9,
            *self.knobs,
            rows_per_pass,
            self._cells(starts),
        )
        n_re, n_im, m_re, m_im = (_host(out).reshape(num_bin, -1) for out in outs)
        return n_re + 1j * n_im, m_re + 1j * m_im

    def entry_fill_global(self, params, starts=None):
        settings = self.settings
        batch = self._params(params)
        num_bin = batch.shape[0]
        templates = self.xp.zeros((1, 3, settings.NT, settings.NF_active), dtype=np.complex128)
        self.wrap.gb_dffr_fill_global(
            templates.reshape(-1),
            *self._grid(),
            batch.reshape(-1),
            self.xp.zeros(num_bin, dtype=np.int32),
            self.xp.ones(num_bin),
            num_bin,
            9,
            *self.knobs,
            self._cells(starts),
        )
        return templates

    def entry_get_ll_grad(self, params, eps, starts=None):
        batch = self._params(params)
        num_bin = batch.shape[0]
        grad = self.xp.zeros(num_bin * 9)
        self.wrap.gb_dffr_get_ll_grad(
            grad,
            *self._grid(),
            batch.reshape(-1),
            *self._indices(num_bin),
            self.xp.asarray(eps, dtype=np.float64),
            num_bin,
            9,
            *self.knobs,
            self._cells(starts),
        )
        return _host(grad).reshape(num_bin, 9)

    def entry_swap_ll_grad(self, params_add, params_remove, eps_add, eps_remove, starts=None):
        batch_add, batch_remove = self._params(params_add), self._params(params_remove)
        num_bin = batch_add.shape[0]
        grad_add, grad_remove = self.xp.zeros(num_bin * 9), self.xp.zeros(num_bin * 9)
        self.wrap.gb_dffr_swap_ll_grad(
            grad_add,
            grad_remove,
            *self._grid(),
            batch_add.reshape(-1),
            batch_remove.reshape(-1),
            *self._indices(num_bin),
            self.xp.asarray(eps_add, dtype=np.float64),
            self.xp.asarray(eps_remove, dtype=np.float64),
            num_bin,
            9,
            *self.knobs,
            32,
            self._cells(starts),
        )
        return _host(grad_add).reshape(num_bin, 9), _host(grad_remove).reshape(num_bin, 9)

    def entry_information_matrix(
        self,
        params,
        eps,
        inds=None,
        noise_index=None,
        easy_central_difference=False,
        rows_per_pass=4,
        starts=None,
    ):
        batch = self._params(params)
        num_bin = batch.shape[0]
        inds = list(range(9)) if inds is None else list(inds)
        noise_index = np.zeros(num_bin) if noise_index is None else noise_index
        info = self.xp.zeros(num_bin * len(inds) ** 2)
        self.wrap.gb_dffr_information_matrix(
            info,
            *self._grid(),
            batch.reshape(-1),
            self.xp.asarray(np.asarray(noise_index, dtype=np.int32)),
            self.xp.asarray(np.asarray(inds, dtype=np.int32)),
            self.xp.asarray(eps, dtype=np.float64),
            num_bin,
            9,
            len(inds),
            *self.knobs,
            easy_central_difference,
            rows_per_pass,
            self._cells(starts),
        )
        return _host(info).reshape(num_bin, len(inds), len(inds))

    def segment_table(self, params):
        """Effective frequencies, chirp rates (num_bin, NT, 3) and band starts (num_bin, NT)."""
        batch = self._params(params)
        num_bin = batch.shape[0]
        segments = num_bin * self.settings.NT
        outputs = dict(
            effective_frequency=self.xp.zeros(segments * 3),
            chirp_rate=self.xp.zeros(segments * 3),
            reference_phasor=self.xp.zeros(segments * 3, dtype=np.complex128),
            link_amplitude=self.xp.zeros(segments * 6, dtype=np.complex128),
            envelope_slope=self.xp.zeros(segments * 6, dtype=np.complex128),
            emission_phasor=self.xp.zeros(segments * 6, dtype=np.complex128),
            emission_frequency=self.xp.zeros(segments * 6),
            tdi_weights=self.xp.zeros(segments * 18, dtype=np.complex128),
            start_freq_ind=self.xp.zeros(segments, dtype=np.int32),
        )
        self.wrap.gb_dffr_segment_table(
            *outputs.values(), *self._grid(), batch.reshape(-1), num_bin, 9, *self.knobs
        )

        def per_segment(name, width):
            values = _host(outputs[name]).reshape(num_bin, self.settings.NT, width)
            return values[..., 0] if width == 1 else values

        return {
            name: per_segment(name, width)
            for name, width in (
                ("effective_frequency", 3),
                ("chirp_rate", 3),
                ("emission_frequency", 6),
                ("start_freq_ind", 1),
            )
        }

    def fill_template(self, params):
        """Compact templates (num_bin, 3, NT, band) and band starts (num_bin, NT)."""
        batch = self._params(params)
        num_bin = batch.shape[0]
        num_times = self.settings.NT
        band = 2 * self.n_side_bins + 1
        template = self.xp.zeros(num_bin * 3 * num_times * band, dtype=np.complex128)
        starts = self.xp.zeros(num_bin * num_times, dtype=np.int32)
        self.wrap.gb_dffr_fill_template(
            template, starts, *self._grid(), batch.reshape(-1), num_bin, 9, *self.knobs
        )
        return (
            _host(template).reshape(num_bin, 3, num_times, band),
            _host(starts).reshape(num_bin, num_times),
        )

    def domain_inner_product(self, templates):
        """(d|h) and (h|h) of a template grid through the domain's own inner product."""
        settings = self.settings
        d_h = self.xp.zeros(1, dtype=np.complex128)
        h_h = self.xp.zeros(1, dtype=np.complex128)
        self.shim.cpp_domain.compute_likelihood_terms(
            d_h,
            h_h,
            templates.reshape(-1),
            self.xp.asarray([0.0]),
            self.xp.asarray([settings.min_freq]),
            1,
            self.xp.zeros(1, dtype=np.int32),
            self.xp.zeros(1, dtype=np.int32),
            settings.NT,
            settings.NF_active,
            False,
        )
        return _host(d_h), _host(h_h)


@unittest.skipUnless(HAVE_STFT_GB, "requires the GBGPU STFT-GB build")
class DFFRWiringTest(unittest.TestCase):
    """Every comp method equals its entry point called with the expected arguments, bit for bit."""

    backend_name = "cpu"

    # * Two settings that differ in every argument the comp passes: generation, envelope, delays,
    # * t_ref, window and window starts.
    SETTINGS = (
        dict(
            options=dict(
                tdi="1st generation",
                linear_envelope=False,
                first_order_delays=False,
                t_ref=1234.5,
                window_alpha=0.0,
            ),
            starts=[2],
        ),
        dict(
            options=dict(
                tdi="2nd generation",
                linear_envelope=True,
                first_order_delays=True,
                t_ref=0.0,
                window_alpha=0.5,
                use_midpoint=True,
            ),
            starts=None,
        ),
    )

    def test_comp_methods_equal_entry_points(self):
        batch = np.stack([GB_PARAMS, GB_PARAMS_B])
        batch_remove = np.stack([GB_PARAMS_B, GB_PARAMS_C])
        for setting in self.SETTINGS:
            options, starts = setting["options"], setting["starts"]
            with self.subTest(**options):
                case = _DFFRCase(self.backend_name, **options)
                pairs = {
                    "get_ll": (case.get_ll(batch, starts), case.entry_get_ll(batch, starts)),
                    "swap": (
                        case.swap(batch, batch_remove, starts),
                        case.entry_swap(batch, batch_remove, starts=starts),
                    ),
                    "fstat": (case.fstat(batch, starts), case.entry_fstat(batch, starts=starts)),
                    "fill_global": (
                        _host(case.fill_global(batch, starts)),
                        _host(case.entry_fill_global(batch, starts)),
                    ),
                    "get_ll_grad": (
                        case.get_ll_grad(batch, GRAD_EPS, starts),
                        case.entry_get_ll_grad(batch, GRAD_EPS, starts),
                    ),
                    "swap_ll_grad": (
                        case.swap_ll_grad(batch, batch_remove, GRAD_EPS, GRAD_EPS, starts),
                        case.entry_swap_ll_grad(batch, batch_remove, GRAD_EPS, GRAD_EPS, starts),
                    ),
                    "information_matrix": (
                        _host(case.comp.information_matrix(batch, start_freq_inds=starts)),
                        case.entry_information_matrix(
                            batch,
                            case.comp._information_matrix_steps(9, None, None, 1e-6)[0],
                            starts=starts,
                        ),
                    ),
                }
                for name, (comp_values, entry_values) in pairs.items():
                    for comp_value, entry_value in zip(_arrays(comp_values), _arrays(entry_values)):
                        with self.subTest(method=name):
                            self.assertGreater(np.abs(entry_value).max(), 0.0)
                            np.testing.assert_array_equal(comp_value, entry_value)
                # ! The table must come from the comp's own orbits.
                self.assertIs(case.shim.orbits_requested, case.comp.cpp_orbits)


@unittest.skipUnless(HAVE_STFT_GB, "requires the GBGPU STFT-GB build")
class DFFRSelfConsistencyTest(unittest.TestCase):
    """The consumer identities of test_stft_gb.py for the DFFR response, CPU backend."""

    backend_name = "cpu"

    def _case(self, **options):
        return _DFFRCase(self.backend_name, **options)

    def test_fill_global_matches_get_ll(self):
        """get_ll equals fill_global followed by the domain inner product."""
        for window_alpha in (0.0, 0.5):
            for tdi in ("1st generation", "2nd generation"):
                with self.subTest(window_alpha=window_alpha, tdi=tdi):
                    case = self._case(window_alpha=window_alpha, tdi=tdi)
                    d_h, h_h = case.get_ll(GB_PARAMS)
                    d_h_fill, h_h_fill = case.domain_inner_product(case.fill_global(GB_PARAMS))
                    self.assertGreater(h_h[0].real, 0.0)
                    np.testing.assert_allclose(d_h_fill, d_h, rtol=1e-10, atol=0)
                    np.testing.assert_allclose(h_h_fill, h_h, rtol=1e-10, atol=0)

    def test_compact_template_equals_fill_global(self):
        """The compact template is the fill_global grid on its band, bit for bit; zero elsewhere."""
        case = self._case(window_alpha=0.5, tdi="2nd generation")
        settings = case.settings
        grid = _host(case.fill_global(GB_PARAMS))[0]
        compact, starts = (array[0] for array in case.fill_template(GB_PARAMS))
        rebuilt = np.zeros_like(grid)
        for segment in range(settings.NT):
            for band_bin in range(compact.shape[-1]):
                bin_index = starts[segment] + band_bin
                if 0 <= bin_index < settings.NF_active:
                    rebuilt[:, segment, bin_index] = compact[:, segment, band_bin]
        self.assertGreater(np.abs(grid).max(), 0.0)
        np.testing.assert_array_equal(rebuilt, grid)

    def test_swap_degenerate_matches_get_ll(self):
        """swap(A, A): every term equals the (d|h) or (h|h) of get_ll(A)."""
        case = self._case(window_alpha=0.5, tdi="2nd generation")
        d_h, h_h = case.get_ll(GB_PARAMS)
        d_h_a, d_h_r, aa, rr, ar = case.swap(GB_PARAMS, GB_PARAMS)
        for value, reference in ((d_h_a, d_h), (d_h_r, d_h), (aa, h_h), (rr, h_h), (ar, h_h)):
            np.testing.assert_allclose(value, reference, rtol=1e-10, atol=0)

    def test_swap_add_remove_symmetry(self):
        """swap(B, A) holds the terms of swap(A, B) relabelled, the cross term conjugated."""
        case = self._case(window_alpha=0.5, tdi="2nd generation")
        ab = case.swap(GB_PARAMS, GB_PARAMS_B)
        ba = case.swap(GB_PARAMS_B, GB_PARAMS)
        np.testing.assert_allclose(ba[0], ab[1], rtol=1e-10, atol=0)
        np.testing.assert_allclose(ba[1], ab[0], rtol=1e-10, atol=0)
        np.testing.assert_allclose(ba[2], ab[3], rtol=1e-10, atol=0)
        np.testing.assert_allclose(ba[3], ab[2], rtol=1e-10, atol=0)
        np.testing.assert_allclose(ba[4], np.conj(ab[4]), rtol=1e-10, atol=0)
        self.assertGreater(np.abs(ab[4][0]), 0.0)

    def test_swap_union_band_matches_compact_templates(self):
        """swap(A, C) sums both templates over the union of their bands, in every segment."""
        case = self._case(window_alpha=0.5, tdi="2nd generation")
        wide_side_bins = N_SIDE_BINS + 6
        wide = self._case(window_alpha=0.5, tdi="2nd generation", n_side_bins=wide_side_bins)
        settings = case.settings
        templates, starts = wide.fill_template(np.stack([GB_PARAMS, GB_PARAMS_C]))
        carriers = starts + wide_side_bins
        data = _host(case.shim.data)[0]
        sums = np.zeros(5, dtype=np.complex128)
        for segment in range(settings.NT):
            low = carriers[:, segment].min() - N_SIDE_BINS
            high = carriers[:, segment].max() + N_SIDE_BINS
            for bin_index in range(max(low, 0), min(high, settings.NF_active - 1) + 1):
                add = templates[0, :, segment, bin_index - starts[0, segment]]
                remove = templates[1, :, segment, bin_index - starts[1, segment]]
                data_pixel = data[:, segment, bin_index]
                sums += [
                    np.vdot(data_pixel, add),
                    np.vdot(data_pixel, remove),
                    np.vdot(add, add),
                    np.vdot(remove, remove),
                    np.vdot(add, remove),
                ]
        reference = 4.0 * settings.df * sums
        # * The carriers differ by more than a bin, so the union band is a real one.
        self.assertGreater(np.abs(carriers[0] - carriers[1]).max(), 1)
        values = np.array([term[0] for term in case.swap(GB_PARAMS, GB_PARAMS_C)])
        np.testing.assert_allclose(values, reference, rtol=1e-10, atol=0)

    def test_fstat_matches_get_ll_and_swap(self):
        """N_i = (d|A_i) and M_ij = (A_i|A_j) equal the get_ll and swap terms of the filters."""
        case = self._case(window_alpha=0.5, tdi="2nd generation")
        n_values, m_values = (array[0] for array in case.fstat(GB_PARAMS))
        basis = []
        for amplitude, iota, psi, phi0 in zip(
            (2.0,) * 4,
            (np.pi / 2,) * 4,
            (0.0, np.pi / 4, 0.0, np.pi / 4),
            (0.0, np.pi, 3 * np.pi / 2, np.pi / 2),
        ):
            filter_params = GB_PARAMS.copy()
            filter_params[[0, 5, 6, 4]] = amplitude, iota, psi, phi0
            basis.append(filter_params)
        n_reference = np.zeros(4, dtype=np.complex128)
        m_reference = np.zeros((4, 4), dtype=np.complex128)
        for first in range(4):
            d_h, h_h = case.get_ll(basis[first])
            n_reference[first], m_reference[first, first] = d_h[0], h_h[0]
            for second in range(first + 1, 4):
                m_reference[first, second] = case.swap(basis[first], basis[second])[4][0]
        np.testing.assert_allclose(n_values, n_reference, rtol=1e-9, atol=0)
        np.testing.assert_allclose(m_values, m_reference[np.triu_indices(4)], rtol=1e-9, atol=0)

    def test_get_ll_grad_matches_host_differences(self):
        """The gradient kernel equals central differences of get_ll; eps <= 0 freezes."""
        case = self._case(window_alpha=0.5, tdi="2nd generation")
        eps = GRAD_EPS.copy()
        eps[4] = -1.0
        grad = case.get_ll_grad(GB_PARAMS, eps)[0]
        reference = np.zeros(9)
        for parameter in range(9):
            if eps[parameter] <= 0.0:
                continue
            values = []
            for sign in (1.0, -1.0):
                shifted = GB_PARAMS.copy()
                shifted[parameter] += sign * eps[parameter]
                d_h, h_h = case.get_ll(shifted)
                values.append(d_h[0].real - 0.5 * h_h[0].real)
            reference[parameter] = (values[0] - values[1]) / (2.0 * eps[parameter])
        self.assertEqual(grad[4], 0.0)
        self.assertGreater(np.abs(grad).max(), 0.0)
        np.testing.assert_allclose(grad, reference, rtol=1e-6, atol=1e-25)

    def test_exact_amplitude_and_phase_gradients(self):
        """At the default steps, dlnL/dA = (Re(d|h) - (h|h)) / A, since h is linear in A, and
        dlnL/dphi0 = Im(d|h), since the pixels carry exp(i Phi) with Phi = -phi0 + ..., so that
        dh/dphi0 = -i h."""
        case = self._case(window_alpha=0.5, tdi="2nd generation")
        for params in (GB_PARAMS, GB_PARAMS_B):
            with self.subTest(amp=params[0]):
                grad = _host(case.comp.get_ll_grad_stft(params))[0]
                d_h, h_h = (value[0] for value in case.get_ll(params))
                exact_amp = (d_h.real - h_h.real) / params[0]
                exact_phase = d_h.imag
                # * Both components far from zero, so a relative bound means something.
                self.assertGreater(abs(exact_amp) * params[0], 0.1 * abs(d_h))
                self.assertGreater(abs(exact_phase), 0.1 * abs(d_h))
                np.testing.assert_allclose(
                    grad[0], exact_amp, rtol=EXACT_AMPLITUDE_GRADIENT_BOUND, atol=0
                )
                np.testing.assert_allclose(
                    grad[4], exact_phase, rtol=EXACT_PHASE_GRADIENT_BOUND, atol=0
                )

    def test_gradient_taylor_remainder_order(self):
        """R1(s) = |lnL(theta + s d) - lnL(theta) - s g.d| falls as s^2 over three halvings, so the
        default-step gradient is the first-order term of get_ll along a direction of natural
        scales."""
        case = self._case(window_alpha=0.5, tdi="2nd generation")
        direction = TAYLOR_DIRECTION_SCALE * GRAD_EPS
        direction[3] = 0.0

        def log_likelihood(params):
            d_h, h_h = case.get_ll(params)
            return d_h[0].real - 0.5 * h_h[0].real

        for params in (GB_PARAMS, GB_PARAMS_B):
            with self.subTest(amp=params[0]):
                slope = _host(case.comp.get_ll_grad_stft(params))[0] @ direction
                base = log_likelihood(params)
                remainders = np.array(
                    [
                        abs(log_likelihood(params + step * direction) - base - step * slope)
                        for step in TAYLOR_STEPS
                    ]
                )
                ratios = remainders[:-1] / remainders[1:]
                self.assertTrue(
                    np.all((ratios >= TAYLOR_RATIO_BOUNDS[0]) & (ratios <= TAYLOR_RATIO_BOUNDS[1])),
                    msg=f"remainder ratios {ratios}",
                )

    def test_swap_ll_grad_matches_host_differences(self):
        """Both swap gradients equal central differences of the swap scalar."""
        case = self._case(window_alpha=0.5, tdi="2nd generation")

        def swap_scalar(params_add, params_remove):
            d_h_a, d_h_r, aa, rr, ar = (term[0] for term in case.swap(params_add, params_remove))
            return (d_h_a - d_h_r - 0.5 * aa - 0.5 * rr + ar).real

        grad_add, grad_remove = (
            grad[0] for grad in case.swap_ll_grad(GB_PARAMS, GB_PARAMS_B, GRAD_EPS, GRAD_EPS)
        )
        for grad, side in ((grad_add, 0), (grad_remove, 1)):
            reference = np.zeros(9)
            for parameter in range(9):
                values = []
                for sign in (1.0, -1.0):
                    pair = [GB_PARAMS.copy(), GB_PARAMS_B.copy()]
                    pair[side][parameter] += sign * GRAD_EPS[parameter]
                    values.append(swap_scalar(*pair))
                reference[parameter] = (values[0] - values[1]) / (2.0 * GRAD_EPS[parameter])
            with self.subTest(side=("add", "remove")[side]):
                np.testing.assert_allclose(grad, reference, rtol=1e-6, atol=1e-25)


@unittest.skipUnless(HAVE_STFT_GB, "requires the GBGPU STFT-GB build")
class DFFREnvelopeAndReplicaTest(unittest.TestCase):
    """The envelope setting of DFFR calls, and device replicas of a DFFR comp."""

    def test_forward_methods_follow_the_group(self):
        """get_ll reads the group's linear_envelope; the setting changes the templates."""
        on = _DFFRCase("cpu", window_alpha=0.5, tdi="2nd generation", linear_envelope=True)
        off = _DFFRCase("cpu", window_alpha=0.5, tdi="2nd generation", linear_envelope=False)
        self.assertFalse(np.array_equal(on.get_ll(GB_PARAMS)[1], off.get_ll(GB_PARAMS)[1]))
        np.testing.assert_array_equal(off.get_ll(GB_PARAMS)[1], off.entry_get_ll(GB_PARAMS)[1])

    def test_gradient_default_is_the_group_setting(self):
        """The default envelope of both gradients is the group's; an explicit value applies."""
        on = _DFFRCase("cpu", window_alpha=0.5, tdi="2nd generation", linear_envelope=True)
        off = _DFFRCase("cpu", window_alpha=0.5, tdi="2nd generation", linear_envelope=False)
        default = on.get_ll_grad(GB_PARAMS, GRAD_EPS)
        np.testing.assert_array_equal(
            on.get_ll_grad(GB_PARAMS, GRAD_EPS, linear_envelope=None), default
        )
        np.testing.assert_array_equal(
            on.get_ll_grad(GB_PARAMS, GRAD_EPS, linear_envelope=True), default
        )
        forced_off = on.get_ll_grad(GB_PARAMS, GRAD_EPS, linear_envelope=False)
        self.assertFalse(np.array_equal(forced_off, default))
        # * The envelope is a kernel argument, so forcing it off equals a group without it.
        np.testing.assert_array_equal(forced_off, off.get_ll_grad(GB_PARAMS, GRAD_EPS))
        np.testing.assert_array_equal(
            off.get_ll_grad(GB_PARAMS, GRAD_EPS, linear_envelope=True), default
        )
        swap_default = on.swap_ll_grad(GB_PARAMS, GB_PARAMS_B, GRAD_EPS, GRAD_EPS)
        swap_forced_off = on.swap_ll_grad(
            GB_PARAMS, GB_PARAMS_B, GRAD_EPS, GRAD_EPS, linear_envelope=False
        )
        swap_off = off.swap_ll_grad(GB_PARAMS, GB_PARAMS_B, GRAD_EPS, GRAD_EPS)
        for side in range(2):
            self.assertFalse(np.array_equal(swap_default[side], swap_off[side]))
            np.testing.assert_array_equal(swap_forced_off[side], swap_off[side])

    def test_replica_keeps_the_response(self):
        """A comp rebuilt from args and kwargs, directly and through LAT's replica builder,
        keeps the response and its settings and gives the same outputs bit for bit."""
        from lisatools.utils.devicereplicas import _build_gb_comp_replica

        case = _DFFRCase("cpu", tdi="2nd generation", first_order_delays=False)
        reference = case.get_ll(GB_PARAMS)
        replicas = {
            "args and kwargs": type(case.comp)(*case.comp.args, **case.comp.kwargs),
            # * Device 1 against primary 0 takes the multi-GPU rebuild path; on numpy the
            # * device contexts are no-ops.
            "replica builder": _build_gb_comp_replica(case.comp, np, 1, 0),
        }
        for name, replica in replicas.items():
            with self.subTest(replica=name):
                self.assertIsNot(replica, case.comp)
                self.assertEqual(replica.response, "dffr")
                self.assertFalse(replica.first_order_delays)
                self.assertEqual(replica.tdi_config.generation, 2)
                replica.stft_comps = case.shim
                replica.get_ll_stft(GB_PARAMS)
                np.testing.assert_array_equal(_host(replica.d_h_out), reference[0])
                np.testing.assert_array_equal(_host(replica.h_h_out), reference[1])


@unittest.skipUnless(HAVE_STFT_GB, "requires the GBGPU STFT-GB build")
class DFFRAgainstExistingResponseTest(unittest.TestCase):
    """DFFR and the existing response model the same source: their templates nearly agree."""

    def test_templates_overlap(self):
        for tdi in ("1st generation", "2nd generation"):
            with self.subTest(tdi=tdi):
                case = _DFFRCase(
                    "cpu", window_alpha=0.5, use_midpoint=True, tdi=tdi, linear_envelope=False
                )
                settings = case.settings
                existing = STFTGBComputations(
                    stft_comps=case.shim,
                    T=NUM_SAMPLES * SAMPLE_DT,
                    t_ref=0.0,
                    orbits=None,
                    tdi_config=TDIConfig(tdi, force_backend="cpu"),
                    force_backend="cpu",
                    n_side_bins=N_SIDE_BINS,
                    linear_envelope=False,
                    response="tdi_on_the_fly",
                )
                dffr = _host(case.fill_global(GB_PARAMS)).ravel()
                column = case.xp.zeros((1, 3, settings.NT, settings.NF_active), dtype=np.complex128)
                existing.fill_global_stft(GB_PARAMS, column)
                column = _host(column).ravel()
                overlap = np.vdot(dffr, column).real / np.sqrt(
                    np.vdot(dffr, dffr).real * np.vdot(column, column).real
                )
                norm_ratio = np.vdot(dffr, dffr).real / np.vdot(column, column).real
                self.assertGreater(overlap, OVERLAP_BOUND, msg=f"overlap {overlap:.6f}")
                self.assertTrue(
                    NORM_RATIO_BOUNDS[0] < norm_ratio < NORM_RATIO_BOUNDS[1],
                    msg=f"norm ratio {norm_ratio:.6f}",
                )


class _FisherShim(_StftCompsShim):
    """The shim plus what the reference Fisher reads from a computation group: the grid, the
    inverse CSD rows it contracts against, and their count."""

    def __init__(self, cpp_fresnel, cpp_domain, table, linear_envelope, inv_c, grid, num_noise):
        super().__init__(
            cpp_fresnel, cpp_domain, table, linear_envelope, "XYZ", num_noise=num_noise
        )
        self.invC_arr = inv_c
        self.settings = grid
        self.num_channels = 3

    def _owned_cpp_array(self, name, arr, dtype):
        return arr


class _DFFRFisherCase(_DFFRCase):
    """A DFFR case whose domain holds two random Hermitian positive-definite inverse CSDs, the
    rows the Fisher matrix contracts against, on the full grid or on a window of it."""

    NUM_NOISE = 2

    def __init__(self, backend_name, start_bin=0, trim_bins=0, **options):
        self.start_bin, self.trim_bins = start_bin, trim_bins
        super().__init__(backend_name, **options)

    def _shim(self, window_alpha, use_midpoint, tdi_type, linear_envelope):
        settings, backend, xp = self.settings, self.comp.backend, self.xp
        nt, nch, num_noise = settings.NT, 3, self.NUM_NOISE
        nf_full = settings.NF_active
        rng = np.random.default_rng(7)
        raw = rng.standard_normal((num_noise, nt, nf_full, nch, nch)) + 1j * rng.standard_normal(
            (num_noise, nt, nf_full, nch, nch)
        )
        hpd = raw @ np.conj(np.swapaxes(raw, -1, -2)) + nch * np.eye(nch)
        nf = nf_full - self.start_bin - self.trim_bins
        # * A window is the full-grid inverse CSD sliced in frequency.
        inv_c = np.transpose(hpd, (0, 3, 4, 1, 2))[..., self.start_bin : self.start_bin + nf]
        inv_c = xp.asarray(np.ascontiguousarray(inv_c))
        data = xp.zeros((num_noise, nch, nt, nf), dtype=np.complex128)
        domain = backend.STFTDomainWrap(
            nt,
            nf,
            nch,
            settings.t0,
            settings.min_freq,
            settings.max_freq,
            settings.dt,
            settings.df,
            data.reshape(-1),
            inv_c.reshape(-1),
            num_noise,
            num_noise,
            backend.TDITypeDict[tdi_type],
        )
        fresnel = backend.STFTFresnelWrap(
            nt,
            nf,
            nch,
            settings.t0,
            settings.min_freq,
            settings.max_freq,
            settings.dt,
            settings.df,
            window_alpha=window_alpha,
            use_midpoint=use_midpoint,
        )
        grid = type("Grid", (), {"NT": nt, "NF_active": nf, "df": settings.df})()
        shim = _FisherShim(fresnel, domain, self.table, linear_envelope, inv_c, grid, num_noise)
        shim.keepalive = (data, inv_c)
        shim.data = data
        return shim

    def fisher(self, params, noise_index, **options):
        return _host(self.comp.information_matrix(params, noise_index=noise_index, **options))

    def reference(self, params, noise_index, **options):
        # * None keeps the group's envelope, which fill_global_stft applies for DFFR.
        return _host(
            stft_information_matrix_reference(
                self.comp, params, noise_index=noise_index, linear_envelope=None, **options
            )
        )


@unittest.skipUnless(HAVE_STFT_GB, "requires the GBGPU STFT-GB build")
class DFFRFisherTest(unittest.TestCase):
    """The DFFR Fisher matrix against the reference Fisher built from fill_global templates, its
    exact entries, and the arguments it shares with the Fisher of the existing response. Random
    Hermitian positive-definite inverse CSDs exercise the cross-channel contraction; the metric is
    |F_kernel - F_ref| / sqrt(F_ii F_jj), because off-diagonal entries pass through zero."""

    backend_name = "cpu"
    # * Two settings that differ in the TDI generation, the window, the envelope and the delays.
    SETTINGS = (
        dict(tdi="2nd generation", window_alpha=0.5, linear_envelope=True, first_order_delays=True),
        dict(
            tdi="1st generation", window_alpha=0.0, linear_envelope=False, first_order_delays=False
        ),
    )

    def _case(self, **options):
        return _DFFRFisherCase(self.backend_name, **options)

    def _sources(self):
        return np.stack([GB_PARAMS, GB_PARAMS_B]), np.array([0, 1], dtype=np.int32)

    def _steps(self, case):
        return case.comp._information_matrix_steps(9, None, None, 1e-6)[0]

    def _assert_parity(self, kernel, reference, bound=FISHER_BOUND):
        diagonal = np.abs(np.diagonal(reference, axis1=1, axis2=2))
        self.assertTrue(np.all(diagonal > 0.0))
        scale = np.sqrt(diagonal[:, :, None] * diagonal[:, None, :])
        worst = float(np.max(np.abs(kernel - reference) / scale))
        self.assertLess(worst, bound, msg=f"kernel left the reference by {worst:.3e}")

    def test_parity_with_reference(self):
        params, noise_index = self._sources()
        for setting in self.SETTINGS:
            case = self._case(**setting)
            for options in (
                {},
                {"inds": TEST_INDS},
                {"inds": TEST_INDS, "easy_central_difference": True},
            ):
                with self.subTest(**setting, **{key: str(value) for key, value in options.items()}):
                    kernel = case.fisher(params, noise_index, **options)
                    self._assert_parity(kernel, case.reference(params, noise_index, **options))
                    np.testing.assert_array_equal(kernel, np.swapaxes(kernel, 1, 2))

    def test_parity_on_one_set_of_states(self):
        """A and phi0 leave the reception states unchanged, so their eight variants share one set:
        the smallest layout, whose build grows the anchor space to hold one variant at a time."""
        params, noise_index = self._sources()
        for setting in self.SETTINGS:
            with self.subTest(**setting):
                case = self._case(**setting)
                kernel = case.fisher(params, noise_index, inds=[0, 4])
                self._assert_parity(kernel, case.reference(params, noise_index, inds=[0, 4]))

    def test_subset_is_submatrix(self):
        case = self._case(**self.SETTINGS[0])
        params, noise_index = self._sources()
        subset = [1, 4, 7]
        full = case.fisher(params, noise_index)
        sub = case.fisher(params, noise_index, inds=subset)
        np.testing.assert_array_equal(sub, full[:, subset][:, :, subset])

    def test_window_reproduces_full_grid(self):
        """Both sources lie inside a window one bin shorter at each end."""
        full_case = self._case(**self.SETTINGS[0])
        window_case = self._case(start_bin=1, trim_bins=1, **self.SETTINGS[0])
        params, noise_index = self._sources()
        full = full_case.fisher(params, noise_index, inds=TEST_INDS)
        windowed = window_case.fisher(
            params, noise_index, inds=TEST_INDS, start_freq_inds=np.array([1, 1], dtype=np.int32)
        )
        np.testing.assert_array_equal(windowed, full)

    def test_nonpositive_step_freezes(self):
        case = self._case(**self.SETTINGS[0])
        params, noise_index = self._sources()
        eps = self._steps(case).copy()
        eps[5] = 0.0
        info = case.fisher(params, noise_index, inds=TEST_INDS, param_eps=eps)
        row = TEST_INDS.index(5)
        self.assertEqual(float(np.abs(info[:, row, :]).max()), 0.0)
        self.assertEqual(float(np.abs(info[:, :, row]).max()), 0.0)
        self.assertGreater(float(np.abs(info[:, 0, 0]).min()), 0.0)

    def test_exact_amplitude_and_phase_entries(self):
        """h is linear in A, and phi0 multiplies it by exp(-i phi0), so F_AA A^2 and F_phi0phi0
        equal (h|h) of get_ll and F_A,phi0 vanishes."""
        params, noise_index = self._sources()
        for setting in self.SETTINGS:
            with self.subTest(**setting):
                case = self._case(**setting)
                info = case.fisher(params, noise_index, inds=[0, 4])
                case.comp.get_ll_stft(params, noise_index=noise_index)
                h_h = _host(case.comp.h_h_out).real
                np.testing.assert_allclose(
                    info[:, 0, 0] * params[:, 0] ** 2, h_h, rtol=EXACT_ENTRY_BOUND, atol=0
                )
                np.testing.assert_allclose(info[:, 1, 1], h_h, rtol=EXACT_ENTRY_BOUND, atol=0)
                cross = np.abs(info[:, 0, 1]) / np.sqrt(info[:, 0, 0] * info[:, 1, 1])
                self.assertLess(float(cross.max()), EXACT_ENTRY_BOUND)

    def test_carrier_crossing_does_not_inflate(self):
        """A source whose carrier sits on a half-bin boundary in one segment, so that the stencil
        moves it across the bin there, against the same source a quarter bin away. All templates
        of a derivative share the union of their bands; with each on its own band the boundary
        source was 1e3 to 2e4 times larger in the existing response."""
        case = self._case(**self.SETTINGS[0])
        settings = case.settings
        segment = settings.NT // 2
        # ? Receivers of the links 12, 23, 31, 13, 32, 21: the carrier of segment_table's build.
        receivers = np.array([0, 1, 2, 0, 2, 1])
        boundary = GB_PARAMS.copy()
        for _ in range(4):
            table = case.segment_table(boundary)
            carrier = np.mean(
                0.5
                * (
                    table["effective_frequency"][0, segment, receivers]
                    + table["emission_frequency"][0, segment]
                )
            )
            offset = (carrier - settings.min_freq) / settings.df
            boundary[1] += settings.min_freq + (np.floor(offset) + 0.5) * settings.df - carrier
        away = boundary.copy()
        away[1] += 0.25 * settings.df
        # * The f0 stencil must cross the bin in that segment, or the test checks nothing.
        step = self._steps(case)[1]
        shifted = np.stack([boundary + sign * step * np.eye(9)[1] for sign in (1.0, -1.0)])
        band_starts = case.segment_table(shifted)["start_freq_ind"][:, segment]
        self.assertNotEqual(band_starts[0], band_starts[1])

        info = case.fisher(np.stack([boundary, away]), np.zeros(2, dtype=np.int32), inds=TEST_INDS)
        ratio = np.abs(np.diagonal(info[0])) / np.abs(np.diagonal(info[1]))
        for index, name in enumerate(("amp", "f0", "fdot", "phi0", "iota", "psi", "lam", "beta")):
            with self.subTest(parameter=name):
                self.assertGreater(ratio[index], 0.5, msg=f"{name} diagonal collapsed")
                self.assertLess(ratio[index], 2.0, msg=f"{name} diagonal inflated: {ratio[index]}")

    def test_chirp_sign_change(self):
        """A source whose reception chirp changes sign inside the fixture still equals the
        reference: rows of both chirp signs in one pass."""
        case = self._case(**self.SETTINGS[0])
        longitudes = np.linspace(0.0, 2.0 * np.pi, 128, endpoint=False)
        candidates = np.repeat(GB_PARAMS[None, :], longitudes.size, axis=0)
        candidates[:, 7] = longitudes
        chirp = case.segment_table(candidates)["chirp_rate"]
        changes = np.any(np.diff(np.sign(chirp), axis=1) != 0, axis=(1, 2))
        self.assertTrue(changes.any(), msg="no longitude of the scan changes the chirp sign")
        source = candidates[int(np.argmax(changes))][None, :]
        noise_index = np.zeros(1, dtype=np.int32)
        kernel = case.fisher(source, noise_index, inds=TEST_INDS)
        self.assertTrue(np.all(np.isfinite(kernel)))
        self._assert_parity(kernel, case.reference(source, noise_index, inds=TEST_INDS))

    def test_reception_invariant_parameters(self):
        """A, phi0, iota and psi, which GBDFFRSource declares, leave every reception state's
        effective frequency, chirp rate and band start unchanged bit for bit; the other five
        parameters move the effective frequencies."""
        case = self._case(**self.SETTINGS[0])
        shifts = np.array([1e-24, 1e-9, 1e-19, 1e-26, 0.1, 0.1, 0.1, 0.1, 0.1])
        reference = case.segment_table(GB_PARAMS)
        for parameter in range(9):
            shifted = GB_PARAMS.copy()
            shifted[parameter] += shifts[parameter]
            table = case.segment_table(shifted)
            with self.subTest(parameter=parameter):
                if parameter in (0, 4, 5, 6):
                    for name in ("effective_frequency", "chirp_rate", "start_freq_ind"):
                        np.testing.assert_array_equal(table[name], reference[name])
                else:
                    self.assertFalse(
                        np.array_equal(
                            table["effective_frequency"], reference["effective_frequency"]
                        )
                    )

    def test_rejects_bad_arguments(self):
        case = self._case(**self.SETTINGS[0])
        params, noise_index = self._sources()
        with self.assertRaises(ValueError):
            case.fisher(params, noise_index, inds=[0, 9])
        # * The entry point checks the index itself, and its rows per pass.
        with self.assertRaises(ValueError):
            case.entry_information_matrix(
                params, self._steps(case), inds=[0, 9], noise_index=noise_index
            )
        with self.assertRaises(ValueError):
            case.entry_information_matrix(
                params, self._steps(case), noise_index=noise_index, rows_per_pass=6
            )


class _AETGroup:
    """A group of another TDI set, as a construction-time check sees it."""

    tdi_type = "AET"


@unittest.skipUnless(HAVE_STFT_GB, "requires the GBGPU STFT-GB build")
class DFFRArgumentTest(unittest.TestCase):
    """Setups DFFR cannot represent raise instead of returning wrong values."""

    def _comp(self, **options):
        return STFTGBComputations(
            stft_comps=options.pop("stft_comps", None),
            T=NUM_SAMPLES * SAMPLE_DT,
            force_backend="cpu",
            response="dffr",
            **options,
        )

    def test_unsupported_setups_raise(self):
        """The checks of the entry points and of the kernels, reached through the comp."""
        with self.assertRaises(ValueError):
            _DFFRCase("cpu").entry_swap(GB_PARAMS, GB_PARAMS_B, rows_per_pass=24)
        case = _DFFRCase("cpu")
        with self.assertRaises(ValueError):
            case.entry_get_ll(GB_PARAMS, knobs=(0.0, N_SIDE_BINS, 3, True, True))
        # ! A table of another grid would place every segment at the wrong time.
        settings = _settings("cpu")
        with self.assertRaises(ValueError):
            _DFFRCase("cpu", first_midpoint=settings.t0 + 3600.0).get_ll(GB_PARAMS)
        # * Bound after construction, an AET domain reaches the kernel's own check.
        with self.assertRaises(ValueError):
            _DFFRCase("cpu", tdi_type="AET").get_ll(GB_PARAMS)

    def test_construction_checks(self):
        custom = TDIConfig("2nd generation", force_backend="cpu").tdi_combinations[:-1]
        with self.assertRaises(ValueError):
            self._comp(tdi_config=TDIConfig(custom, force_backend="cpu"))
        with self.assertRaises(ValueError):
            self._comp(stft_comps=_AETGroup())
        comp = self._comp(tdi_config="2nd generation")
        with self.assertRaises(ValueError):
            comp.tdi_config = TDIConfig(custom, force_backend="cpu")
        # * The TDI-on-the-fly response takes any combination, as before.
        STFTGBComputations(
            stft_comps=None,
            T=1.0,
            force_backend="cpu",
            tdi_config=TDIConfig(custom, force_backend="cpu"),
            response="tdi_on_the_fly",
        )

    def test_group_without_kinematics_raises(self):
        case = _DFFRCase("cpu")
        bare = type("BareGroup", (), {})()
        bare.cpp_fresnel, bare.cpp_domain, bare.d_d = (
            case.shim.cpp_fresnel,
            case.shim.cpp_domain,
            None,
        )
        case.comp.stft_comps = bare
        with self.assertRaises(TypeError):
            case.comp.get_ll_stft(GB_PARAMS)


def _cuda_backend_name():
    """Name of a usable CUDA backend, or None."""
    try:
        import cupy

        import lisatools

        if cupy.cuda.runtime.getDeviceCount() < 1:
            return None
        for name in ("cuda13x", "cuda12x", "cuda11x"):
            if lisatools.has_backend(name):
                cupy.zeros(1)  # * a device that cannot allocate is not usable
                return name
    except Exception:
        return None
    return None


@unittest.skipUnless(HAVE_STFT_GB and _cuda_backend_name(), "requires a usable CUDA backend")
class DFFRGPUWiringTest(DFFRWiringTest):
    """The wiring test on the CUDA backend."""

    backend_name = "gpu"


@unittest.skipUnless(HAVE_STFT_GB and _cuda_backend_name(), "requires a usable CUDA backend")
class DFFRGPUTest(DFFRSelfConsistencyTest):
    """The same tests on the CUDA backend, and the CUDA values against the CPU values."""

    backend_name = "gpu"

    def test_gpu_matches_cpu(self):
        """Compact templates and inner products agree with the CPU, and the 16-row and 32-row
        tiles of swap and the F-statistic with each other, within GPU_RELATIVE_BOUND of their
        peaks."""
        gpu = self._case(window_alpha=0.5, tdi="2nd generation")
        cpu = _DFFRCase("cpu", window_alpha=0.5, tdi="2nd generation")
        pairs = {
            "compact template": (gpu.fill_template(GB_PARAMS)[0], cpu.fill_template(GB_PARAMS)[0]),
            "get_ll": (
                np.concatenate(gpu.get_ll(GB_PARAMS)),
                np.concatenate(cpu.get_ll(GB_PARAMS)),
            ),
            "swap": (gpu.swap(GB_PARAMS, GB_PARAMS_B)[4], cpu.swap(GB_PARAMS, GB_PARAMS_B)[4]),
            "swap rows": (
                gpu.entry_swap(GB_PARAMS, GB_PARAMS_B, 16)[4],
                gpu.entry_swap(GB_PARAMS, GB_PARAMS_B, 32)[4],
            ),
            "fstat rows": (gpu.entry_fstat(GB_PARAMS, 16)[1], gpu.entry_fstat(GB_PARAMS, 32)[1]),
        }
        for name, (first, second) in pairs.items():
            with self.subTest(quantity=name):
                self.assertLessEqual(
                    np.abs(first - second).max(), GPU_RELATIVE_BOUND * np.abs(second).max()
                )


@unittest.skipUnless(HAVE_STFT_GB and _cuda_backend_name(), "requires a usable CUDA backend")
class DFFRGPUFisherTest(unittest.TestCase):
    """The DFFR Fisher matrix on the CUDA backend: against the reference and the CPU, and the
    4-row tables against the 8-row ones, per entry relative to sqrt(F_ii F_jj), on the
    8 parameters of the global fit; and against the CPU on A and phi0 alone, whose variants share
    one set of reception states."""

    def _assert_close(self, first, second):
        diagonal = np.abs(np.diagonal(second, axis1=1, axis2=2))
        scale = np.sqrt(diagonal[:, :, None] * diagonal[:, None, :])
        self.assertLess(float(np.max(np.abs(first - second) / scale)), GPU_FISHER_BOUND)

    def test_gpu_matches_reference_and_cpu(self):
        params, noise_index = np.stack([GB_PARAMS, GB_PARAMS_B]), np.array([0, 1], dtype=np.int32)
        setting = DFFRFisherTest.SETTINGS[0]
        gpu = _DFFRFisherCase("gpu", **setting)
        cpu = _DFFRFisherCase("cpu", **setting)
        eps = cpu.comp._information_matrix_steps(9, None, None, 1e-6)[0]
        kernel = gpu.fisher(params, noise_index, inds=TEST_INDS)
        pairs = {
            "reference": (kernel, gpu.reference(params, noise_index, inds=TEST_INDS)),
            "cpu": (kernel, cpu.fisher(params, noise_index, inds=TEST_INDS)),
            "rows": (
                gpu.entry_information_matrix(
                    params, eps, inds=TEST_INDS, noise_index=noise_index, rows_per_pass=8
                ),
                gpu.entry_information_matrix(
                    params, eps, inds=TEST_INDS, noise_index=noise_index, rows_per_pass=4
                ),
            ),
            "one set of states": (
                gpu.fisher(params, noise_index, inds=[0, 4]),
                cpu.fisher(params, noise_index, inds=[0, 4]),
            ),
        }
        for name, (first, second) in pairs.items():
            with self.subTest(quantity=name):
                self._assert_close(first, second)


if __name__ == "__main__":
    unittest.main()
