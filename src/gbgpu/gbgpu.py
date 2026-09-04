from multiprocessing.sharedctypes import Value
import contextlib
import time
import warnings
import abc

import numpy as np

# import constants
from lisatools.utils.constants import YRSID_SI, C_SI
from .utils.citation import *
from .utils.parallelbase import GBGPUParallelModule

# try:
#     from lisatools.sensitivity import A1TDISens
#     tdi_available = True

# except (ModuleNotFoundError, ImportError) as e:
#     tdi_available = False
#     warnings.warn("tdi module not found. No sensitivity information will be included.")

from .utils.utility import *

from lisatools.detector import EqualArmlengthOrbits, Orbits, L1Orbits
from typing import Optional, Sequence, TypeVar, Union, Dict, Any

from gpubackendtools.parallelbase import ParallelModuleBase

tdi_channel_setup_map = {"XYZ": 1, "AET": 2, "AE": 3}
window_map = {"tukey": 1, "planck": 2} # None is mapped to 0

class GBGPUBase(GBGPUParallelModule, abc.ABC):
    """Generate Galactic Binary Waveforms

    This class generates galactic binary waveforms in the frequency domain,
    in the form of LISA TDI channels X, A, and E. It generates waveforms in batches.
    It can also provide injection signals and calculate likelihoods in batches.
    These batches are run on GPUs or CPUs. When CPUs are used, all available threads
    are leveraged with OpenMP. To adjust the available threads, use ``OMP_NUM_THREADS``
    environmental variable or :func:`gbgpu.utils.set_omp_num_threads`.

    This class can generate waveforms for two different types of GB sources:

        * Circular Galactic binaries
        * Circular Galactic binaries with an eccentric third body (inherited)

    Args:
        force_backend (str, optional): Change to general backend in use. Example options are ``'cpu'`` and ``'gpu'``.
            Default is ``None``. 

    Attributes:
        get_basis_tensors (obj): Cython function.
        GenWave (obj): Cython function.
        GenWaveThird (obj): Cython function.
        unpack_data_1 (obj): Cython function.
        XYZ (obj): Cython function.
        num_bin (int): Number of binaries in the current calculation.
        N_max (int): Maximum points in a waveform based on maximum harmonic mode considered.
        start_inds (list of 1D int xp.ndarray): Start indices into data stream array. q - N/2.
        df (double): Fourier bin spacing.
        X_out, A_out, E_out (1D complex xp.ndarrays): X, A, or E channel TDI templates.
            Each array is a 2D complex array
            of shape (number of points, number of binaries) that is flattened. These can be
            accessed in python with the properties ``X``, ``A``, ``E``.
        N (int): Last N value used.
        d_d (double): <d|d> term in the likelihood.

    """

    def __init__(self, orbits: Optional[Orbits | L1Orbits] = None, force_backend = None, t0 = None, flip_ref_phase=False):
        self.force_backend = force_backend
        GBGPUParallelModule.__init__(self, force_backend=self.force_backend)
        
        self.d_d = None
        
        if orbits is not None and not isinstance(orbits, (Orbits, L1Orbits)):
            raise ValueError("Expected an Orbits object or None")

        # The property setter substitutes EqualArmlengthOrbits when given None.
        self.orbits = orbits
        
        # `gpus` controls multi-GPU dispatch. `None` -> CPU mode (or single-GPU on a single-device system).
        self.gpus = None

        # absolute start time for spacecraft-position evaluation. ``t0`` arg
        # overrides; otherwise read the orbit's ``sc_t0`` (0.0 if absent).
        if t0 is None:
            t0_abs = getattr(self.orbits, 'sc_t0', 0.0)
        else:
            t0_abs = t0
        self.t0_abs = t0_abs

        # mojito: option to flip the reference phase in waveform generation.
        self.flip_ref_phase = flip_ref_phase


    @classmethod
    def supported_backends(cls):
        return cls.GPU_RECOMMENDED()

    # ---- CPU/GPU dispatch helpers ----
    # * These make the shared-memory backend device-agnostic. The C++ side
    # * already treats a negative device id as "leave the current device alone"
    # * (`if (inputs.device >= 0)` in SharedMemoryGBGPU.cu); the helpers below
    # * mirror that convention on the Python side, so every call site can invoke
    # * them unconditionally on numpy and on cupy alike.

    _NO_DEVICE = -1

    def _xp_has_device(self) -> bool:
        """True when the active backend exposes a CUDA device layer."""
        return hasattr(self.xp, "cuda")

    def _xp_get_device(self) -> int:
        """Current device id, or the no-device sentinel on CPU."""
        if self._xp_has_device():
            return self.xp.cuda.runtime.getDevice()
        return self._NO_DEVICE

    def _xp_set_device(self, device) -> None:
        """cuda.runtime.setDevice on GPU; no-op on CPU or for the sentinel."""
        if self._xp_has_device() and device is not None and device >= 0:
            self.xp.cuda.runtime.setDevice(device)

    def _xp_sync(self) -> None:
        """No-op on CPU; cuda.runtime.deviceSynchronize on GPU."""
        if self._xp_has_device():
            self.xp.cuda.runtime.deviceSynchronize()

    def _xp_device(self, device):
        """cuda.Device context manager on GPU; nullcontext on CPU or sentinel."""
        if self._xp_has_device() and device is not None and device >= 0:
            return self.xp.cuda.Device(device)
        return contextlib.nullcontext()

    def _device_iter(self):
        """Devices the dispatch loops iterate over.

        ``self.gpus is None`` yields a single pass carrying the no-device
        sentinel, which is how the CPU backend and a single unpinned GPU both
        reach the same kernel launch code.
        """
        return self.gpus if self.gpus is not None else [self._NO_DEVICE]

    def _require_shared_memory_backend(self, use_c_implementation, caller_name) -> None:
        """Reject an explicit opt-out where no Python fallback exists."""
        if not use_c_implementation:
            raise NotImplementedError(
                f"{caller_name} is only implemented on the shared-memory C++/CUDA "
                f"backend, so use_c_implementation=False has no alternative to fall "
                f"back to. Omit the argument or pass True."
            )

    def _store_response(self, response, tdi_channel_setup) -> None:
        """Publish run_wave output under the attribute matching the TDI setup.

        Exactly one of ``XYZf`` / ``AETf`` survives, so a stale array left by an
        earlier call with a different setup can never be read back.
        """
        if tdi_channel_setup == "XYZ":
            self.XYZf = response
            if hasattr(self, "AETf"):
                del self.AETf
        else:
            self.AETf = response
            if hasattr(self, "XYZf"):
                del self.XYZf

    def _intra_split_index(self, index_arr, data_splits, gpu, num_per_gpu):
        """Intra-shard row index for each entry of ``index_arr`` on shard ``gpu``.

        The per-GPU buffers hold their rows in ascending global order, so the
        intra-shard index is the *rank* of the row among all rows that
        ``data_splits`` assigns to ``gpu``. Exact for uneven contiguous splits
        (``np.array_split`` with ``n % ngpus != 0``), where the legacy
        ``% num_per_gpu`` block trick misindexes every shard after the first.
        The modulo fallback only remains for callers passing no
        ``data_splits`` (single-buffer path, where rank == global index).
        """
        xp = self.xp
        if data_splits is None:
            return (index_arr % num_per_gpu).astype(np.int32)
        shard_rows = xp.where(xp.asarray(data_splits) == gpu)[0]
        return xp.searchsorted(shard_rows, xp.asarray(index_arr)).astype(np.int32)

    def _check_one_entry_per_gpu(self, caller_name, **named_lists):
        """Require one array per GPU for every input indexed by GPU position.

        A single array cannot be resident on more than one device, so passing one
        array while running on several GPUs cannot be satisfied.
        """
        if self.gpus is None:
            return

        num_gpus = len(self.gpus)
        wrong = {
            name: len(entries)
            for name, entries in named_lists.items()
            if len(entries) != num_gpus
        }
        if wrong:
            detail = ", ".join(f"{name} has {count}" for name, count in wrong.items())
            raise ValueError(
                f"{caller_name} indexes these inputs by GPU position, so each needs "
                f"exactly one entry per GPU. Running on {num_gpus} GPU(s) "
                f"({self.gpus}) but {detail}."
            )
    
    @property
    def get_ll_func(self):
        """get_ll c func."""
        return getattr(self.backend, 'get_ll')

    @property
    def fill_global_func(self):
        """fill_global c func."""
        return getattr(self.backend, 'fill_global')

    @property
    def orbits(self) -> Orbits:
        """Orbits class."""
        return self._orbits

    @orbits.setter
    def orbits(self, orbits: Orbits) -> None:
        if orbits is None:
            self._orbits = EqualArmlengthOrbits()
        elif not isinstance(orbits, Orbits):
            raise ValueError(
                "Input orbits must be of type Orbits (from LISA Analysis Tools)"
            )
        else:
            self._orbits = orbits
        self.orbits._ensure_configured()

    @property
    def citation(self):
        """Get citations for this class"""
        return zenodo + cornish_fastb + robson_triple


    def run_wave(
        self,
        amp,
        f0,
        fdot,
        fddot,
        phi0,
        iota,
        psi,
        lam,
        beta,
        *args,
        N: Optional[int] = None,
        T: float = 4 * YRSID_SI,
        dt: float = 10.0,
        oversample: int = 1,
        tdi2: bool = False,
        tdi_channel_setup: str = "AE",
        use_c_implementation: bool = False,
        window: Optional[str] = None,
        window_alpha: float = 0.0
    ):
        """Create waveforms in batches.

        This call creates the TDI templates in batches.

        The parameters and code below are based on an implementation of Fast GB
        in the LISA Data Challenges' ``ldc`` package.

        This class can be inherited to build fast waveforms for systems
        with additional astrophysical effects.

        Args:
            amp (double or 1D double np.ndarray): Amplitude parameter.
            f0 (double or 1D double np.ndarray): Initial frequency of gravitational
                wave in Hz.
            fdot (double or 1D double np.ndarray): Initial time derivative of the
                frequency given as Hz/s.
            fddot (double or 1D double np.ndarray): Initial second derivative with
                respect to time of the frequency given in Hz/s^2.
            phi0 (double or 1D double np.ndarray): Initial phase angle of gravitational
                wave given in radians.
            iota (double or 1D double np.ndarray): Inclination of the Galactic binary
                orbit given in radians.
            psi (double or 1D double np.ndarray): Polarization angle of the Galactic
                binary orbit in radians.
            lam (double or 1D double np.ndarray): Ecliptic longitutude of the source
                given in radians.
            beta (double or 1D double np.ndarray): Ecliptic Latitude of the source
                given in radians. This is converted to the spherical polar angle.
            *args (tuple, optional): Flexible parameter to allow for a flexible
                number of argmuments when inherited by other classes.
                If running a circular Galactic binarys, ``args = ()``.
                If ``len(args) != 0``, then the inheriting class must have a
                ``prepare_additional_args`` method.
            N (int, optional): Number of points in waveform.
                This should be determined by the initial frequency, ``f0``. Default is ``None``.
                If ``None``, will use :func:`gbgpu.utils.utility.get_N` function to determine proper ``N``.
            T (double, optional): Observation time in seconds. Default is ``4 * YRSID_SI``.
            dt (double, optional): Observation cadence in seconds. Default is ``10.0`` seconds.
            oversample(int, optional): Oversampling factor compared to the determined ``N``
                value. Final N will be ``oversample * N``. This is only used if N is
                not provided. Default is ``1``.
            tdi2 (bool, optional): If ``True``, produce the TDI channels for TDI 2nd-generation.
                If ``False``, produce TDI 1st-generation. Technically, the current TDI computation
                is not valid for generic LISA orbits, which are dealth with with 2nd-generation TDI,
                only those with an "equal-arm length" condition. Default is ``False``.

            Raises:
                ValueError: Length of ``*args`` is not 0 or 5.

        """

        # get number of observation points and adjust T accordingly
        N_obs = int(T / dt)
        T = N_obs * dt
        
        # === MODIFIED: Check orbital data availability ===
        # For L1Orbits, use sc_t_base directly instead of t_base
        t_source = getattr(self.orbits, 'sc_t_base', self.orbits.t_base)
        orbit_t_min = t_source.min()
        orbit_t_max = t_source.max()
        
        if self.t0_abs < orbit_t_min or self.t0_abs + T > orbit_t_max: 
            raise ValueError(
                f"Observation time window [{self.t0_abs:.2f}, {self.t0_abs + T:.2f}] "
                f"exceeds orbital data range [{orbit_t_min:.2f}, {orbit_t_max:.2f}]"
            )

        # if given scalar parameters, make sure at least 1D
        amp = np.atleast_1d(amp)
        f0 = np.atleast_1d(f0)
        fdot = np.atleast_1d(fdot)
        fddot = np.atleast_1d(fddot)
        phi0 = np.atleast_1d(phi0)
        iota = np.atleast_1d(iota)
        psi = np.atleast_1d(psi)
        lam = np.atleast_1d(lam)
        beta = np.atleast_1d(beta)
        
        if self.flip_ref_phase:
            # if matching jaxgb, then we need to input - phi0
            phi0 = -phi0

        # if circular base
        if len(args) == 0:
            add_args = ()

        else:
            # add_args = self.prepare_additional_args(*args)
            add_args = getattr(self, "prepare_additional_args")(*args)

        # set N if it is not given based on timescales in the waveform
        if N is None:
            # N_temp = self.special_get_N(amp, f0, T, *args, oversample=oversample)
            N_temp = getattr(self, "special_get_N")(amp, f0, T, *args, oversample=oversample)
            
            if hasattr(N_temp.max(), "item"):
                N = int(N_temp.max().item())
            else:
                N = int(N_temp.max())

        self.N = N
        
        # setup window mapping
        if window is not None:
            try:
                window_type = window_map[window]
            except:
                raise KeyError(f"The window '{window}' is currently not supported. Only the 'tukey', 'planck', or no/rectangular windows are supported.")
        else:
            window_type = 0
            assert window_alpha == 0.0, "No/Rectangular window does not have a smoothing factor"
        
        # get spacecraft positions
        tm_rel = self.xp.linspace(0, T, num=N, endpoint=False)
        tm_abs = tm_rel + self.t0_abs
        Ps = self._spacecraft(tm_abs)
        Ps_arr = self.xp.array(Ps).flatten()
        
        # number of binaries is determined from length of amp array
        self.num_bin = num_bin = len(amp)

        # polar angle from ecliptic latitude
        theta = np.pi / 2 - beta

        # bin spacing
        self.df = df = 1 / T

        # instantiate GPU/CPU arrays

        # copy to GPU if needed
        amp = self.xp.asarray(amp.copy())
        f0 = self.xp.asarray(f0.copy())  # in mHz
        fdot = self.xp.asarray(fdot.copy())
        fddot = self.xp.asarray(fddot.copy())
        phi0 = self.xp.asarray(phi0.copy())
        iota = self.xp.asarray(iota.copy())
        psi = self.xp.asarray(psi.copy())
        lam = self.xp.asarray(lam.copy())
        theta = self.xp.asarray(theta.copy())
        
        cosiota = self.xp.cos(iota.copy())
        
        # figure out start inds
        # q_check = (f0 * T).astype(np.int32)
        # self.start_inds = (q_check - N / 2).astype(xp.int32)
        # self.start_inds = (q_check - N / 2).astype(xp.int32)

        cosiota = self.xp.cos(iota)

        # transfer frequency
        fstar = C_SI / (self.orbits.armlength * 2 * np.pi)
        
        if use_c_implementation:
            if tdi_channel_setup == "AE":
                nchannels = 2
            else:
                nchannels = 3

            response_out = self.xp.zeros((self.num_bin * nchannels * int(N),), dtype=complex)
            _start_inds = self.xp.zeros((self.num_bin,), dtype=np.int32)
            if self.num_bin == 0:
                raise ValueError("No binaries were provided (num_bin is 0).")


            tuple_in = (
                response_out,
                _start_inds,
                amp, f0, fdot, fddot, phi0, iota, psi, lam, theta,
                T, dt, N, num_bin, 
                tdi_channel_setup_map[tdi_channel_setup],
                Ps_arr, self.orbits.armlength, tdi2,
                window_type, window_alpha
            )
            self.backend.sharedmem.SharedMemoryWaveComp_wrap(*tuple_in)
            self.start_inds = _start_inds
            response_out = response_out.reshape(self.num_bin, nchannels, N)

            # setup waveforms for efficient GPU likelihood or global template building
            self._store_response(response_out, tdi_channel_setup)
            return

        cosps, sinps = self.xp.cos(2.0 * psi), self.xp.sin(2.0 * psi)

        Aplus = amp * (1.0 + cosiota * cosiota)
        Across = -2.0 * amp * cosiota

        DP = Aplus * cosps - 1.0j * Across * sinps
        DC = -Aplus * sinps - 1.0j * Across * cosps

        # sky location basis vectors
        sinth, costh = self.xp.sin(theta), self.xp.cos(theta)
        sinph, cosph = self.xp.sin(lam), self.xp.cos(lam)
        u = self.xp.array([costh * cosph, costh * sinph, -sinth]).T[:, None, :]
        v = self.xp.array([sinph, -cosph, self.xp.zeros_like(cosph)]).T[:, None, :]
        k = self.xp.array([-sinth * cosph, -sinth * sinph, -costh]).T[:, None, :]

        # polarization tensors
        eplus = self.xp.matmul(v.transpose(0, 2, 1), v) - self.xp.matmul(
            u.transpose(0, 2, 1), u
        )
        ecross = self.xp.matmul(u.transpose(0, 2, 1), v) + self.xp.matmul(
            v.transpose(0, 2, 1), u
        )

        # # time points evaluated
        # # tm = self.xp.linspace(0, T, num=N, endpoint=False)
        # tm = self.xp.linspace(t0_abs, t0_abs + T, num=N, endpoint=False)
        # # get the spacecraft positions from orbits
        # Ps = self._spacecraft(tm)
    
        # time domain information
        # === MODIFIED: Pass RELATIVE time to waveform construction ===
        Gs, q = self._construct_slow_part(
            T,
            self.orbits.armlength,
            Ps,
            tm_rel,
            f0,
            fdot,
            fddot,
            fstar,
            phi0,
            k,
            DP,
            DC,
            eplus,
            ecross,
            *add_args,
        )

        if window_type != 0:
            w_arr = self._get_window_array(window_type, window_alpha, N)
            for ij in Gs.keys():
                Gs[ij] =  Gs[ij] * w_arr[None,:]
        
        # transform to TDI observables
        XYZf, f_min = self._computeXYZ(T, Gs, f0, fdot, fddot, fstar, amp, q, tm_rel)
        # NOTE this means that the correct times are passed down to construct_slow_part and _computeXYZ
        # Only need to check if these arguments are used correctly
        # int32 to match the start indices the shared-memory kernel writes back.
        self.start_inds = self.xp.round(f_min / df).astype(self.xp.int32)
        fctr = 0.5 * T / N

        # adjust for TDI2 if needed
        if tdi2:
            omegaL = 2 * np.pi * f0 * (self.orbits.armlength / C_SI)
            tdi2_factor = 2.0j * self.xp.sin(2 * omegaL) * self.xp.exp(-2j * omegaL)
            fctr *= tdi2_factor

        if isinstance(fctr, (float, int)):
            fctr = self.xp.array([fctr])

        XYZf *= fctr[:, None, None]

        # * XYZ is what _computeXYZ produces natively, so it is handed straight
        # * through. AET costs a conversion and is only formed when requested.
        if tdi_channel_setup == "XYZ":
            response_out = XYZf
        else:
            # we do not care about T right now
            AETf = self.xp.asarray(
                AET(XYZf[:, 0], XYZf[:, 1], XYZf[:, 2])
            ).transpose(1, 0, 2)
            # Contiguous so the slice matches the 2-channel buffer the kernel returns.
            response_out = (
                self.xp.ascontiguousarray(AETf[:, :2])
                if tdi_channel_setup == "AE"
                else AETf
            )

        # setup waveforms for efficient GPU likelihood or global template building
        self._store_response(response_out, tdi_channel_setup)

    @property
    def X_out(self):
        """X channel."""
        return self.XYZf[:, 0].T.flatten()

    @property
    def Y_out(self):
        """Y channel."""
        return self.XYZf[:, 1].T.flatten()

    @property
    def Z_out(self):
        """Z channel."""
        return self.XYZf[:, 2].T.flatten()

    @property
    def A_out(self):
        """A channel."""
        return self.AETf[:, 0].T.flatten()

    @property
    def E_out(self):
        """E channel."""
        return self.AETf[:, 1].T.flatten()

    @property
    def T_out(self):
        """T channel."""
        if self.AETf.shape[1] == 2: # TODO: can this be removed?
            raise ValueError("Requesting T channel when AE was generated.")

        return self.AETf[:, 2].T.flatten()

    def _computeXYZ(self, T, Gs, f0, fdot, fddot, fstar, ampl, q, tm):
        """Compute TDI X, Y, Z from y_sr"""

        # get true frequency as a function of time
        f = (
            f0[:, None]
            + fdot[:, None] * tm[None, :]
            + 1 / 2 * fddot[:, None] * tm[None, :] ** 2
        )

        # compute transfer function
        omL = f / fstar
        SomL = self.xp.sin(omL)
        fctr = self.xp.exp(-1.0j * omL)
        fctr2 = 4.0 * omL * SomL * fctr / ampl[:, None]

        # Notes from LDC below

        ### I have factored out 1 - exp(1j*omL) and transformed to
        ### fractional frequency: those are in fctr2
        ### I have rremoved Ampl to reduce dynamical range, will restore it later

        Xsl = Gs["21"] - Gs["31"] + (Gs["12"] - Gs["13"]) * fctr
        Ysl = Gs["32"] - Gs["12"] + (Gs["23"] - Gs["21"]) * fctr
        Zsl = Gs["13"] - Gs["23"] + (Gs["31"] - Gs["32"]) * fctr

        # time domain slow part
        XYZsl = fctr2[:, None, :] * self.xp.array([Xsl, Ysl, Zsl]).transpose(1, 0, 2)

        # frequency domain slow part
        XYZf_slow = ampl[:, None, None] * self.xp.fft.fft(XYZsl, axis=-1)

        # for testing
        # Xtry = 4.0*(self.G21 - self.G31 + (self.G12 - self.G13)*fctr)/self.ampl

        M = XYZf_slow.shape[2]  # len(XYZf_slow)
        XYZf = self.xp.fft.fftshift(XYZf_slow, axes=-1)

        # closest bin frequency
        f0 = (q - M / 2) / T  # freq = (q + self.xp.arange(M) - M/2)/T
        return XYZf, f0

    def _spacecraft(self, t):
        """Compute space craft positions as a function of time"""

        # output arrays
        P1 = self.orbits.get_pos(t, self.xp.full_like(t, 1).astype(self.xp.int32))
        P2 = self.orbits.get_pos(t, self.xp.full_like(t, 2).astype(self.xp.int32))
        P3 = self.orbits.get_pos(t, self.xp.full_like(t, 3).astype(self.xp.int32))
        return [P1, P2, P3]

    def _construct_slow_part(
        self,
        T,
        arm_length,
        Ps,
        tm,
        f0,
        fdot,
        fddot,
        fstar,
        phi0,
        k,
        DP,
        DC,
        eplus,
        ecross,
        *add_args,
    ):
        """Construct the time-domain function for the slow part of the waveform."""

        # these are the orbits (equal-arm lengths assumed)
        P1, P2, P3 = Ps
        r = dict()

        # unit vectors of constellation arms
        r["12"] = (P2 - P1) / arm_length  ## [3xNt]
        r["13"] = (P3 - P1) / arm_length
        r["23"] = (P3 - P2) / arm_length
        r["31"] = -r["13"]

        # wave propagation axis dotted with constellation unit vectors
        kdotr = dict()
        for ij in ["12", "13", "23"]:
            kdotr[ij] = self.xp.dot(k.squeeze(), r[ij].T)  ### should be size Nt
            kdotr[ij[-1] + ij[0]] = -kdotr[ij]

        # wave propagation axis dotted with spacecraft positions
        kdotP = self.xp.array(
            [self.xp.dot(k, P1.T), self.xp.dot(k, P2.T), self.xp.dot(k, P3.T)]
        )[:, :, 0].transpose(1, 0, 2)

        kdotP /= C_SI

        Nt = len(tm)

        # delayed time at the spacecraft
        xi = tm - kdotP

        # instantaneous frequency of wave at the spacecraft at xi
        fi = (
            f0[:, None, None]
            + fdot[:, None, None] * xi
            + 1 / 2.0 * fddot[:, None, None] * xi**2
        )

        # for regular GBGPU shift is zero
        # fi[:] = self.shift_frequency(fi, xi, *add_args)
        fi[:] = getattr(self, "shift_frequency")(fi, xi, *add_args)

        # transfer frequency ratio
        fonfs = fi / fstar  # Ratio of true frequency to transfer frequency

        # LDC notes with '###'
        ### compute transfer f-n
        q = np.rint(f0 * T)  # index of nearest Fourier bin
        df = 2.0 * np.pi * (q / T)
        om = 2.0 * np.pi * f0

        ### The expressions below are arg2_i with om*kR_i factored out
        A = dict()
        for ij in ["12", "23", "31"]:
            aij = (
                self.xp.dot(eplus, r[ij].T) * r[ij].T * DP[:, None, None]
                + self.xp.dot(ecross, r[ij].T) * r[ij].T * DC[:, None, None]
            )
            A[ij] = aij.sum(axis=1)

        # below is information from the LDC about matching the original LDC.
        # The current code matches the time-domain-generated tempaltes in the LDC.

        # These are wfm->TR + 1j*TI in c-code

        # arg2_1 = 2.0*np.pi*f0*xi[0] + phi0 - df*tm + np.pi*fdot*(xi[0]**2)
        # arg2_2 = 2.0*np.pi*f0*xi[1] + phi0 - df*tm + np.pi*fdot*(xi[1]**2)
        # arg2_3 = 2.0*np.pi*f0*xi[2] + phi0 - df*tm + np.pi*fdot*(xi[2]**2)

        ### These (y_sr) reproduce exactly the FastGB results
        # self.y12 = 0.25*np.sin(arg12)/arg12 * np.exp(1.j*(arg12 + arg2_1)) * ( Dp12*self.DP + Dc12*self.DC )
        # self.y23 = 0.25*np.sin(arg23)/arg23 * np.exp(1.j*(arg23 + arg2_2)) * ( Dp23*self.DP + Dc23*self.DC )
        # self.y31 = 0.25*np.sin(arg31)/arg31 * np.exp(1.j*(arg31 + arg2_3)) * ( Dp31*self.DP + Dc31*self.DC )
        # self.y21 = 0.25*np.sin(arg21)/arg21 * np.exp(1.j*(arg21 + arg2_2)) * ( Dp12*self.DP + Dc12*self.DC )
        # self.y32 = 0.25*np.sin(arg32)/arg32 * np.exp(1.j*(arg32 + arg2_3)) * ( Dp23*self.DP + Dc23*self.DC )
        # self.y13 = 0.25*np.sin(arg13)/arg13 * np.exp(1.j*(arg13 + arg2_1)) * ( Dp31*self.DP + Dc31*self.DC )

        ### Those are corrected values which match the time domain results.
        ## om*kdotP_i singed out for comparison with another code.

        argS = (
            phi0[:, None, None]
            + (om[:, None, None] - df[:, None, None]) * tm[None, None, :]
            + np.pi * fdot[:, None, None] * (xi**2)
            + 1 / 3 * np.pi * fddot[:, None, None] * (xi**3)
        )

        # called kdotP in LDC code
        arg_phasing = om[:, None, None] * kdotP - argS
        # arg_phasing[:] = self.add_to_phasing(arg_phasing, f0, fdot, fddot, xi, *add_args)
        arg_phasing[:] = getattr(self, "add_to_phasing")(arg_phasing, f0, fdot, fddot, xi, *add_args)
        
        # get Gs transfer functions
        Gs = dict()
        for ij, ij_sym, s in [
            ("12", "12", 0),
            ("23", "23", 1),
            ("31", "31", 2),
            ("21", "12", 1),
            ("32", "23", 2),
            ("13", "31", 0),
        ]:

            # TODO: evolution of the amplitude
            arg_ij = 0.5 * fonfs[:, s, :] * (1 + kdotr[ij])
            Gs[ij] = (
                0.25
                * self.xp.sin(arg_ij)
                / arg_ij
                * self.xp.exp(-1.0j * (arg_ij + arg_phasing[:, s]))
                * A[ij_sym]
            )
        ### Lines blow are extractions from another python code and from C-code in LDC
        # y = -0.5j*self.omL*A*sinc(args)*np.exp(-1.0j*(args + self.om*kq))
        # args = 0.5*self.omL*(1.0 - kn)
        # arg12 = 0.5*fonfs[0,:] * (1 + kdotr12)
        # arg2_1 = 2.0*np.pi*f0*xi[0] + phi0 - df*tm + np.pi*self.fdot*(xi[0]**2)  -> om*k.Ri
        # arg1 = 0.5*wfm->fonfs[i]*(1. + wfm->kdotr[i][j])
        # arg2 =  PI*2*f0*wfm->xi[i] + phi0 - df*t
        # sinc = 0.25*sin(arg1)/arg1
        # tran1r = aevol*(wfm->dplus[i][j]*wfm->DPr + wfm->dcross[i][j]*wfm->DCr)
        # tran1i = aevol*(wfm->dplus[i][j]*wfm->DPi + wfm->dcross[i][j]*wfm->DCi)
        # tran2r = cos(arg1 + arg2)
        # tran2i = sin(arg1 + arg2)
        # wfm->TR[i][j] = sinc*(tran1r*tran2r - tran1i*tran2i)
        # wfm->TI[i][j] = sinc*(tran1r*tran2i + tran1i*tran2r)
        return Gs, q

    def _get_window_array(self, window_type: int, window_alpha: float, N: int) -> Any:
        """Generate window array"""
        if window_type == 0 or window_alpha <= 0.0 or N <= 1:
            return self.xp.ones(N, dtype=self.xp.float64)
        
        i = self.xp.arange(N)
        if window_type == 1:  # Tukey
            alpha = window_alpha
            w = self.xp.zeros(N)
            r = (2.0 * i) / (alpha * (N - 1))
            l1 = int(self.xp.rint((alpha * (N - 1)) / 2.0))
            l2 = int(self.xp.rint((N - 1) * (1.0 - alpha / 2.0)))
            
            mask1 = i < l1
            w = self.xp.where(mask1, 0.5 * (1.0 + self.xp.cos(np.pi * (r - 1.0))), w)
            mask2 = (i >= l1) & (i < l2)
            w = self.xp.where(mask2, 1.0, w)
            mask3 = i >= l2
            w = self.xp.where(mask3, 0.5 * (1.0 + self.xp.cos(np.pi * (r - 2.0 / alpha + 1.0))), w)
            return w

        elif window_type == 2:  # Planck
            epsilon = window_alpha
            w = self.xp.zeros(N)
            n1 = epsilon * (N - 1)
            n2 = (1.0 - epsilon) * (N - 1)

            mask1 = i < n1
            z1 = self.xp.where(mask1, epsilon * (N - 1) / (i + 1e-15) + epsilon * (N - 1) / ((i - epsilon * (N - 1)) + 1e-15), 0.0)
            w = self.xp.where(mask1, 1.0 / (1.0 + self.xp.exp(self.xp.clip(z1, -700.0, 700.0))), w)

            mask2 = (i >= n1) & (i <= n2)
            w = self.xp.where(mask2, 1.0, w)

            mask3 = i > n2
            z2 = self.xp.where(mask3, epsilon * (N - 1) / ((N - 1) - i + 1e-15) + epsilon * (N - 1) / ((N - 1) - i - epsilon * (N - 1) + 1e-15), 0.0)
            w = self.xp.where(mask3, 1.0 / (1.0 + self.xp.exp(self.xp.clip(z2, -700.0, 700.0))), w)
            return w
        
        else:
            raise ValueError("Window type not supported. Supported types are 'None': 0 (retangular/no window), 'tukey': 1 (tukey window) 'planck': 2 (planck window).")

    
    @property
    def A(self):
        """return A channel reshaped based on number of binaries"""
        return self.AETf[:, 0]

    @property
    def E(self):
        """return E channel reshaped based on number of binaries"""
        return self.AETf[:, 1]

    @property
    def T(self):
        """return T channel reshaped based on number of binaries"""
        if self.AETf.shape[1] == 2:
            raise ValueError("Requesting T channel when AE was generated.")
        return self.AETf[:, 2]

    @property
    def X(self):
        """return X channel reshaped based on number of binaries"""
        return self.X_out.reshape(self.N, self.num_bin).T

    @property
    def Y(self):
        """return Y channel reshaped based on number of binaries"""
        return self.Y_out.reshape(self.N, self.num_bin).T

    @property
    def Z(self):
        """return Z channel reshaped based on number of binaries"""
        return self.Z_out.reshape(self.N, self.num_bin).T

    @property
    def freqs(self):
        """Return frequencies associated with each signal"""
        freqs_out = (
            self.xp.arange(self.N)[None, :] + self.start_inds[:, None]
        ) * self.df
        return freqs_out

    def get_ll(
        self,
        params,
        data_minus_template,
        psd,
        start_freq_ind=0,
        data_index=None,
        noise_index=None,
        adjust_inplace=False,
        use_c_implementation=True,
        N=None,
        T=4 * YRSID_SI,
        dt=10.0,
        data_length=None,
        data_splits=None,
        phase_maximize=False,
        tdi_channel_setup="AE",
        num_per_gpu=None,
        oversample=1,
        return_cupy=False,
        tdi2: bool = False,
        window: Optional[str] = None,
        window_alpha: float = 0.0
        # **kwargs
    ):
        if self.d_d is None:
            raise ValueError(
                "self.d_d attribute must be set before computing log-Likelihood. This attribute is the data with data inner product (<d|d>)."
            )

        self._require_shared_memory_backend(use_c_implementation, "get_ll")

        if num_per_gpu is not None:
            raise NotImplementedError("Need to check this.")

        if self.gpus is not None:
            # set first index gpu device to control main operations
            return_to_main_device = self.gpus[0]
            self._xp_set_device(return_to_main_device)

        self.num_bin = num_bin = params.shape[0]

        if self.flip_ref_phase:
            # if matching jaxgb, then we need to input - phi0
            params = params.copy()
            params[:, 4] = -params[:, 4]

        if N is None:
            # TODO: G
            N = get_N(self.xp.asarray(params[:, 0]), self.xp.asarray(params[:, 1]), T, oversample=oversample, armlength=self.orbits.armlength)
            if self.xp.any(N == 0):
                raise ValueError("N contains zeros.")
        else:
            if isinstance(N, self.xp.ndarray):
                assert params.shape[0] == N.shape[0]
            elif isinstance(N, (int, np.integer)):
                N = self.xp.full(params.shape[0], N)

        unique_N, inverse = self.xp.unique(self.xp.asarray(N), return_inverse=True)
        N_groups = self.xp.arange(len(unique_N))[inverse]

        # setup window mapping
        if window is not None:
            try:
                window_type = window_map[window]
            except:
                raise KeyError(f"The window '{window}' is currently not supported. Only the 'tukey', 'planck', or no/rectangular windows are supported.")
        else:
            window_type = 0
            assert window_alpha == 0.0, "No/Rectangular window does not have a smoothing factor"
        
        # fill index values if not given
        if data_index is None:
            data_index = self.xp.zeros(self.num_bin, dtype=self.xp.int32)
        if noise_index is None:
            noise_index = self.xp.zeros(self.num_bin, dtype=self.xp.int32)
        
        # check that index values are ready for computation
        assert data_index.dtype == self.xp.int32
        assert noise_index.dtype == self.xp.int32

        nchannels = 3 if tdi_channel_setup != "AE" else 2

        if isinstance(data_minus_template, self.xp.ndarray):
            data_minus_template_in = [data_minus_template.astype(self.xp.complex128)]
        else:
            data_minus_template_in = [tmp.astype(self.xp.complex128) for tmp in data_minus_template]

        if isinstance(psd, self.xp.ndarray):
            psd_in = [psd.astype(self.xp.complex128)]
        else:
            psd_in = [p.astype(self.xp.complex128) for p in psd]

        num_data = []
        for t_i, t in enumerate(data_minus_template_in):
            if t.ndim == 1:
                assert data_length is not None
                assert isinstance(data_length, int)
                num_data.append(int(t.shape[0] / (data_length * nchannels)))
                
            elif t.ndim == 2:
                num_data.append(1)
                assert t.shape[0] == nchannels
                data_length = t.shape[1]
                # TODO: print("check this does not create memory issues")
                data_minus_template_in[t_i] = t.flatten()

            else:
                ntemplate, _nchannels, data_length = t.shape
                num_data.append(ntemplate)
                assert _nchannels == nchannels
                # TODO: print("check this does not create memory issues")
                
                data_minus_template_in[t_i] = t.flatten()

        if tdi_channel_setup == "AE" or tdi_channel_setup == "AET":
            # assumes nchannels will 1D really
            psd_sub_shape = (nchannels,)
            
        else:
            assert nchannels == 3
            psd_sub_shape = (nchannels, nchannels)

        num_psd = []
        for t_i, t in enumerate(psd_in):
            if t.ndim == 1:
                assert data_length is not None
                assert isinstance(data_length, int)
                num_psd.append(int(t.shape[0] / (data_length * np.prod(psd_sub_shape))))
                
            elif t.ndim == 2:
                assert tdi_channel_setup in ["AET", "AE"]
                num_psd.append(1)
                assert t.shape[0] == nchannels
                data_length = t.shape[1]
                # print("check this does not create memory issues")
                psd_in[t_i] = t.flatten()

            elif t.ndim == 3:
                if tdi_channel_setup in ["AE", "AET"]:
                    ntemplate, _nchannels, data_length = t.shape
                    num_psd.append(ntemplate)
                    assert t.shape[1] == nchannels
                    data_length = t.shape[2]
                    # print("check this does not create memory issues")
                    psd_in[t_i] = t.flatten()


                else:  # XYZ
                    _nchannels, _nchannels, data_length = t.shape
                    num_psd.append(1)
                    assert t.shape[1] == t.shape[0] == nchannels
                    data_length = t.shape[2]
                    # print("check this does not create memory issues")
                    psd_in[t_i] = t.flatten()
                assert _nchannels == nchannels
                
            elif t.ndim == 4:
                assert tdi_channel_setup == "XYZ"
                ntemplate, _nchannels, _nchannels, data_length = t.shape
                num_psd.append(ntemplate)
                assert t.shape[1] == t.shape[2] == nchannels
                data_length = t.shape[3]
                # print("check this does not create memory issues")
                psd_in[t_i] = t.flatten()
            
                assert _nchannels == nchannels
            # print("check this does not create memory issues")
        
        self._check_one_entry_per_gpu(
            "get_ll", data=data_minus_template_in, psd=psd_in
        )

        # initialize Likelihood terms <d|h> and <h|h>
        d_h = self.xp.zeros(self.num_bin, dtype=self.xp.complex128)
        h_h = self.xp.zeros(self.num_bin, dtype=self.xp.complex128)
        
        do_synchronize = False
        main_device = self._xp_get_device()
        devices = self._device_iter()
        if data_splits is None:
            assert len(devices) == 1
            data_splits = self.xp.full(num_data[0], devices[0])

        if num_per_gpu is None:
            assert len(devices) == 1
            num_per_gpu = int(2**31 - 1)
            # make really high so just keeps (int32-safe: numpy 2 rejects
            # int32 arrays modulo a Python int beyond the int32 range)

        inputs_in = []
        for nnn, N_here in enumerate(unique_N):
            N_here = N_here.item()
            # get spacecraft positions (mojito: real-orbit Ps_arr threaded
            # into the kernels for mojito-orbit / TDI2 aware computation).
            tm_rel = self.xp.linspace(0, T, num=N_here, endpoint=False)
            tm_abs = tm_rel + self.t0_abs
            Ps_arr = self.xp.array(self._spacecraft(tm_abs)).flatten()

            for gpu_i, gpu in enumerate(devices):
                self._xp_set_device(main_device)
                keep_bool = (N_groups == nnn) & (self.xp.asarray(data_splits)[data_index] == gpu)
                num_split_here = keep_bool.sum().item()
                inds_here = self.xp.arange(len(keep_bool))[keep_bool]
                if num_split_here == 0:
                    continue
                self._xp_set_device(gpu)

                params_here = self.xp.asarray(params)[keep_bool]
                
                # theta_add = np.pi / 2 - beta_add
                params_here[:, 8] = np.pi / 2 - params_here[:, 8]
                
                data_minus_template_here = data_minus_template_in[gpu_i]
                psd_here = psd_in[gpu_i]

                params_tuple = tuple([pars_tmp.copy()for pars_tmp in params_here.T])
                
                if isinstance(start_freq_ind, int):
                    start_freq_ind_tmp = self.xp.full(num_data[gpu_i], start_freq_ind, dtype=np.int32)
                
                else:
                    assert isinstance(start_freq_ind, self.xp.ndarray) and start_freq_ind.dtype == np.int32
                    # TODO: fix this num_data
                    
                    start_freq_ind_tmp = start_freq_ind
                assert len(start_freq_ind_tmp) == num_data[gpu_i]

                d_h_temp = self.xp.zeros(num_split_here, dtype=self.xp.complex128)
                h_h_temp = self.xp.zeros(num_split_here, dtype=self.xp.complex128)
                
                noise_index_in = self.xp.asarray(noise_index[keep_bool] % num_per_gpu).astype(np.int32)
                data_index_in = self.xp.asarray(data_index[keep_bool] % num_per_gpu).astype(np.int32)
        
                tuple_in = (
                    (
                        d_h_temp,
                        h_h_temp,
                        data_minus_template_here,
                        psd_here,
                        data_index_in,
                        noise_index_in,
                    )
                    + params_tuple
                    + (
                        T, dt, N_here, 
                        num_split_here, start_freq_ind_tmp, data_length, 
                        tdi_channel_setup_map[tdi_channel_setup], 
                        gpu, do_synchronize,
                        num_data[gpu_i], num_psd[gpu_i],
                        Ps_arr, self.orbits.armlength, tdi2,
                        window_type, window_alpha
                    )
                )

                self._xp_sync()
                self.backend.sharedmem.SharedMemoryLikeComp_wrap(*tuple_in)
                inputs_in.append([gpu, inds_here, tuple_in])
                self._xp_sync()

        for gpu, inds_gpu, inputs_tmp in inputs_in:
            with self._xp_device(gpu):
                self._xp_sync()

        for gpu, inds_gpu, inputs_tmp in inputs_in:
            with self._xp_device(main_device):
                self._xp_sync()

                d_h[inds_gpu] = inputs_tmp[0][:]
                h_h[inds_gpu] = inputs_tmp[1][:]
                
        self._xp_set_device(main_device)
        self._xp_sync()
        
        if phase_maximize:
            self.non_marg_d_h = d_h.copy()
            # Maximising rotation (matches swap_likelihood_difference's
            # convention): d_h * exp(-1j * phase_angle) is real-positive.
            # Callers subtract this from the sampling-basis phi0 on accept.
            self.phase_angle = self.xp.arctan2(d_h.imag, d_h.real)
            try:
                self.non_marg_d_h = self.non_marg_d_h.get()
            except AttributeError:
                pass

            d_h = self.xp.abs(d_h)
        else:
            self.phase_angle = None

        # store these likelihood terms for later if needed
        self.h_h = h_h
        self.d_h = d_h

        # compute Likelihood
        like_out = -1.0 / 2.0 * (self.d_d + h_h - 2 * d_h).real

        if return_cupy:
            return like_out

        # back to CPU if on GPU
        try:
            return like_out.get()

        except AttributeError:
            return like_out


    def get_fstat_ll(
        self,
        params,
        data_minus_template,
        psd,
        start_freq_ind=0,
        data_index=None,
        noise_index=None,
        adjust_inplace=False,
        use_c_implementation=True,
        N=None,
        T=4 * YRSID_SI,
        dt=10.0,
        data_length=None,
        data_splits=None,
        phase_maximize=False,
        tdi_channel_setup="AE",
        num_per_gpu=None,
        oversample=1,
        return_cupy=False,
        tdi2: bool = False,
        window: Optional[str] = None,
        window_alpha: float = 0.0
        # **kwargs
    ):
        self._require_shared_memory_backend(use_c_implementation, "get_fstat_ll")

        if num_per_gpu is not None:
            raise NotImplementedError("Need to check this.")

        if self.gpus is not None:
            # set first index gpu device to control main operations
            return_to_main_device = self.gpus[0]
            self._xp_set_device(return_to_main_device)

        self.num_bin = params.shape[0]

        if N is None:
            # TODO: G
            #params are different for fstat
            N = get_N(self.xp.full_like(params[:, 0], 1e-25), self.xp.asarray(params[:, 0]), T, oversample=oversample, armlength=self.orbits.armlength)
            if self.xp.any(N == 0):
                raise ValueError("N contains zeros.")
        else:
            if isinstance(N, self.xp.ndarray):
                assert params.shape[0] == N.shape[0]
            elif isinstance(N, (int, np.integer)):
                N = self.xp.full(params.shape[0], N)

        unique_N, inverse = self.xp.unique(self.xp.asarray(N), return_inverse=True)
        N_groups = self.xp.arange(len(unique_N))[inverse]

        # setup window mapping
        if window is not None:
            try:
                window_type = window_map[window]
            except:
                raise KeyError(f"The window '{window}' is currently not supported. Only the 'tukey', 'planck', or no/rectangular windows are supported.")
        else:
            window_type = 0
            assert window_alpha == 0.0, "No/Rectangular window does not have a smoothing factor"
        
        # check that index values are ready for computation
        if data_index is None:
            data_index = self.xp.zeros(self.num_bin, dtype=self.xp.int32)
        
        if noise_index is None:
            noise_index = self.xp.zeros(self.num_bin, dtype=self.xp.int32)
        
        assert data_index.dtype == self.xp.int32
        assert noise_index.dtype == self.xp.int32
        
        nchannels = 3 if tdi_channel_setup != "AE" else 2

        if isinstance(data_minus_template, self.xp.ndarray):
            data_minus_template_in = [data_minus_template.astype(self.xp.complex128)]
        else:
            data_minus_template_in = [tmp.astype(self.xp.complex128) for tmp in data_minus_template]

        if isinstance(psd, self.xp.ndarray):
            psd_in = [psd.astype(self.xp.complex128)]
        else:
            psd_in = [p.astype(self.xp.complex128) for p in psd]

        num_data = []
        for t_i, t in enumerate(data_minus_template_in):
            if t.ndim == 1:
                assert data_length is not None
                assert isinstance(data_length, int)
                num_data.append(int(t.shape[0] / (data_length * nchannels)))
                
            elif t.ndim == 2:
                num_data.append(1)
                assert t.shape[0] == nchannels
                data_length = t.shape[1]
                # TODO: print("check this does not create memory issues")
                data_minus_template_in[t_i] = t.flatten()

            else:
                ntemplate, _nchannels, data_length = t.shape
                num_data.append(ntemplate)
                assert _nchannels == nchannels
                # TODO: print("check this does not create memory issues")
                
                data_minus_template_in[t_i] = t.flatten()

        if tdi_channel_setup == "AE" or tdi_channel_setup == "AET":
            # assumes nchannels will 1D really
            psd_sub_shape = (nchannels,)
            
        else:
            assert nchannels == 3
            psd_sub_shape = (nchannels, nchannels)

        num_psd = []
        for t_i, t in enumerate(psd_in):
            if t.ndim == 1:
                assert data_length is not None
                assert isinstance(data_length, int)
                num_psd.append(int(t.shape[0] / (data_length * np.prod(psd_sub_shape))))
                
            elif t.ndim == 2:
                assert tdi_channel_setup in ["AET", "AE"]
                num_psd.append(1)
                assert t.shape[0] == nchannels
                data_length = t.shape[1]
                # print("check this does not create memory issues")
                psd_in[t_i] = t.flatten()

            elif t.ndim == 3:
                if tdi_channel_setup in ["AE", "AET"]:
                    ntemplate, _nchannels, data_length = t.shape
                    num_psd.append(ntemplate)
                    assert t.shape[1] == nchannels
                    data_length = t.shape[2]
                    # print("check this does not create memory issues")
                    psd_in[t_i] = t.flatten()


                else:  # XYZ
                    _nchannels, _nchannels, data_length = t.shape
                    num_psd.append(1)
                    assert t.shape[1] == t.shape[0] == nchannels
                    data_length = t.shape[2]
                    # print("check this does not create memory issues")
                    psd_in[t_i] = t.flatten()
                assert _nchannels == nchannels
                
            elif t.ndim == 4:
                assert tdi_channel_setup == "XYZ"
                ntemplate, _nchannels, _nchannels, data_length = t.shape
                num_psd.append(ntemplate)
                assert t.shape[1] == t.shape[2] == nchannels
                data_length = t.shape[3]
                # print("check this does not create memory issues")
                psd_in[t_i] = t.flatten()
            
                assert _nchannels == nchannels
            # print("check this does not create memory issues")
        
        self._check_one_entry_per_gpu(
            "get_fstat_ll", data=data_minus_template_in, psd=psd_in
        )

        # initialize Likelihood terms <d|h> and <h|h>
        M_mat = self.xp.zeros((self.num_bin, 4, 4))
        N_arr = self.xp.zeros((self.num_bin, 4))
        
        do_synchronize = False
        main_device = self._xp_get_device()
        devices = self._device_iter()
        if data_splits is None:
            assert len(devices) == 1
            data_splits = self.xp.full(num_data[0], devices[0])

        if num_per_gpu is None:
            assert len(devices) == 1
            num_per_gpu = int(2**31 - 1)
            # make really high so just keeps (int32-safe: numpy 2 rejects
            # int32 arrays modulo a Python int beyond the int32 range)

        inputs_in = []
        for nnn, N_here in enumerate(unique_N):
            N_here = N_here.item()
            # get spacecraft positions
            tm_rel = self.xp.linspace(0, T, num=N_here, endpoint=False)
            tm_abs = tm_rel + self.t0_abs
            Ps_arr = self.xp.array(self._spacecraft(tm_abs)).flatten()

            for gpu_i, gpu in enumerate(devices):
                self._xp_set_device(main_device)
                keep_bool = (N_groups == nnn) & (self.xp.asarray(data_splits)[data_index] == gpu)
                num_split_here = keep_bool.sum().item()
                inds_here = self.xp.arange(len(keep_bool))[keep_bool]
                if num_split_here == 0:
                    continue
                self._xp_set_device(gpu)

                params_here = self.xp.asarray(params)[keep_bool]

                # theta_add = np.pi / 2 - beta_add
                params_here[:, -1] = np.pi / 2 - params_here[:, -1]
                
                data_minus_template_here = data_minus_template_in[gpu_i]
                psd_here = psd_in[gpu_i]

                params_tuple = tuple([pars_tmp.copy()for pars_tmp in params_here.T])
                
                if isinstance(start_freq_ind, int):
                    start_freq_ind_tmp = self.xp.full(num_data[gpu_i], start_freq_ind, dtype=np.int32)
                
                else:
                    assert isinstance(start_freq_ind, self.xp.ndarray) and start_freq_ind.dtype == np.int32
                    # TODO: fix this num_data
                    start_freq_ind_tmp = start_freq_ind
                assert len(start_freq_ind_tmp) == num_data[gpu_i]
                    
                M_mat_temp = self.xp.zeros(num_split_here * 4 * 4, dtype=self.xp.complex128)
                N_arr_temp = self.xp.zeros(num_split_here * 4, dtype=self.xp.complex128)
                
                noise_index_in = self.xp.asarray(noise_index[keep_bool] % num_per_gpu).astype(np.int32)
                data_index_in = self.xp.asarray(data_index[keep_bool] % num_per_gpu).astype(np.int32)
                    
                tuple_in = (
                    (
                        M_mat_temp,
                        N_arr_temp,
                        data_minus_template_here,
                        psd_here,
                        data_index_in,
                        noise_index_in,
                    ) + params_tuple
                    + (
                        T, dt, N_here, 
                        num_split_here, start_freq_ind_tmp, data_length, 
                        tdi_channel_setup_map[tdi_channel_setup], 
                        gpu, do_synchronize,
                        num_data[gpu_i], num_psd[gpu_i],
                        Ps_arr, self.orbits.armlength, tdi2,
                        window_type, window_alpha
                    )
                )

                self._xp_sync()
                self.backend.sharedmem.SharedMemoryFstatLikeComp_wrap(*tuple_in)
                inputs_in.append([gpu, inds_here, tuple_in])
                self._xp_sync()

        for gpu, inds_gpu, inputs_tmp in inputs_in:
            with self._xp_device(gpu):
                self._xp_sync()

        for gpu, inds_gpu, inputs_tmp in inputs_in:
            with self._xp_device(main_device):
                self._xp_sync()
                # TODO: change inside to double instead of cmplx?
                M_mat[inds_gpu] = inputs_tmp[0][:].reshape(-1, 4, 4).real
                N_arr[inds_gpu] = inputs_tmp[1][:].reshape(-1, 4).real
                
        self._xp_set_device(main_device)
        self._xp_sync()
        
        # Solve M * a = N for maximum-likelihood amplitude vector a_i
        # Avoiding explicit matrix inversion M^{-1} reduces numerical error
        a_coeffs = self.xp.linalg.solve(M_mat, N_arr[..., None])[..., 0]

        # Profile log-likelihood: F = 1/2 * (a . N) = 1/2 * (N^T * M^{-1} * N)
        fstat_logl = 0.5 * self.xp.sum(a_coeffs * N_arr, axis=-1)

        a1, a2, a3, a4 = a_coeffs[:, 0], a_coeffs[:, 1], a_coeffs[:, 2], a_coeffs[:, 3]

        # Harmonic projections
        u_cos_minus = a1 + a4
        u_sin_minus = a3 - a2
        u_cos_plus  = a1 - a4
        u_sin_plus  = -(a2 + a3)

        # Polarization amplitudes via vector norms:
        #   norm_minus = 1/2 * (A+ + Ax) = 1/2 * A * (1 - cos(iota))^2
        #   norm_plus  = 1/2 * (A+ - Ax) = 1/2 * A * (1 + cos(iota))^2
        # For iota < pi/2 (cos(iota) > 0): norm_minus < norm_plus
        #   A+ = norm_minus + norm_plus = A * (1 + cos^2(iota))
        #   Ax = norm_minus - norm_plus = -2 * A * cos(iota)
        norm_minus = self.xp.hypot(u_cos_minus, u_sin_minus)
        norm_plus  = self.xp.hypot(u_cos_plus,  u_sin_plus)

        amp_plus  = norm_minus + norm_plus
        amp_cross = norm_minus - norm_plus

        # Intrinsic amplitude A and inclination iota
        sqrt_factor = 2.0 * self.xp.sqrt(norm_minus * norm_plus)
        two_amp = amp_plus + sqrt_factor

        self.A_max = 0.5 * two_amp
        cos_iota = self.xp.clip(-amp_cross / two_amp, -1.0, 1.0)
        self.iota_max = self.xp.arccos(cos_iota)

        # Decoupled phase angles:
        #   theta_minus = atan2(u_sin_minus, u_cos_minus) = 2*psi - phi0
        #   theta_plus  = atan2(u_sin_plus,  u_cos_plus)  = 2*psi + phi0
        theta_minus = self.xp.arctan2(u_sin_minus, u_cos_minus)
        theta_plus  = self.xp.arctan2(u_sin_plus,  u_cos_plus)

        self.psi_max  = (0.25 * (theta_plus + theta_minus)) % np.pi
        self.phi0_max = (0.50 * (theta_plus - theta_minus)) % (2.0 * np.pi)

        if return_cupy:
            return fstat_logl

        try:
            return fstat_logl.get()
        except AttributeError:
            return fstat_logl

    # def get_chi_sqared(
    #     self,
    #     params,
    #     psd,
    #     phase_maximize=False,
    #     start_freq_ind=0,
    #     noise_index=None,
    #     use_c_implementation=True,
    #     N: int=None,
    #     T=4 * YRSID_SI,
    #     dt=10.0,
    #     oversample=1,
    #     return_cupy=False,
    #     **kwargs,
    # ):
    #     """Get batched log likelihood

    #     Generate the individual log likelihood for a batched set of Galactic binaries.
    #     This is also GPU/CPU agnostic.

    #     Args:
    #         params (2D double np.ndarrays): Parameters of all binaries to be calculated.
    #             The shape is ``(number of parameters, number of binaries)``.
    #         data (length 2 list of 1D or 2D complex128 xp.ndarrays): List of arrays representing the data
    #             stream. These should be CuPy arrays if running on the GPU, NumPy
    #             arrays if running on a CPU. The list should be [A channel, E channel].
    #             Should be 1D if only one data stream is analyzed. If 2D, shape is
    #             ``(number of data streams, data_length)``. If 2D,
    #             user must also provide ``data_index`` kwarg.
    #         psd (length 2 list of 1D or 2D double xp.ndarrays): List of arrays representing
    #             the power spectral density (PSD) in the noise.
    #             These should be CuPy arrays if running on the GPU, NumPy
    #             arrays if running on a CPU. The list should be [A channel, E channel].
    #             Should be 1D if only one PSD is analyzed. If 2D, shape is
    #             ``(number of PSDs, data_length)``. If 2D,
    #             user must also provide ``noise_index`` kwarg.
    #         phase_maximize (bool, optional): If True, marginalize over the initial phase.
    #             Default is False.
    #         start_freq_ind (int, optional): Starting index into the frequency-domain data stream
    #             for the first entry of ``data``/``psd``. This is used if a subset of a full data stream
    #             is presented for the computation. If providing mutliple data streams in ``data``, this single
    #             start index value will apply to all of them.
    #         data_index (1D xp.int32 array, optional): If providing 2D ``data``, need to provide ``data_index``
    #             to indicate the data stream associated with each waveform for which the log-Likelihood
    #             is being computed. For example, if you have 100 binaries with 5 different data streams,
    #             ``data_index`` will be a length-100 xp.int32 array with values 0 to 4, indicating the specific
    #             data stream to use for each source.
    #             If ``None``, this will be filled with zeros and only analyzed with the first
    #             data stream given. Default is ``None``.
    #         noise_index (1D xp.int32 array, optional): If providing 2D ``psd``, need to provide ``noise_index``
    #             to indicate the PSD associated with each waveform for which the log-Likelihood
    #             is being computed. For example, if you have 100 binaries with 5 different PSDs,
    #             ``noise_index`` will be a length-100 xp.int32 array with values 0 to 4, indicating the specific
    #             PSD to use for each source.
    #             If ``None``, this will be filled with zeros and only analyzed with the first
    #             PSD given. Default is ``None``.
    #         return_cupy (bool, optional): If ``True``, return CuPy array. Default is ``False``.
    #         **kwargs (dict, optional): Passes keyword arguments to the :func:`run_wave` method.

    #     Raises:
    #         TypeError: If data arrays are NumPy/CuPy while template arrays are CuPy/NumPy.

    #     Returns:
    #         1D double np.ndarray: Log likelihood values associated with each binary.

    #     """

    #     if self.gpus is not None:
    #         # set first index gpu device to control main operations
    #         self._xp_set_device(self.gpus[0])

    #     # get number of observation points and adjust T accordingly
    #     N_obs = int(T / dt)
    #     T = N_obs * dt

    #     self.num_bin = num_bin = params.shape[0]

    #     if N is None:
    #         # TODO: G
    #         N = get_N(self.xp.asarray(params[:, 0]), self.xp.asarray(params[:, 1]), T, oversample=oversample).max().item()

    #     # else N will be int

    #     df = self.df = 1. / T

    #     # get shape of information
    #     if not isinstance(psd, self.xp.ndarray):
    #         raise NotImplementedError

    #     num_data, nchannels, data_length = psd.shape
        
    #     if nchannels < 2:
    #         raise ValueError("Calculates for A and E channels.")
    #     elif nchannels > 2:
    #         warnings.warn("Only calculating A and E channels here currently.")

    #     df = self.df  = 1. / T
    #     psd = [dat.copy().flatten() for dat in psd.transpose(1, 0, 2)]

    #     if noise_index is None:
    #         noise_index = self.xp.zeros(self.num_bin, dtype=self.xp.int32)

    #     # check that index values are ready for computation
    #     assert noise_index.dtype == self.xp.int32

    #     comparison_length = len(psd[0])

    #     assert noise_index.max() * data_length <= comparison_length

    #     if isinstance(start_freq_ind, int):
    #         start_freq_ind_tmp = self.xp.full(num_data[gpu_i], start_freq_ind, dtype=np.int32)
        
    #     else:
    #         assert isinstance(start_freq_ind, self.xp.ndarray) and start_freq_ind.dtype == np.int32
    #         # TODO: fix this num_data
    #         start_freq_ind_tmp = start_freq_ind
    #     assert len(start_freq_ind_tmp) == num_data[gpu_i]
              
    #     num_here = params.shape[0]

    #     num_comps_all = int(num_here * (num_here + 1) / 2) - num_here

    #     # initialize Likelihood terms <d|h> and <h|h>
    #     h1_h1 = self.xp.zeros(num_comps_all, dtype=self.xp.complex128)
    #     h2_h2 = self.xp.zeros(num_comps_all, dtype=self.xp.complex128)        
    #     h1_h2 = self.xp.zeros(num_comps_all, dtype=self.xp.complex128)        
    
    #! ======================================================================
    # TODO: check if we actually need chi_squared because it has some errors, additionally, it is not updated for Mojito orbits.
    # def get_chi_sqared(
    #     self,
    #     params,
    #     psd,
    #     phase_maximize=False,
    #     start_freq_ind=0,
    #     noise_index=None,
    #     use_c_implementation=True,
    #     N: Optional[int] = None,
    #     T=4 * YRSID_SI,
    #     dt=10.0,
    #     oversample=1,
    #     return_cupy=False,
    #     **kwargs,
    # ):
    #     """Get batched log likelihood

    #     Generate the individual log likelihood for a batched set of Galactic binaries.
    #     This is also GPU/CPU agnostic.

        # gpu = self.xp.cuda.runtime.getDevice() if hasattr(self.xp, "cuda") else -1
        # do_synchronize = True

    #     Raises:
    #         TypeError: If data arrays are NumPy/CuPy while template arrays are CuPy/NumPy.

    #     Returns:
    #         1D double np.ndarray: Log likelihood values associated with each binary.

    #     """

    #     if self.gpus is not None:
    #         # set first index gpu device to control main operations
    #         self.xp.cuda.runtime.setDevice(self.gpus[0])

    #     # get number of observation points and adjust T accordingly
    #     N_obs = int(T / dt)
    #     T = N_obs * dt

    #     self.num_bin = num_bin = params.shape[0]

    #     if N is None:
    #         # TODO: G
    #         N = get_N(self.xp.asarray(params[:, 0]), self.xp.asarray(params[:, 1]), T, oversample=oversample).max().item()

    #     # else N will be int

    #     df = self.df = 1. / T

    #     # get shape of information
    #     if not isinstance(psd, self.xp.ndarray):
    #         raise NotImplementedError

    #     num_data, nchannels, data_length = psd.shape
        
    #     if nchannels < 2:
    #         raise ValueError("Calculates for A and E channels.")
    #     elif nchannels > 2:
    #         warnings.warn("Only calculating A and E channels here currently.")

    #     psd = [dat.copy().flatten() for dat in psd.transpose(1, 0, 2)]

    #     if noise_index is None:
    #         noise_index = self.xp.zeros(self.num_bin, dtype=self.xp.int32)

    #     # check that index values are ready for computation
    #     assert noise_index.dtype == self.xp.int32

    #     comparison_length = len(psd[0])

    #     assert noise_index.max() * data_length <= comparison_length

    #     if isinstance(start_freq_ind, int):
    #         start_freq_ind_tmp = self.xp.full(num_data[gpu_i], start_freq_ind, dtype=np.int32)
        
    #     else:
    #         assert isinstance(start_freq_ind, self.xp.ndarray) and start_freq_ind.dtype == np.int32
    #         # TODO: fix this num_data
    #         start_freq_ind_tmp = start_freq_ind
    #     assert len(start_freq_ind_tmp) == num_data[gpu_i]
              
    #     num_here = params.shape[0]

    #     num_comps_all = int(num_here * (num_here + 1) / 2) - num_here

    #     # initialize Likelihood terms <d|h> and <h|h>
    #     h1_h1 = self.xp.zeros(num_comps_all, dtype=self.xp.complex128)
    #     h2_h2 = self.xp.zeros(num_comps_all, dtype=self.xp.complex128)        
    #     h1_h2 = self.xp.zeros(num_comps_all, dtype=self.xp.complex128)        
    
    #     amp, f0, fdot, fddot, phi0, iota, psi, lam, beta = [self.xp.atleast_1d(self.xp.asarray(pars_tmp.copy()))for pars_tmp in params.T]
    #     self.num_bin = num_bin = len(amp)

    #     theta = np.pi / 2 - beta

    #     gpu = self.xp.cuda.runtime.getDevice()
    #     do_synchronize = True

    #     # raise NotImplementedError
    # TODO check if T dependence is not spacecraft related
    #     self.backend.sharedmem.SharedMemoryChiSquaredComp_wrap(
    #         h1_h1,
    #         h2_h2,
    #         h1_h2,
    #         psd[0],
    #         psd[1],
    #         noise_index,
    #         amp, f0, fdot, fddot, phi0, iota, psi, lam, theta, T, dt, N, num_bin, start_freq_ind_tmp, data_length, gpu, do_synchronize, num_data, num_psd
    #     )


    #     if phase_maximize:
    #         self.non_marg_h1_h2 = h1_h2.copy()
    #         try:
    #             self.non_marg_h1_h2 = self.non_marg_h1_h2.get()
    #         except AttributeError:
    #             pass

    #         h1_h2 = self.xp.abs(h1_h2)

    #     # store these likelihood terms for later if needed
    #     self.h1_h1 = h1_h1
    #     self.h2_h2 = h2_h2
    #     self.normalized_corr = h1_h2 / np.sqrt(h1_h1 * h2_h2)

    #     # compute Likelihood
    #     chi_squared_out = -1.0 / 2.0 * (h1_h1 + h2_h2 - 2 * h1_h2).real

    #     if return_cupy:
    #         return chi_squared_out

    #     # back to CPU if on GPU
    #     try:
    #         return chi_squared_out.get()

    #     except AttributeError:
    #         return chi_squared_out
    #! ======================================================================

    
    def fill_global_template(
        self, group_index, templates, A, E, start_inds, N=None, start_freq_ind=0
    ):
        """Fill many global templates with waveforms

        This method takes already generated waveforms (``A, E, start_inds``)
        and their associated grouping index (``group_index``) and fills
        buffer tempalte arrays (``templates``).

        This method combines waveforms that have already been created.
        When a user does not have the waveforms in hand, they should
        use the :func:`generate_global_template` method.

        Args:
            group_index (1D double int32 xp.ndarray): Index indicating to which template each individual binary belongs.
            templates (3D complex128 xp.ndarray): Buffer array for template output to filled in place.
                The shape is ``(number of templates, 2, data_length)``. The ``2`` is
                for the ``A`` and ``E`` TDI channels in that order.
            A (1D or 2D complex128 xp.ndarray): TDI A channel template values for each individual binary.
                The shape if 2D is ``(number of binaries, N)''. In 1D, the array should be arranged so that
                it resembles ``(number of binaries, N).transpose().flatten()``.
                After running waveforms, this is how ``self.A_out`` is arranged.
            E (1D 2D complex128 xp.ndarray): TDI E channel template values for each individual binary.
                The shape if 2D is ``(number of binaries, N)''. In 1D, the array should be arranged so that
                it resembles ``(number of binaries, N).transpose().flatten()``.
                After running waveforms, this is how ``self.E_out`` is arranged.
            start_inds (1D int32 xp.ndarray): The start indices of each binary waveform
                in the full Fourier transform: ``int(f0/T) - N/2``.
            N (int, optional): The length of the A and E channels for each individual binary.
                When ``A`` and ``E`` are 1D, ``N`` must be given. Default is ``None``.
            start_freq_ind (int, optional): Starting index into the frequency-domain data stream
                for the first entry of ``templates``. This is used if a subset of a full data stream
                is presented for the computation.

        Raises:
            TypeError: If data arrays are NumPy/CuPy while tempalte arrays are CuPy/NumPy.
            ValueError: Inputs are not correctly provided.

        """

        # get shape of information
        total_groups, nchannels, data_length = templates.shape
        group_index = self.xp.asarray(group_index, dtype=self.xp.int32)
        self.num_bin = num_bin = len(group_index)

        if nchannels < 2:
            raise ValueError("Calculates for A and E channels.")
        elif nchannels > 2:
            warnings.warn("Only calculating A and E channels here currently.")

        # check if arrays are of same type
        if isinstance(templates, self.xp.ndarray) is False:
            raise TypeError(
                "Make sure the data arrays are the same type as template arrays (cupy vs numpy)."
            )

        # prepare temporary buffers for C/CUDA
        # These are required to ensure the python memory order
        # is read properly in C/CUDA
        template_A = self.xp.zeros_like(
            templates[:, 0], dtype=self.xp.complex128
        ).flatten()
        template_E = self.xp.zeros_like(
            templates[:, 1], dtype=self.xp.complex128
        ).flatten()

        # shift start inds (see above)
        start_inds = (start_inds - start_freq_ind).astype(self.xp.int32)

        # check A, E, N inputs
        if A.ndim > 2 or E.ndim > 2:
            raise ValueError("A_in, E_in have maximum allowable dimension of 2.")
        elif A.ndim == 2:
            N = A.shape[1]
            assert E.ndim == 2
            # assumes the shape is the same as self.A
            A = A.T.flatten()
            E = E.T.flatten()

        elif A.ndim == 1:
            if N is None:
                raise ValueError(
                    "If providing a 1D flattened array for A and E, the N kwarg also needs to be provided."
                )

        # fill the templates in C/CUDA
        self.fill_global_func(
            template_A,
            template_E,
            A,
            E,
            start_inds,
            N,
            num_bin,
            group_index,
            data_length,
        )

        # read out to buffer arrays
        templates[:, 0] += template_A.reshape(total_groups, data_length)
        templates[:, 1] += template_E.reshape(total_groups, data_length)

    def generate_global_template(
        self,
        params,
        group_index,
        templates,
        start_freq_ind=0,
        use_c_implementation=True,
        N=None,
        T=4 * YRSID_SI,
        dt=10.0,
        batch_size=None,
        oversample=1,
        data_length=None,
        tdi_channel_setup="AE",
        data_splits=None,
        num_per_gpu=None,
        factors=None,
        tdi2: bool = False,
        window: Optional[str] = None,
        window_alpha: float = 0.0,
        **kwargs,
    ):
        """Generate global templates from binary parameters

        Generate waveforms in batches and then combine them into
        global fit templates. This method wraps :func:`fill_global_template`
        by building the waveforms first.

        Args:
            params (2D double np.ndarrays): Parameters of all binaries to be calculated.
                The shape is ``(number of parameters, number of binaries)``.
            group_index (1D double int32 xp.ndarray): Index indicating to which template each individual binary belongs.
            templates (3D complex128 xp.ndarray): Buffer array for template output to filled in place.
                The shape is ``(number of templates, 2, data_length)``. The ``2`` is
                for the ``A`` and ``E`` TDI channels in that order.
            start_freq_ind (int, optional): Starting index into the frequency-domain data stream
                for the first entry of ``templates``. This is used if a subset of a full data stream
                is presented for the computation.
            **kwargs (dict, optional): Passes keyword arguments to :func:`run_wave` function above.

        """
        if self.gpus is not None:
            # set first index gpu device to control main operations
            self._xp_set_device(self.gpus[0])

        self.num_bin = num_bin = params.shape[0]

        if N is None:
            # TODO: G
            N = get_N(self.xp.asarray(params[:, 0]), self.xp.asarray(params[:, 1]), T, oversample=oversample, armlength=self.orbits.armlength)
            if self.xp.any(N == 0):
                raise ValueError("N contains zeros.")
        else:
            if isinstance(N, self.xp.ndarray):
                assert params.shape[0] == N.shape[0]
            elif isinstance(N, (int, np.integer)):
                N = self.xp.full(params.shape[0], N)

        unique_N, inverse = self.xp.unique(self.xp.asarray(N), return_inverse=True)
        N_groups = self.xp.arange(len(unique_N))[inverse]
        
        if factors is None:
            factors = self.xp.ones(self.num_bin, dtype=self.xp.float64)
        
        # setup window mapping
        if window is not None:
            try:
                window_type = window_map[window]
            except:
                raise KeyError(f"The window '{window}' is currently not supported. Only the 'tukey', 'planck', or no/rectangular windows are supported.")
        else:
            window_type = 0
            assert window_alpha == 0.0, "No/Rectangular window does not have a smoothing factor"
        
        # check that index values are ready for computation
        assert len(factors) == self.num_bin
        assert factors.dtype == self.xp.float64
        
        assert group_index.dtype == self.xp.int32
        
        nchannels = 3 if tdi_channel_setup != "AE" else 2

        # Python implementation: build the waveforms with run_wave, then
        # accumulate them into the templates with fill_global_template.
        if not use_c_implementation:
            if not isinstance(templates, self.xp.ndarray) or templates.ndim != 3:
                raise ValueError(
                    "The Python path accumulates through fill_global_template, which "
                    "needs one 3D (num_templates, 2, data_length) array. Pass "
                    "use_c_implementation=True for the flattened or per-device layouts."
                )
            if tdi_channel_setup != "AE":
                raise NotImplementedError(
                    "fill_global_template accumulates the A and E channels only, so "
                    f"the Python path cannot serve tdi_channel_setup={tdi_channel_setup!r}."
                )
            if not bool(self.xp.all(factors == 1.0)):
                raise NotImplementedError(
                    "fill_global_template takes no per-binary factor, so the Python "
                    "path cannot apply factors."
                )

            # ! params is passed through unflipped: run_wave applies
            # ! flip_ref_phase itself, so flipping here would double-negate phi0.
            waveform_kwargs = dict(kwargs)
            waveform_kwargs.update(
                T=T, dt=dt, oversample=oversample, tdi2=tdi2,
                tdi_channel_setup=tdi_channel_setup,
                window=window, window_alpha=window_alpha,
                use_c_implementation=False,
            )

            # run_wave takes a single N for every binary handed to it, so the
            # sources are grouped by their N before being batched.
            for nnn, N_here in enumerate(unique_N):
                in_group = N_groups == nnn
                num_here = int(in_group.sum().item())
                if num_here == 0:
                    continue

                params_N = self.xp.asarray(params)[in_group]
                group_index_N = group_index[in_group]
                waveform_kwargs["N"] = int(N_here.item())

                batch_size_here = num_here if batch_size is None else int(batch_size)
                for start in range(0, num_here, batch_size_here):
                    end = min(start + batch_size_here, num_here)

                    # produce TDI templates
                    self.run_wave(*params_N[start:end].T, **waveform_kwargs)
                    self.fill_global_template(
                        group_index_N[start:end],
                        templates,
                        self.A_out,
                        self.E_out,
                        self.start_inds,
                        self.N,
                        start_freq_ind=start_freq_ind,
                    )
            return

        if self.flip_ref_phase:
            # if matching jaxgb, then we need to input - phi0
            params = params.copy()
            params[:, 4] = -params[:, 4]

        if isinstance(templates, self.xp.ndarray):
            templates_in = [templates]
        else:
            templates_in = templates

        num_templates = []
        for t_i, t in enumerate(templates_in):
            if t.ndim > 1 and not t.flags["C_CONTIGUOUS"]:
                raise ValueError(
                    f"generate_global_template writes into templates in place, so entry "
                    f"{t_i} must be C-contiguous. Pass xp.ascontiguousarray(...) instead."
                )

            if t.ndim == 1:
                assert data_length is not None
                assert isinstance(data_length, int)
                num_templates.append(int(t.shape[0] / (data_length * nchannels)))

            elif t.ndim == 2:
                num_templates.append(1)
                assert t.shape[0] == nchannels
                data_length = t.shape[1]
                # print("check this does not create memory issues")
                templates_in[t_i] = t.ravel()

            else:
                ntemplate, _nchannels, data_length = t.shape
                num_templates.append(ntemplate)
                assert _nchannels == nchannels
                # print("check this does not create memory issues")

                templates_in[t_i] = t.ravel()

        self._check_one_entry_per_gpu(
            "generate_global_template", templates=templates_in
        )

        do_synchronize = False
        main_device = self._xp_get_device()
        devices = self._device_iter()
        if data_splits is None:
            assert len(devices) == 1
            data_splits = self.xp.full(num_templates[0], devices[0])

        if num_per_gpu is None:
            assert len(devices) == 1
            num_per_gpu = int(2**31 - 1)
            # make really high so just keeps (int32-safe: numpy 2 rejects
            # int32 arrays modulo a Python int beyond the int32 range)

        inputs_in = []
        for nnn, N_here in enumerate(unique_N):
            N_here = N_here.item()

            # get spacecraft positions
            tm_rel = self.xp.linspace(0, T, num=N_here, endpoint=False)
            tm_abs = tm_rel + self.t0_abs
            Ps_arr = self.xp.array(self._spacecraft(tm_abs)).flatten()

            for gpu_i, gpu in enumerate(devices):
                self._xp_set_device(main_device)
                keep_bool = (N_groups == nnn) & (self.xp.asarray(data_splits)[group_index] == gpu)
                num_split_here = keep_bool.sum().item()
                if num_split_here == 0:
                    continue
                self._xp_set_device(gpu)

                params_here = self.xp.asarray(params)[keep_bool]
                group_index_here = self._intra_split_index(
                    group_index[keep_bool], data_splits, gpu, num_per_gpu
                )
                factors_here = factors[keep_bool]

                if int(group_index_here.max()) >= num_templates[gpu_i]:
                    raise ValueError(
                        f"group_index reaches {int(group_index_here.max())} but only "
                        f"{num_templates[gpu_i]} templates exist on device {gpu}."
                    )

                # theta_add = np.pi / 2 - beta_add
                params_here[:, 8] = np.pi / 2 - params_here[:, 8]

                templates_here = templates_in[gpu_i]
                params_N_tuple = tuple([pars_tmp.copy()for pars_tmp in params_here.T])

                if isinstance(start_freq_ind, int):
                    start_freq_ind_tmp = self.xp.full(num_templates[gpu_i], start_freq_ind, dtype=np.int32)

                else:
                    assert isinstance(start_freq_ind, self.xp.ndarray) and start_freq_ind.dtype == np.int32
                    # TODO: fix this num_templates
                    start_freq_ind_tmp = start_freq_ind
                assert len(start_freq_ind_tmp) == num_templates[gpu_i]

                assert isinstance(T, float)
                assert isinstance(dt, float)

                tuple_in = (
                    (
                        templates_here,
                        group_index_here,
                        factors_here,
                    )
                    + params_N_tuple
                    + (
                        T, dt, N_here,
                        num_split_here, start_freq_ind_tmp, data_length,
                        tdi_channel_setup_map[tdi_channel_setup],
                        gpu, do_synchronize,
                        Ps_arr, self.orbits.armlength, tdi2,
                        window_type, window_alpha
                    )
                )

                self._xp_sync()
                self.backend.sharedmem.SharedMemoryGenerateGlobal_wrap(*tuple_in)
                inputs_in.append([gpu, tuple_in])
                self._xp_sync()

        for gpu, inputs_tmp in inputs_in:
            with self._xp_device(gpu):
                self._xp_sync()

        self._xp_set_device(main_device)
        self._xp_sync()
        return

    def swap_likelihood_difference(
        self,
        params_remove,
        params_add,
        data_minus_template,
        psd,
        start_freq_ind=0,
        data_index=None,
        noise_index=None,
        adjust_inplace=False,
        use_c_implementation=True,
        N=None,
        T=4 * YRSID_SI,
        dt=10.0,
        oversample=1,
        data_length=None,
        data_splits=None,
        phase_maximize=False,
        tdi_channel_setup="AE",
        num_per_gpu=None,
        tdi2: bool = False,
        window: Optional[str] = None,
        window_alpha: float = 0.0,
        **kwargs,
    ):
        self._require_shared_memory_backend(
            use_c_implementation, "swap_likelihood_difference"
        )

        if num_per_gpu is not None:
            raise NotImplementedError("Need to check this.")

        if self.gpus is not None:
            # set first index gpu device to control main operations
            return_to_main_device = self.gpus[0]
            self._xp_set_device(return_to_main_device)

        self.num_bin = num_bin = params_add.shape[0]
        
        if self.flip_ref_phase:
            # if matching jaxgb, then we need to input - phi0
            params_add = params_add.copy()
            params_remove = params_remove.copy()
            params_add[:, 4] = -params_add[:, 4]
            params_remove[:, 4] = -params_remove[:, 4]


        if N is None:
            # TODO: G
            N = get_N(self.xp.asarray(params_add[:, 0]), self.xp.asarray(params_add[:, 1]), T, oversample=oversample, armlength=self.orbits.armlength)
            if self.xp.any(N == 0):
                raise ValueError("N contains zeros.")
        else:
            if isinstance(N, self.xp.ndarray):
                assert params_add.shape[0] == N.shape[0]
            elif isinstance(N, (int, np.integer)):
                N = self.xp.full(params_add.shape[0], N)

        unique_N, inverse = self.xp.unique(self.xp.asarray(N), return_inverse=True)
        N_groups = self.xp.arange(len(unique_N))[inverse]

        # setup window mapping
        if window is not None:
            try:
                window_type = window_map[window]
            except:
                raise KeyError(f"The window '{window}' is currently not supported. Only the 'tukey', 'planck', or no/rectangular windows are supported.")
        else:
            window_type = 0
            assert window_alpha == 0.0, "No/Rectangular window does not have a smoothing factor"

        # check that index values are ready for computation
        if data_index is None:
            data_index = self.xp.zeros(self.num_bin, dtype=self.xp.int32)
        
        if noise_index is None:
            noise_index = self.xp.zeros(self.num_bin, dtype=self.xp.int32)
        
        assert data_index.dtype == self.xp.int32
        assert noise_index.dtype == self.xp.int32
        
        nchannels = 3 if tdi_channel_setup != "AE" else 2

        if isinstance(data_minus_template, self.xp.ndarray):
            data_minus_template_in = [data_minus_template.astype(self.xp.complex128)]
        else:
            data_minus_template_in = [tmp.astype(self.xp.complex128) for tmp in data_minus_template]

        if isinstance(psd, self.xp.ndarray):
            psd_in = [psd.astype(self.xp.complex128)]
        else:
            psd_in = [p.astype(self.xp.complex128) for p in psd]

        num_data = []
        for t_i, t in enumerate(data_minus_template_in):
            if t.ndim == 1:
                assert data_length is not None
                assert isinstance(data_length, int)
                num_data.append(int(t.shape[0] / (data_length * nchannels)))
                
            elif t.ndim == 2:
                num_data.append(1)
                assert t.shape[0] == nchannels
                data_length = t.shape[1]
                # TODO: print("check this does not create memory issues")
                data_minus_template_in[t_i] = t.flatten()

            else:
                ntemplate, _nchannels, data_length = t.shape
                num_data.append(ntemplate)
                assert _nchannels == nchannels
                # TODO: print("check this does not create memory issues")
                
                data_minus_template_in[t_i] = t.flatten()

        if tdi_channel_setup == "AE" or tdi_channel_setup == "AET":
            # assumes nchannels will 1D really
            psd_sub_shape = (nchannels,)
            
        else:
            assert nchannels == 3
            psd_sub_shape = (nchannels, nchannels)

        num_psd = []
        for t_i, t in enumerate(psd_in):
            if t.ndim == 1:
                assert data_length is not None
                assert isinstance(data_length, int)
                num_psd.append(int(t.shape[0] / (data_length * np.prod(psd_sub_shape))))
                
            elif t.ndim == 2:
                assert tdi_channel_setup in ["AET", "AE"]
                num_psd.append(1)
                assert t.shape[0] == nchannels
                data_length = t.shape[1]
                # print("check this does not create memory issues")
                psd_in[t_i] = t.flatten()

            elif t.ndim == 3:
                if tdi_channel_setup in ["AE", "AET"]:
                    ntemplate, _nchannels, data_length = t.shape
                    num_psd.append(ntemplate)
                    assert t.shape[1] == nchannels
                    data_length = t.shape[2]
                    # print("check this does not create memory issues")
                    psd_in[t_i] = t.flatten()


                else:  # XYZ
                    _nchannels, _nchannels, data_length = t.shape
                    num_psd.append(1)
                    assert t.shape[1] == t.shape[0] == nchannels
                    data_length = t.shape[2]
                    # print("check this does not create memory issues")
                    psd_in[t_i] = t.flatten()
                assert _nchannels == nchannels
                
            elif t.ndim == 4:
                assert tdi_channel_setup == "XYZ"
                ntemplate, _nchannels, _nchannels, data_length = t.shape
                num_psd.append(ntemplate)
                assert t.shape[1] == t.shape[2] == nchannels
                data_length = t.shape[3]
                # print("check this does not create memory issues")
                psd_in[t_i] = t.flatten()
            
                assert _nchannels == nchannels
            # print("check this does not create memory issues")
        
        self._check_one_entry_per_gpu(
            "swap_likelihood_difference", data=data_minus_template_in, psd=psd_in
        )

        # initialize Likelihood terms <d|h> and <h|h>
        d_h_remove = self.xp.zeros(self.num_bin, dtype=self.xp.complex128)
        d_h_add = self.xp.zeros(self.num_bin, dtype=self.xp.complex128)
        add_remove = self.xp.zeros(self.num_bin, dtype=self.xp.complex128)
        remove_remove = self.xp.zeros(self.num_bin, dtype=self.xp.complex128)
        add_add = self.xp.zeros(self.num_bin, dtype=self.xp.complex128)

        do_synchronize = False
        main_device = self._xp_get_device()
        devices = self._device_iter()
        if data_splits is None:
            assert len(devices) == 1
            data_splits = self.xp.full(num_data[0], devices[0])

        if num_per_gpu is None:
            assert len(devices) == 1
            num_per_gpu = int(2**31 - 1)
            # make really high so just keeps (int32-safe: numpy 2 rejects
            # int32 arrays modulo a Python int beyond the int32 range)

        inputs_in = []
        for nnn, N_here in enumerate(unique_N):
            N_here = N_here.item()

            # get spacecraft positions
            tm_rel = self.xp.linspace(0, T, num=N_here, endpoint=False)
            tm_abs = tm_rel + self.t0_abs
            Ps_arr = self.xp.array(self._spacecraft(tm_abs)).flatten()

            for gpu_i, gpu in enumerate(devices):
                self._xp_set_device(main_device)
                keep_bool = (N_groups == nnn) & (self.xp.asarray(data_splits)[data_index] == gpu)
                num_split_here = keep_bool.sum().item()
                inds_here = self.xp.arange(len(keep_bool))[keep_bool]
                if num_split_here == 0:
                    continue
                self._xp_set_device(gpu)

                params_remove_here = self.xp.asarray(params_remove)[keep_bool]
                params_add_here = self.xp.asarray(params_add)[keep_bool]

                # theta_add = np.pi / 2 - beta_add
                params_remove_here[:, 8] = np.pi / 2 - params_remove_here[:, 8]
                params_add_here[:, 8] = np.pi / 2 - params_add_here[:, 8]

                data_minus_template_here = data_minus_template_in[gpu_i]
                psd_here = psd_in[gpu_i]

                params_remove_tuple = tuple([pars_tmp.copy()for pars_tmp in params_remove_here.T])
                params_add_tuple = tuple([pars_tmp.copy()for pars_tmp in params_add_here.T])

                if isinstance(start_freq_ind, int):
                    start_freq_ind_tmp = self.xp.full(num_data[gpu_i], start_freq_ind, dtype=np.int32)

                else:
                    assert isinstance(start_freq_ind, self.xp.ndarray) and start_freq_ind.dtype == np.int32
                    # TODO: fix this num_data
                    start_freq_ind_tmp = start_freq_ind
                assert len(start_freq_ind_tmp) == num_data[gpu_i]

                d_h_remove_temp = self.xp.zeros(num_split_here, dtype=self.xp.complex128)
                d_h_add_temp = self.xp.zeros(num_split_here, dtype=self.xp.complex128)
                add_remove_temp = self.xp.zeros(num_split_here, dtype=self.xp.complex128)
                remove_remove_temp = self.xp.zeros(num_split_here, dtype=self.xp.complex128)
                add_add_temp = self.xp.zeros(num_split_here, dtype=self.xp.complex128)

                noise_index_in = self.xp.asarray(noise_index[keep_bool] % num_per_gpu).astype(np.int32)
                data_index_in = self.xp.asarray(data_index[keep_bool] % num_per_gpu).astype(np.int32)

                tuple_in = (
                    (
                        d_h_remove_temp,
                        d_h_add_temp,
                        remove_remove_temp,
                        add_add_temp,
                        add_remove_temp,
                        data_minus_template_here,
                        psd_here,
                        data_index_in,
                        noise_index_in,
                    ) + params_add_tuple
                    + params_remove_tuple
                    + (
                        T, dt, N_here,
                        num_split_here, start_freq_ind_tmp, data_length,
                        tdi_channel_setup_map[tdi_channel_setup],
                        gpu, do_synchronize,
                        num_data[gpu_i], num_psd[gpu_i],
                        Ps_arr, self.orbits.armlength, tdi2,
                        window_type, window_alpha
                    )
                )

                self._xp_sync()
                self.backend.sharedmem.SharedMemorySwapLikeComp_wrap(*tuple_in)
                inputs_in.append([gpu, inds_here, tuple_in])
                self._xp_sync()

        for gpu, inds_gpu, inputs_tmp in inputs_in:
            with self._xp_device(gpu):
                self._xp_sync()

        for gpu, inds_gpu, inputs_tmp in inputs_in:
            with self._xp_device(main_device):
                self._xp_sync()

                d_h_remove[inds_gpu] = inputs_tmp[0][:]
                d_h_add[inds_gpu] = inputs_tmp[1][:]
                remove_remove[inds_gpu] = inputs_tmp[2][:]
                add_add[inds_gpu] = inputs_tmp[3][:]
                add_remove[inds_gpu] = inputs_tmp[4][:]

        self._xp_set_device(main_device)
        self._xp_sync()

        if phase_maximize:
            self.phase_angle = self.xp.arctan2(d_h_add.imag + add_remove.imag, d_h_add.real + add_remove.real)  
            self.non_marg_d_h_add = d_h_add.copy()
            self.non_marg_add_remove = add_remove.copy()
            try:
                self.non_marg_d_h_add = self.non_marg_d_h_add.get()
                self.non_marg_add_remove = self.non_marg_add_remove.get()
            except AttributeError:
                pass

            d_h_add *= self.xp.exp(-1j * self.phase_angle)
            add_remove *= self.xp.exp(-1j * self.phase_angle)
            
        self.d_h_remove = d_h_remove
        self.d_h_add = d_h_add
        self.add_remove = add_remove
        self.remove_remove = remove_remove
        self.add_add = add_add
        
        # compute Likelihood
        ll_diff = -1 / 2 * (-2 * d_h_add + 2 * d_h_remove - 2 * add_remove + add_add + remove_remove).real
        # back to CPU if on GPU
        try:
            return ll_diff.get()

        except AttributeError:
            return ll_diff

    # def inject_signal(self, *args, fmax=None, T=4.0 * YRSID_SI, dt=10.0, **kwargs):
    #     """Inject a single signal

    #     Provides the injection of a single signal into a data stream with frequencies
    #     spanning from 0.0 to fmax with 1/T spacing (from Fourier transform).

    #     Args:
    #         *args (list, tuple, or 1D double np.array): Arguments to provide to
    #             :func:`run_wave` to build the TDI templates for injection.
    #         fmax (double, optional): Maximum frequency to use in data stream.
    #             If ``None``, will use ``1/(2 * dt)``.
    #             Default is ``None``.
    #         T (double, optional): Observation time in seconds. Default is ``4 * YRSID_SI``.
    #         dt (double, optional): Observation cadence in seconds. Default is ``10.0`` seconds.
    #         **kwargs (dict, optional): Passes kwargs to :func:`run_wave`.

    #     Returns:
    #         Tuple of 1D np.ndarrays: NumPy arrays for the A channel and
    #             E channel: ``(A channel, E channel)``. Need to convert to CuPy if working
    #             on GPU.

    #     """

    #     # get binspacing
    #     if fmax is None:
    #         fmax = 1 / (2 * dt)
        
    #     # TODO: change to t0 since this is orbit related
    #     if T > self.orbits.t_base.max():
    #         raise ValueError(
    #             f"Observation time ({T}) is larger than length of time in orbital information ({self.orbits.t_base.max()})"
    #         )

    #     # adjust inputs for run wave
    #     N_obs = int(T / dt)
    #     T = N_obs * dt
    #     kwargs["T"] = T
    #     kwargs["dt"] = dt
    #     self.df = df = 1 / T

    #     # create frequencies
    #     f = np.arange(0.0, fmax + df, df)
    #     num = len(f)

    #     # NumPy arrays for data streams of injections
    #     A_out = np.zeros(num, dtype=np.complex128)
    #     E_out = np.zeros(num, dtype=np.complex128)

    #     # build the templates
    #     self.run_wave(*args, **kwargs)

    #     # add each mode to the templates
    #     start = self.start_inds[0]

    #     # if using GPU, will return to CPU
    #     if self.backend.name == "gpu":
    #         A_temp = self.A_out.squeeze().get()
    #         E_temp = self.E_out.squeeze().get()

    #     else:
    #         A_temp = self.A_out.squeeze()
    #         E_temp = self.E_out.squeeze()

    #     # fill the data streams at the4 proper frqeuencies
    #     A_out[start.item() : start.item() + self.N] = A_temp
    #     E_out[start.item() : start.item() + self.N] = E_temp

    #     return A_out, E_out

    def _apply_parameter_transforms(self, params, parameter_transforms):
        """Apply parameter transformations to params for Information Matrix."""
        for ind_trans, trans in parameter_transforms.items():
            if isinstance(ind_trans, int):
                params[ind_trans] = trans(params[ind_trans])
            else:
                params[np.asarray(ind_trans)] = trans(*params[np.asarray(ind_trans)])
        return params

    def information_matrix(
        self,
        params,
        psd,
        inds=None,
        eps: float = 1e-9,
        parameter_transforms: Dict = {},
        easy_central_difference: bool = False,
        noise_index=None,
        use_c_implementation: bool = False,
        N = None,
        T: float = 4 * YRSID_SI,
        dt: float = 10.0,
        data_length=None,
        data_splits=None,
        tdi_channel_setup: str = "AE",
        num_per_gpu: Optional[int] = None,
        oversample: int = 1,
        return_cupy: bool = False,
        tdi2: bool = False,
        window: Optional[str] = None,
        window_alpha: float = 0.0,
        batch_size: int = 50000,
        workspace_budget_bytes: int = 20 * 1024 ** 3,
        **kwargs
    ):
        """Get the information matrix for a batch.

        This function computes the Information matrix for a batch of Galactic binaries.
        It uses a 2nd order calculation for the derivative if ``easy_central_difference`` is ``False``:

        ..math:: \\frac{dh}{d\\lambda_i} = \\frac{-h(\\lambda_i + 2\\epsilon) + h(\\lambda_i - 2\\epsilon) + 8(h(\\lambda_i + \\epsilon) - h(\\lambda_i - \\epsilon))}{12\\epsilson}

        Otherwise, it will just calculate the derivate with a first-order central difference.
        
        This function calculates the Information Matrix dynamically. It internally
        handles memory-safe batching if running in Python, or dispatches the entire
        array to the C++/CUDA backend if `use_c_implementation=True`.
        
        Args:
            params (2D double np.ndarrays): Parameters of all binaries to be calculated.
                Shape should be (num_sources, num_params).
            psd (xp.ndarray): The 1D flattened linear_psd_arr from AnalysisContainerArray.
            noise_index (1D int np.ndarray): Specific walker indices mapped to the binaries.
            data_length (int): Length of the original time-domain data stream.
            inds (1D int np.ndarray, optional): Indices of the parameters to test.
            eps (double, optional): Step to take when calculating the derivative. Default is ``1e-9``.
            parameter_transforms (dict, optional): Dictionary containing the parameter transform functions.
            easy_central_difference (bool, optional): If ``True``, compute derivatives with a first-order difference.
            use_c_implementation (bool): If True, invokes the C++/CUDA backend.
            batch_size (int): Chunk size to use to prevent VRAM overflow when running the Python backend.
            
        Returns:
            3D xp.ndarray: Information Matrices for all binaries with shape: ``(num_sources, num_derivs, num_derivs)``.        
        """
        params = self.xp.atleast_2d(params)
        self.num_bin = num_bin = params.shape[0]
        num_params = params.shape[1]

        if self.flip_ref_phase:
            # if matching jaxgb, then we need to input - phi0
            params = params.copy()
            params[:, 4] = -params[:, 4]

        if parameter_transforms is not None:
            phys_base = self._apply_parameter_transforms(params.T.copy(), parameter_transforms).T
        else:
            phys_base = params.copy()
        
        if N is None:
            N = get_N(self.xp.asarray(phys_base[:, 0]), self.xp.asarray(phys_base[:, 1]), T, oversample=oversample, armlength=self.orbits.armlength)
            if self.xp.any(N == 0):
                raise ValueError("N contains zeros.")
        else:
            if isinstance(N, self.xp.ndarray):
                assert phys_base.shape[0] == N.shape[0]
            elif isinstance(N, (int, self.xp.integer)):
                N = self.xp.full(phys_base.shape[0], N)

        # This matches q = rint(f0 * T) in build_single_waveform. 
        q_check = self.xp.rint(phys_base[:, 1] * T).astype(self.xp.int32)
        start_freq_inds = (q_check - N / 2).astype(self.xp.int32)

        if inds is None:
            inds = self.xp.arange(num_params)
        inds = self.xp.asarray(inds, dtype=self.xp.int32)

        if noise_index is None:
            noise_index = self.xp.zeros(num_bin, dtype=self.xp.int32)
        
        num_derivs = len(inds)
        self.df = 1 / T

        # Setup window mapping
        if window is not None:
            try:
                window_type = window_map[window]
            except KeyError:
                raise KeyError(f"The window '{window}' is not supported.")
        else:
            window_type = 0
            assert window_alpha == 0.0, "No/Rectangular window does not have a smoothing factor"

        nchannels = 3 if tdi_channel_setup != "AE" else 2

        # shared-memory C++/CUDA backend, device-agnostic
        if use_c_implementation:
            if data_length is None:
                raise ValueError(
                    "data_length is required by the shared-memory backend so the psd "
                    "can be indexed by frequency bin."
                )
            info_matrix = self.xp.zeros((num_bin, num_derivs, num_derivs), dtype=self.xp.float64)

            eps_scaled = self.xp.zeros((num_bin, num_derivs), dtype=self.xp.float64)
            if parameter_transforms:
                for i, ind in enumerate(inds):
                    p_up = params.copy()
                    p_up[:, ind] += eps
                    p_down = params.copy()
                    p_down[:, ind] -= eps
                    
                    phys_up = self._apply_parameter_transforms(p_up.T.copy(), parameter_transforms).T
                    phys_down = self._apply_parameter_transforms(p_down.T.copy(), parameter_transforms).T
                    
                    phys_up[:, 8] = np.pi / 2.0 - phys_up[:, 8]
                    phys_down[:, 8] = np.pi / 2.0 - phys_down[:, 8]
                        
                    eps_scaled[:, i] = (phys_up[:, ind] - phys_down[:, ind]) / (2.0 * eps)
            else:
                eps_scaled = self.xp.ones((num_bin, num_derivs), dtype=self.xp.float64)
                if 8 in inds:
                    idx_8 = list(inds).index(8)
                    eps_scaled[:, idx_8] = -1.0  # Theta = pi/2 - Beta logic

            phys_base[:, 8] = np.pi / 2.0 - phys_base[:, 8]
            
            if isinstance(psd, self.xp.ndarray):
                psd_in = [psd.astype(self.xp.complex128)]
            else:
                psd_in = [p.astype(self.xp.complex128) for p in psd]

            psd_sub_shape = (nchannels,) if tdi_channel_setup in ["AE", "AET"] else (nchannels, nchannels)
            num_psd = []
            for t_i, t in enumerate(psd_in):
                if t.ndim == 1:
                    num_psd.append(int(t.shape[0] / (data_length * np.prod(psd_sub_shape))))
                else:
                    if t.ndim == 2: num_psd.append(1); data_length = t.shape[1]
                    elif t.ndim == 3: num_psd.append(t.shape[0] if tdi_channel_setup in ["AE", "AET"] else 1); data_length = t.shape[2]
                    elif t.ndim == 4: num_psd.append(t.shape[0]); data_length = t.shape[3]
                    psd_in[t_i] = t.flatten()

            self._check_one_entry_per_gpu("information_matrix", psd=psd_in)

            unique_N, inverse = self.xp.unique(self.xp.asarray(N), return_inverse=True)
            N_groups = self.xp.arange(len(unique_N))[inverse]

            do_synchronize = False
            main_device = self._xp_get_device()
            devices = self._device_iter()

            if data_splits is None:
                data_splits = self.xp.full(num_psd[0], devices[0])
            if int(self.xp.asarray(noise_index).max()) >= len(data_splits):
                raise ValueError(
                    f"noise_index reaches {int(self.xp.asarray(noise_index).max())} but "
                    f"data_splits has length {len(data_splits)}."
                )
            if num_per_gpu is None:
                num_per_gpu = int(2**31 - 1)

            inputs_in = []
            for nnn, N_here in enumerate(unique_N):
                N_here = N_here.item()
                tm_rel = self.xp.linspace(0, T, num=N_here, endpoint=False)
                tm_abs = tm_rel + self.t0_abs
                Ps_arr = self.xp.array(self._spacecraft(tm_abs)).flatten()

                for gpu_i, gpu in enumerate(devices):
                    self._xp_set_device(main_device)
                    keep_bool = (N_groups == nnn) & (self.xp.asarray(data_splits)[noise_index] == gpu)
                    num_split_total = keep_bool.sum().item()
                    
                    if num_split_total == 0: continue

                    bytes_per_source = num_derivs * 3 * N_here * 16
                    batch_size_here = max(
                        1, min(int(batch_size), int(workspace_budget_bytes) // bytes_per_source)
                    )

                    batches = self.xp.arange(0, num_split_total, batch_size_here, dtype=self.xp.int32)
                    if batches[-1] < num_split_total:
                        batches = self.xp.concatenate([batches, self.xp.array([num_split_total], dtype=self.xp.int32)])

                    full_inds_here = self.xp.arange(len(keep_bool))[keep_bool]
                    psd_here = psd_in[gpu_i]

                    # Allocated once and reused across batches. Every launch below is
                    # synchronised, so no batch reads the workspace while the next writes it.
                    self._xp_set_device(gpu)
                    d_dh_workspace = self.xp.zeros(
                        min(batch_size_here, num_split_total) * num_derivs * 3 * N_here,
                        dtype=self.xp.complex128,
                    )

                    for start, end in zip(batches[:-1], batches[1:]):
                        num_split_here = int(end - start)
                        self._xp_set_device(gpu)

                        inds_here = full_inds_here[start:end]
                        params_here = phys_base[inds_here]
                        params_tuple = tuple([pars_tmp.copy() for pars_tmp in params_here.T])

                        start_freq_ind_tmp = start_freq_inds[inds_here]
                        noise_index_in = self.xp.asarray(noise_index[inds_here] % num_per_gpu).astype(np.int32)
                        eps_scaled_here = eps_scaled[inds_here].flatten()

                        info_mat_temp = self.xp.zeros(num_split_here * num_derivs * num_derivs, dtype=self.xp.float64)

                        tuple_in = (
                            (info_mat_temp, d_dh_workspace, psd_here, noise_index_in, inds)
                            + params_tuple 
                            + (eps_scaled_here, eps, T, dt, N_here, num_split_here, num_derivs, 
                               start_freq_ind_tmp, data_length, tdi_channel_setup_map[tdi_channel_setup], 
                               gpu, do_synchronize, num_psd[gpu_i], Ps_arr, self.orbits.armlength, 
                               tdi2, easy_central_difference, window_type, window_alpha)
                        )

                        self._xp_sync()
                        self.backend.sharedmem.SharedMemoryInfoMatComp_wrap(*tuple_in)
                        inputs_in.append([gpu, inds_here, tuple_in])
                        self._xp_sync()

            for gpu, inds_gpu, inputs_tmp in inputs_in:
                with self._xp_device(main_device):
                    self._xp_sync()
                    info_matrix[inds_gpu] = inputs_tmp[0][:].reshape(-1, num_derivs, num_derivs)

            self._xp_set_device(main_device)
            self._xp_sync()

            if not return_cupy:
                # .get() only exists on cupy arrays; numpy output passes through.
                try:
                    info_matrix = info_matrix.get()
                except AttributeError:
                    pass

            return info_matrix
        
        # python implementation
        else:
            if data_length is None:
                raise ValueError(
                    "data_length is required so the psd can be reshaped and sliced."
                )

            info_matrix = self.xp.zeros((num_bin, num_derivs, num_derivs), dtype=self.xp.float64)

            if tdi_channel_setup == "XYZ":
                reshaped_psd = psd.reshape(-1, 3, 3, data_length)
            else:
                reshaped_psd = psd.reshape(-1, nchannels, data_length)

            # I think this is fine for now, cannot have multiple N when setting up dh later
            N_val = int(N.max().item()) if isinstance(N, self.xp.ndarray) else int(N)

            waveform_kwargs: Dict[str, Any] = dict(
                N=N_val, T=T, dt=dt, oversample=oversample, tdi2=tdi2,
                tdi_channel_setup=tdi_channel_setup, use_c_implementation=False,
                window=window, window_alpha=window_alpha
            )
            
            # batching here to avoid memory allocation issues (dh is of size 10000x9x3x1024=4.4 GBytes)
            batches = self.xp.arange(0, num_bin, batch_size, dtype=self.xp.int32)
            if batches[-1] < num_bin:
                batches = self.xp.concatenate([batches, self.xp.array([num_bin], dtype=self.xp.int32)])
            
            for start, end in zip(batches[:-1], batches[1:]):
                batched_params = params[start:end]
                noise_idx_batch = noise_index[start:end]
                start_idx_batch = start_freq_inds[start:end]
                num_bin_batch = int(end - start)
                
                # Get correct psd slice for each binary out of psd_linear_arr
                f_idx = start_idx_batch[:, None] + self.xp.arange(N_val)
                if int(f_idx.min()) < 0 or int(f_idx.max()) >= data_length:
                    raise ValueError(
                        f"Waveform frequency bins span [{int(f_idx.min())}, {int(f_idx.max())}] "
                        f"but the psd only covers [0, {data_length - 1}]. A source lies too "
                        f"close to the band edge for N = {N_val}."
                    )
                w_idx = self.xp.asarray(noise_idx_batch)


                if tdi_channel_setup == "XYZ":
                    w_idx_exp = w_idx[:, None, None, None]               
                    c1_idx = self.xp.arange(3)[None, :, None, None]       
                    c2_idx = self.xp.arange(3)[None, None, :, None]      
                    f_idx_exp = f_idx[:, None, None, :]                  
                    inv_psd = reshaped_psd[w_idx_exp, c1_idx, c2_idx, f_idx_exp]
                else: 
                    w_idx_exp = w_idx[:, None, None]
                    c_idx = self.xp.arange(nchannels)[None, :, None]
                    f_idx_exp = f_idx[:, None, :]
                    inv_psd = reshaped_psd[w_idx_exp, c_idx, f_idx_exp]
                
                dh = self.xp.zeros(
                    (num_bin_batch, num_derivs, nchannels, N_val),
                    dtype=self.xp.complex128
                )

                def get_wave(params_to_run): # wrapper function for wave_gen with parameter transforms
                    transformed_params = params_to_run.T.copy()
                    if parameter_transforms:
                        transformed_params = self._apply_parameter_transforms(
                            transformed_params, 
                            parameter_transforms
                        )
                    self.run_wave(*transformed_params, **waveform_kwargs)
                    if tdi_channel_setup == "XYZ":
                        return self.XYZf.copy()
                    elif tdi_channel_setup == "AE":
                        return self.AETf[:, :2].copy()
                    else: # "AET"
                        return self.AETf.copy()

                for i, ind in enumerate(inds):
                    # 1 eps up derivative
                    params_up_1 = batched_params.copy()
                    params_up_1[:, ind] += eps # to match sharedmem backend * batched_params[:, ind]
                    h_I_up_eps = get_wave(params_up_1)

                    # 1 eps down derivative
                    params_down_1 = batched_params.copy()
                    params_down_1[:, ind] -= eps # to match sharedmem backend * batched_params[:, ind]
                    h_I_down_eps = get_wave(params_down_1)

                    if easy_central_difference: # compute derivative and store
                        dh[:, i] = (h_I_up_eps - h_I_down_eps) / (2.0 * eps)
                        
                    else: # higher degree derivative computation
                        # 2 eps up derivative
                        params_up_2 = batched_params.copy()
                        params_up_2[:, ind] += 2.0 * eps # to match sharedmem backend * batched_params[:, ind]
                        h_I_up_2eps = get_wave(params_up_2)

                        # 2 eps down derivative
                        params_down_2 = batched_params.copy()
                        params_down_2[:, ind] -= 2.0 * eps # to match sharedmem backend * batched_params[:, ind]
                        h_I_down_2eps = get_wave(params_down_2)

                        dh[:, i] = (-h_I_up_2eps + h_I_down_2eps + 8.0 * (h_I_up_eps - h_I_down_eps)) / (12.0 * eps)

                # TODO: Possibly add a check for scenario where \pm 2\epsilon is outside of freq band?
                
                # compute Information matrix via inner products
                if tdi_channel_setup == "XYZ":
                    # dh shape = [num_bins, num_derivs, 3, N]
                    # inv_csd shape = [num_bins, 3, 3, N]
                    # b=num_bins, i=deriv1, c=chan1, k=freq_bin, d=chan2, j=deriv2
                    info_matrix[start:end] = 4.0 * self.df * self.xp.einsum(
                        "bick,bcdk,bjdk->bij", dh.conj(), inv_psd, dh
                    ).real
                    # output shape = [num_bin, num_derivs, num_derivs]
                else:
                    # similar indices as above, but inv_psd is diagonal
                    info_matrix[start:end] = 4.0 * self.df * self.xp.einsum(
                        "bick,bjck,bck->bij", dh.conj(), dh, inv_psd
                    ).real

            if self.backend.name == "gpu" and not return_cupy:
                info_matrix = info_matrix.get()

            return info_matrix



class GBGPU(GBGPUBase):
    """Inherit this class to expand on GBGPU waveforms.

    The required methods to be added are shown below.

    """

    def prepare_additional_args(self, *args):
        """Prepare the arguments special to this class

        This function must take in the extra ``args`` input
        into :meth:`GBGPU.run_wave` and transform them as needed
        to input into the rest of the code. If using GPUs,
        this is where the parameters are copied to GPUs.

        Args:
            *args (tuple): Any additional args to be dealt with.

        Returns:
            Tuple: New args. In the rest of the code this is ``add_args``.

        """
        return ValueError(
            "If providing more args than the base args, must be a class derived from GBGPUBase with an adjusted 'prepare_additional_args' method."
        )

    def special_get_N(
        self,
        amp,
        f0,
        T,
        *args,
        oversample=1,
    ):
        """Determine proper sampling rate in time domain for slow-part.

        Args:
            amp (double or 1D double np.ndarray): Amplitude parameter.
            f0 (double or 1D double np.ndarray): Initial frequency of gravitational
                wave in Hz.
            T (double): Observation time in seconds.
            *args (tuple): Args input for beyond-GBGPU functionality.
            oversample(int, optional): Oversampling factor compared to the determined ``N``
                value. Final N will be ``oversample * N``. This is only used if N is
                not provided. Default is ``1``.
        Returns:
            1D int32 xp.ndarray: Number of time-domain points recommended for each binary.

        """
        return get_N(amp, f0, T, oversample=oversample, armlength=self.orbits.armlength)
    
    def shift_frequency(self, fi, xi, *args):
        """Shift the evolution of the frequency in the slow part

        Args:
            fi (3D double xp.ndarray): Instantaneous frequencies of the
                wave before applying third-body effect at each spacecraft as a function of time.
                The shape is ``(num binaries, 3 spacecraft, N)``.
            xi (3D double xp.ndarray): Time at each spacecraft.
                The shape is ``(num binaries, 3 spacecraft, N)``.
            *args (tuple): Args returned from :meth:`prepare_additional_args`.

        Returns:
            3D double xp.ndarray: Updated frequencies with third-body effect.

        """
        return fi

    def add_to_phasing(self, arg_phase, f0, fdot, fddot, xi, *args):
        """Update ``argS`` in FastGB formalism for third-body effect

        ``argS`` is an effective phase that goes into ``kdotP`` in the construction
        of the slow part of the waveform. ``kdotP`` is then included directly
        in the transfer function. See :meth:`gbgpu.gbgpu.GBGPU._construct_slow_part`
        for the use of argS in the larger code.

        # TODO: need to check new C code against this

        Args:
            arg_phase (3D double xp.ndarray): Special phase evaluation beyond ``kdotP``.
                Shape is ``(num binaries, 3 spacecraft, N)``.
            f0 (1D double np.ndarray): Initial frequency of gravitational
                wave in Hz.
            fdot (1D double np.ndarray): Initial time derivative of the
                frequency given as Hz/s.
            fddot (1D double np.ndarray): Initial second derivative with
                respect to time of the frequency given in Hz/s^2.
            xi (3D double xp.ndarray): Time at each spacecraft.
                The shape is ``(num binaries, 3 spacecraft, N)``.
            T (double): Observation time in seconds.
            *args (tuple): Args returned from :meth:`prepare_additional_args`.

        Returns:
            3D double xp.ndarray: Updated ``argS`` with third-body effect

        """
        return arg_phase
