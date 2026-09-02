from ast import Mod
import numpy as np
import warnings

from lisatools.utils.constants import MSUN_SI, G_SI, C_SI, PC_SI, fstar, YRSID_SI

try:
    from lisatools import sensitivity as tdi

    tdi_available = True

except (ModuleNotFoundError, ImportError) as e:
    tdi_available = False
    warnings.warn("tdi module not found. No sensitivity information will be included.")

# NOTE: import the runtime module via the attribute form (`from cupy.cuda import
# runtime`), NOT the submodule form (`from cupy.cuda.runtime import setDevice`).
# The submodule form forces Python to load cupy's `cupy/cuda/runtime.py` shim
# (a `from ... import *` re-export that drops underscore-private names) and
# rebinds `cupy.cuda.runtime` process-wide to that shim, which breaks
# `cupy.cuda.get_local_runtime_version()` for other libraries (e.g. FastEMRIWaveforms).
try:
    from cupy.cuda import runtime as _cuda_runtime

    setDevice = _cuda_runtime.setDevice
except (ModuleNotFoundError, ImportError):
    setDevice = None


def AET(X, Y, Z):
    """Return the A,E,T channels from X,Y,Z

    Args:
        X,Y,Z (xp.ndarray): Arrays holding ``XYZ`` TDI information. The signals can be in any domain.

    Returns:
        Tuple: ``(A,E,T)`` with array shapes the same as the input ``XYZ``.

    """
    return (
        (Z - X) / np.sqrt(2.0),
        (X - 2.0 * Y + Z) / np.sqrt(6.0),
        (X + Y + Z) / np.sqrt(3.0),
    )


def get_fGW(f0, fdot, fddot, t, xp=None):
    """Get instantaneous frequency of gravitational wave

    Computes :math:`f(t)\\approx f_0 + \\dot{f}_0 t + \\frac{1}{2}\\ddot{f}_0 t^2.`
    This assumes the initial time is 0.

    Args:
        f0 (1D xp.ndarray): Initial frequency of source in Hz. Shape is ``(num_bin_all,)``.
        fdot (1D xp.ndarray): Initial frequency derivative of source in Hz/s. Shape is ``(num_bin_all,)``.
        fddot (1D xp.ndarray): Initial second derivative of the frequency of source in Hz/s^2. Shape is ``(num_bin_all,)``.
        t (xp.ndarray): Time values at which to evaluate the frequencies.
            If ``t.ndim > 1``, it will cast the frequency information properly.
            In this case, ``t.shape[0]`` must be equal to ``num_bin_all``.

    Returns:
        xp.ndarray: All frequencies evaluated at the given times.

    Raises:
        AssertionError: ``t.shape[0] != num_bin_all and t.shape[0] != 1``.


    """
    if xp is None:
        xp = np

    # get dimensionality of t
    tdim = t.ndim

    # adjust dimensions if needed
    dim_diff = tdim - 1
    if dim_diff > 0:
        assert (t.shape[0] == len(f0)) or (t.shape[0] == 1)

        dims_to_expand = tuple(list(np.arange(1, dim_diff + 1)))
        f0 = xp.expand_dims(f0, dims_to_expand)
        fdot = xp.expand_dims(fdot, dims_to_expand)
        fddot = xp.expand_dims(fddot, dims_to_expand)

    # calculate
    f = f0 + fdot * t + 0.5 * fddot * t * t
    return f


def get_chirp_mass(m1, m2):
    """Get chirp mass

    Args:
        m1 (xp.ndarray): Mass 1.
        m2 (xp.ndarray): Mass 2.

    Returns:
        xp.ndarray: Chirp mass.


    """
    return (m1 * m2) ** (3 / 5) / (m1 + m2) ** (1 / 5)


def get_eta(m1, m2):
    """Get symmetric mass ratio

    Args:
        m1 (xp.ndarray): Mass 1.
        m2 (xp.ndarray): Mass 2.

    Returns:
        xp.ndarray: Symetric mass ratio.


    """
    return (m1 * m2) / (m1 + m2) ** 2


def get_amplitude(m1, m2, f, d):
    """Get amplitude of GW

    Args:
        m1 (xp.ndarray): Mass 1 in solar masses.
        m2 (xp.ndarray): Mass 2 in solar masses.
        f (xp.ndarray): Frequency of gravitational wave in Hz.
        d (xp.ndarray): Luminosity distance in kpc.

    Returns:
        xp.ndarray: Amplitude.

    """
    Mc = get_chirp_mass(m1, m2) * MSUN_SI
    d = d * 1e3 * PC_SI  # kpc to meters
    A = 2 * (G_SI * Mc) ** (5.0 / 3.0) / (C_SI**4 * d) * (np.pi * f) ** (2.0 / 3.0)
    return A


def get_fdot(f, m1=None, m2=None, Mc=None):
    """Get fdot of GW

    Must provide either ``m1`` and ``m2`` or ``Mc``.

    Args:
        f (xp.ndarray): Frequency of gravitational wave in Hz.
        m1 (xp.ndarray, optional): Mass 1 in solar masses.
        m2 (xp.ndarray, optional): Mass 2 in solar masses.
        Mc (xp.ndarray, optional): Chirp mass in solar masses.

    Returns:
        xp.ndarray: fdot.

    Raises:
        ValueError: Inputs are incorrect.
        AssertionError: Inputs are incorrect.

    """

    # prepare inputs and convert to chirp mass if needed
    if m1 is None and m2 is None and Mc is None:
        raise ValueError("Must provide either m1 & m2 or Mc.")
    elif m1 is not None or m2 is not None:
        assert m1 is not None and m2 is not None
        Mc = get_chirp_mass(m1, m2) * MSUN_SI
    elif Mc is not None:
        # no in-place op: callers pass views of live parameter arrays
        Mc = Mc * MSUN_SI

    # calculate fdot
    fdot = (
        (96.0 / 5.0)
        * np.pi ** (8 / 3)
        * (G_SI * Mc / C_SI**3) ** (5 / 3)
        * f ** (11 / 3)
    )
    return fdot


def get_chirp_mass_from_f_fdot(f, fdot):
    """Get chirp mass from f and fdot of GW

    Args:
        f (xp.ndarray): Frequency of gravitational wave in Hz.
        fdot (xp.ndarray): Frequency derivative of gravitational wave in Hz/s.

    Returns:
        xp.ndarray: chirp mass.

    """

    # backout
    Mc_SI = (
        5.0
        / 96.0
        * np.pi ** (-8 / 3)
        * (G_SI / C_SI**3) ** (-5 / 3)
        * f ** (-11 / 3)
        * fdot
    ) ** (3 / 5)
    Mc = Mc_SI / MSUN_SI
    return Mc


def get_N(amp, f0, Tobs, oversample=1, armlength=None):
    """Determine sampling rate for slow part of FastGB waveform.

    Args:
        amp (xp.ndarray): Amplitude of gravitational wave.
        f0 (xp.ndarray): Frequency of gravitational wave in Hz.
        Tobs (double): Observation time in seconds.
        oversample (int, optional): Oversampling factor. This function will return
            ``oversample * N``, if N is the determined sample number.
            (Default: ``1``).
        armlength (double, optional): Constellation arm length in meters, used for the
            transfer frequency. If ``None``, uses the nominal LISA value. Pass
            ``orbits.armlength`` for L1 or Mojito orbits, whose arm length differs.
            (Default: ``None``).

    Returns:
        int xp.ndarray: N values for each binary entered.

    """
    # The transfer frequency must follow the orbits in use, not the nominal constant.
    transfer_freq = fstar if armlength is None else C_SI / (armlength * 2 * np.pi)

    # make sure they are arrays
    amp = np.atleast_1d(amp)
    f0 = np.atleast_1d(f0)

    # default mult
    mult = 8

    # adjust mult based on observation time
    if (Tobs / YRSID_SI) <= 1.0:
        mult = 1

    elif (Tobs / YRSID_SI) <= 2.0:
        mult = 2

    elif (Tobs / YRSID_SI) <= 4.0:
        mult = 4

    elif (Tobs / YRSID_SI) <= 8.0:
        mult = 8

    # cast for all binaries
    mult = np.full_like(f0, mult, dtype=np.int32)

    N = 32 * mult

    # adjust based on the frequency of the source
    N[f0 >= 0.1] = 1024 * mult[f0 >= 0.1]
    N[(f0 >= 0.03) & (f0 < 0.1)] = 512 * mult[(f0 >= 0.03) & (f0 < 0.1)]
    N[(f0 >= 0.01) & (f0 < 0.03)] = 256 * mult[(f0 >= 0.01) & (f0 < 0.03)]
    N[(f0 >= 0.001) & (f0 < 0.01)] = 64 * mult[(f0 >= 0.001) & (f0 < 0.01)]

    # if a sensitivity curve is available, verify the SNR is not too high
    # if it is, needs more points
    if tdi_available:
        fonfs = f0 / transfer_freq

        SnX = np.sqrt(tdi.X1TDISens.get_Sn(f0))

        #  calculate michelson noise
        Sm = SnX / (4.0 * np.sin(fonfs) * np.sin(fonfs))

        Acut = amp * np.sqrt(Tobs / Sm)

        exponent = np.ceil(np.log(Acut) / np.log(2.0) + 1)
        M = np.round(2.0 ** exponent).astype(int)

        N = np.maximum(M, N)
    else:
        warnings.warn(
            "Sensitivity information not available. The number of points in the waveform "
            "will not be determined by the signal strength."
        )
        
    # adjust with oversample
    N_out = (N * oversample).astype(np.int32)

    # cuda implementation relies on Nmax=2048
    Nmax = 2048
    N_out = np.minimum(N_out, Nmax)

    # np.any(N_out) == 0 tested whether every entry was zero, not whether any was.
    if np.any(N_out == 0):
        raise ValueError("get_N produced zero-length waveforms for at least one binary.")

    return N_out


def cuda_set_device(dev):
    """Globally sets CUDA device

    Args:
        dev (int): CUDA device number.

    """
    if setDevice is not None:
        setDevice(dev)
    else:
        warnings.warn("Setting cuda device, but cupy/cuda not detected.")
