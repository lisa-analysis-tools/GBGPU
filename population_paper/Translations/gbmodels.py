import numpy as np


from abc import ABC
import shutil
import wave
import numpy as np
import os
from scipy import stats
import h5py
import time
from scipy.stats import rv_continuous
import kalepy as kale
from scipy.interpolate import CubicSpline
from scipy.integrate import quad

from eryn.prior import ProbDistContainer
from eryn.ensemble import EnsembleSampler
from eryn.paraensemble import ParaEnsembleSampler
from eryn.utils import PeriodicContainer

try:
    import cupy as xp
    from cupy.cuda.runtime import setDevice

    gpu_available = True
except ModuleNotFoundError:
    import numpy as xp

    gpu_available = False
# import matplotlib.pyplot as plt

from lisatools.sensitivity import get_sensitivity
from lisatools.sampling.likelihood import Likelihood
from lisatools.diagnostic import (
    inner_product,
    snr,
    fisher,
    covariance,
    mismatch_criterion,
    cutler_vallisneri_bias,
    scale_snr,
)
from eryn.prior import uniform_dist
from eryn.utils import TransformContainer
from eryn.state import State, ParaState, BranchSupplemental
from eryn.backends import HDFBackend

from lisatools.sampling.stopping import SNRStopping, SearchConvergeStopping

from gbgpu.gbgpu import GBGPU
from gbgpu.thirdbody import GBGPUThirdBody

from gbgpu.utils.constants import *
from gbgpu.utils.utility import *

from lisatools.sampling.samplingguide import GBGuide

import warnings

warnings.filterwarnings("ignore")
use_gpu = gpu_available

class TemplateSetup(ABC):

    def __init__(self, name, ndim, template_gen, priors, periodic=None, transform_fn=None, use_gpu=False):

        self.name = name
        self.ndim = ndim
        self.use_gpu = use_gpu
        self.template_gen = template_gen
        self.priors = priors    
        self.transform_fn = transform_fn
        self.periodic = periodic


class BaseTemplateSetup(TemplateSetup):
    def __init__(self, use_gpu=False):

        # setup everything specific to the Base Template
        name = "base"
        template_gen = GBGPU(use_gpu=use_gpu)
        
        # build priors
        priors_in = {
            0: uniform_dist(np.log(1e-24), np.log(1e-20)),  # amp
            1: uniform_dist(0.0, 1.0),  # mapped f0
            2: uniform_dist(0.0, 1.0),  # mapped fdot
            3: uniform_dist(0.0, 2 * np.pi),  # phi0
            4: uniform_dist(-1, 1),  # cosinc
            5: uniform_dist(0.0, np.pi),  # psi
            6: uniform_dist(0.0, 2 * np.pi),  # lam
            7: uniform_dist(-1, 1),  # sinbeta
        }
        
        # initialize full prior setup
        priors = {"gb": ProbDistContainer(priors_in, use_cupy=use_gpu, return_gpu = True)}

        # transform function from sampling basis to waveform basis
        transform_fn_in = {
            0: (lambda x: np.exp(x)),  # ln(amp) -> amp
            1: (lambda x: x * 1e-3),  # ms -> s (f0)
            5: (lambda x: np.arccos(x)),  # cos(inc) -> inc
            8: (lambda x: np.arcsin(x)),  # sin(beta) -> beta
        }

        # transform function for fddot(f, fdot) = 11/3 fdot^2 / f0
        # this assumes GR is determining inner binary orbit
        transform_fn_in[(1, 2, 3)] = lambda f0, fdot, fddot: (
            f0,
            fdot,
            11 / 3.0 * fdot ** 2 / f0,
        )

        # waveform model takes fddot as a parameter
        # we do not sample over it
        # we add a place for it in the array
        # and then adjust its value with the transform function
        fill_dict = {
            "fill_inds": np.array([3]),
            "ndim_full": 9,
            "fill_values": np.array([0.0]),
        }

        transform_fn = {"gb": 
            TransformContainer(
                parameter_transforms=transform_fn_in, fill_dict=fill_dict
            )
        }

        # sampling dimensionality
        ndim = 8

        # periodic parameters
        periodic = PeriodicContainer({"gb": {3: 2 * np.pi, 5: np.pi, 6: 2 * np.pi}})

        # initialize parent class that ensures all necessary information is included
        super().__init__(name, ndim, template_gen, priors, periodic=periodic, transform_fn=transform_fn, use_gpu=use_gpu)


class PriorTransformFn:
    def __init__(self, f_min: float, f_max: float, fdot_min: float, fdot_max: float, P2_min: float=None, P2_max: float=None):
        self.f_min, self.f_max, self.fdot_min, self.fdot_max = f_min, f_max, fdot_min, fdot_max
        self.P2_min, self.P2_max = P2_min, P2_max
        self.is_third =  self.P2_min is not None

    def adjust_logp(self, logp, groups_running):
        if groups_running is None:
            groups_running = xp.arange(len(self.f_min))
        f_min_here = self.f_min[groups_running]
        f_max_here = self.f_max[groups_running]
        f_logpdf = np.log(1. / (f_max_here - f_min_here))

        fdot_min_here = self.fdot_min[groups_running]
        fdot_max_here = self.fdot_max[groups_running]
        fdot_logpdf = np.log(1. / (fdot_max_here - fdot_min_here))

        logp[:] += f_logpdf[:, None, None]
        logp[:] += fdot_logpdf[:, None, None]

        if self.is_third:
            P2_min_here = self.P2_min[groups_running]
            P2_max_here = self.P2_max[groups_running]
            P2_logpdf = np.log(1. / (P2_max_here - P2_min_here))
            logp[:] += P2_logpdf[:, None, None]

        return logp

    def transform_to_prior_basis(self, coords, groups_running):
        if groups_running is None:
            groups_running = xp.arange(len(self.f_min))

        f_min_here = self.f_min[groups_running]
        f_max_here = self.f_max[groups_running]
        coords[:, :, :, 1] = (coords[:, :, :, 1] - f_min_here[:, None, None]) / (f_max_here[:, None, None] - f_min_here[:, None, None])

        fdot_min_here = self.fdot_min[groups_running]
        fdot_max_here = self.fdot_max[groups_running]
        coords[:, :, :, 2] = (coords[:, :, :, 2] - fdot_min_here[:, None, None]) / (fdot_max_here[:, None, None] - fdot_min_here[:, None, None])
        
        if self.is_third:
            P2_min_here = self.P2_min[groups_running]
            P2_max_here = self.P2_max[groups_running]
            coords[:, :, :, -2] = (coords[:, :, :, -2] - P2_min_here[:, None, None]) / (P2_max_here[:, None, None] - P2_min_here[:, None, None])

        return


class ThirdBodyTemplateSetup(TemplateSetup):
    def __init__(self, ecc_lim=0.985, use_gpu=False):

        # setup everything specific to the Third Template
        name = "third"

        use_gpu = use_gpu
        template_gen = GBGPUThirdBody(use_gpu=use_gpu)

        template_gen.get_ll = template_gen.get_ll_special
        
        # build priors
        priors_in = {
            0: uniform_dist(np.log(1e-24), np.log(1e-20)),  # log(amp)
            1: uniform_dist(0.0, 1.0),  # mapped f0
            2: uniform_dist(0.0, 1.0),  # mapped fdot
            3: uniform_dist(0.0, 2 * np.pi),  # phi0
            4: uniform_dist(-1, 1),  # cosinc
            5: uniform_dist(0.0, np.pi),  # psi
            6: uniform_dist(0.0, 2 * np.pi),  # lam
            7: uniform_dist(-1, 1),  # sinbeta
            8: uniform_dist(np.log(1.0), np.log(1e5)),  # A2 dist,
            9: uniform_dist(0.0, 2 * np.pi),  # varpi
            10: uniform_dist(0.0, ecc_lim),  # ecc
            11: uniform_dist(0.0, 1.0), # mapped P2
            12: uniform_dist(0.0, 1.0),  # T2
        }

        # initialize full prior setup
        priors = {"gb": ProbDistContainer(priors_in, use_cupy=use_gpu, return_gpu = True)}

        # transform function from sampling basis to waveform basis
        transform_fn_in = {
            0: (lambda x: np.exp(x)),  # ln(amp) -> amp
            1: (lambda x: x * 1e-3),  # ms -> s (f0)
            5: (lambda x: np.arccos(x)),  # cos(inc) -> inc
            8: (lambda x: np.arcsin(x)),  # sin(beta) -> beta
            9: (lambda x: np.exp(x)),  # ln(A2) -> A2
            (12, 13): lambda P2, T2: (P2, T2 * P2),  # T2 / P2 -> T2
        }

        # transform function for fddot(f, fdot) = 11/3 fdot^2 / f0
        # this assumes GR is determining inner binary orbit
        transform_fn_in[(1, 2, 3)] = lambda f0, fdot, fddot: (
            f0,
            fdot,
            11 / 3.0 * fdot ** 2 / f0,
        )
        
        # waveform model takes fddot as a parameter
        # we do not sample over it
        # we add a place for it in the array
        # and then adjust its value with the transform function
        fill_dict = {
            "fill_inds": np.array([3]),
            "ndim_full": 14,
            "fill_values": np.array([0.0]),
        }

        transform_fn = {"gb": 
            TransformContainer(
                parameter_transforms=transform_fn_in, fill_dict=fill_dict
            )
        }

        # sampling dimensionality
        ndim = 13

        # periodic parameters (phi0, psi, lam, varpi, T2)
        periodic = PeriodicContainer({"gb": {3: 2 * np.pi, 5: np.pi, 6: 2 * np.pi, 9: 2 * np.pi, 12: 1.0}})

        super().__init__(name, ndim, template_gen, priors, periodic=periodic, transform_fn=transform_fn, use_gpu=use_gpu)


class LogLikeFn:
    def __init__(self, gb, data, psd, start_freq, df, transform_fn, N_vals, data_length, d_d_all, **waveform_kwargs):
        self.data, self.psd, self.start_freq, self.df = data, psd, start_freq, df
        self.transform_fn = transform_fn
        self.N_vals = N_vals
        self.gb = gb
        self.waveform_kwargs = waveform_kwargs
        self.data_length = data_length
        self.d_d_all = d_d_all

    def __call__(self, x, *args, branch_supps=None, **kwargs):
        if branch_supps is None:
            raise ValueError("Branch supps needed.")

        data_index = branch_supps["gb"]["data_inds"]
        # breakpoint()
        x_in = self.transform_fn["gb"].both_transforms(x, xp=xp, copy=True)

        N = self.N_vals[data_index]
        
        assert data_index.dtype == np.int32

        self.gb.d_d = self.d_d_all[data_index]
        
        if "N" in self.waveform_kwargs:
            self.waveform_kwargs.pop("N")
        # breakpoint()
        ll = self.gb.get_ll(x_in.get(), self.data, self.psd, data_index=data_index, data_length=self.data_length, noise_index=data_index, start_freq_ind=self.start_freq, N=N, **self.waveform_kwargs)

        return ll