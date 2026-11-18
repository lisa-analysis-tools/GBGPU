import shutil
import wave
import numpy as np
import os
from scipy import stats
import h5py 
import time

from eryn.prior import ProbDistContainer
from eryn.ensemble import EnsembleSampler
from eryn.paraensemble import ParaEnsembleSampler
from eryn.utils import PeriodicContainer

from gbgpu.thirdbody import third_body_factors

try:
    import cupy as cp
    from cupy.cuda.runtime import setDevice

    gpu_available = True
except ModuleNotFoundError:
    gpu_available = False

import matplotlib.pyplot as plt

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

from gbmodels import BaseTemplateSetup, ThirdBodyTemplateSetup, PriorTransformFn, LogLikeFn, TemplateSetup
import warnings

from settings import get_settings

from typing import Callable, Union, List, Tuple

from abc import ABC

from eryn.utils.utility import stepping_stone_log_evidence, thermodynamic_integration_log_evidence


class Guide(ABC):

    def __init__(self, name, settings: dict, gpu: int=None):
        self.settings = settings
        self.name = name

        self.gpu = gpu
        if gpu is not None:
            assert gpu_available
            self.use_gpu = True
            self.xp.cuda.runtime.setDevice(gpu)
        else:
            self.use_gpu = False

        self.third_info = ThirdBodyTemplateSetup(use_gpu=self.use_gpu)
        self.base_info = BaseTemplateSetup(use_gpu=self.use_gpu)
        self.template_info = {"third": self.third_info, "base": self.base_info}
        self.df = 1 / self.waveform_kwargs["T"]

    @property
    def xp(self):
        xp = cp if self.use_gpu else np
        return xp

    def get_search_file(self, indicator: str) -> str:
        return self.dir_info["search_dir"] + self.dir_info["base_string"] + f"_search_info_{indicator}.h5"
        
    def get_triples_setup_file(self, indicator: str) -> str:
        return self.dir_info["triples_setup_directory"] + self.dir_info["base_string"] + f"_pop_for_search_{indicator}.dat"

    def get_evidence_file(self, indicator: str) -> str:
        return self.dir_info["evidence_dir"] + self.dir_info["base_string"] + f"_evidence_info_{indicator}.h5"

    def get_pe_file(self, indicator: str) -> str:
        return self.dir_info["pe_dir"] + self.dir_info["base_string"] + f"_pe_info_{indicator}.h5"

    @property
    def verbose(self) -> bool:
        return self.settings["verbose"]

    @property
    def dir_info(self) -> dict:
        return self.settings["dir_info"]

    @property
    def limits_info(self) -> dict:
        return self.settings["limits_info"]

    @property
    def sampler_settings(self) -> dict:
        return self.settings["sampler_settings"]
    
    @property
    def waveform_kwargs(self) -> dict:
        return self.settings["waveform_kwargs"]

    @classmethod
    def get_out_fp(self, indicator: str) -> str:
        raise NotImplementedError

    def run_directory_list(self):
        for directory in self.dir_info["population_directory_list"]:
            if not os.path.exists(directory):
                raise ValueError(f"Directy in population directory list does not exist ({directory}).")

            self.run_single_directory(directory)

    def check_run_status(self, indicator: str) -> bool:
        # if indicator == "A2_s555" or indicator == "A2_s633" or indicator == "C2_s633":
        #     return False
        if not os.path.exists(self.status_file):
            with open(self.status_file, "w") as fp:
                pass
        
        with open(self.status_file, "r") as fp:
            lines = fp.readlines()

        if f"{indicator}\n" in lines:
            # already run 
            return False

        else:
            return True

    def add_to_status_file(self, indicator: str):

        if not os.path.exists(self.status_file):
            with open(self.status_file, "w") as fp:
                pass
        
        with open(self.status_file, "a") as fp:
            fp.write(f"{indicator}\n")

        return

    @property
    def status_file(self) -> str:
        return self.dir_info["main_dir"] + self.dir_info["status_file_base"] + "_" + self.dir_info["base_string"] + f"_{self.name}.txt"

    def retrieve_input_files(self, directory: str):
        for fp in os.listdir(directory):
            if fp[-4:] == ".dat":
                indicator = fp.split("data_")[-1].split(".dat")[0]
                need_to_run = self.check_run_status(indicator)
                if need_to_run:
                    yield (indicator, fp)
                else:
                    if self.verbose:
                        print(f"Not running {fp}. Already in folder.")

    @classmethod
    def run_single_model(self, directory: str, indicator: str, fp: str):
        raise NotImplementedError

    def run_single_directory(self, directory: str):
        for indicator, fp in self.retrieve_input_files(directory):
            self.run_single_model(directory, indicator, fp)

    def get_input_data(self, path_to_input_file: str) -> np.ndarray:
        data = np.genfromtxt(path_to_input_file, dtype=None)
        keys = [
            'id',
            'P',
            'l_deg',
            'b_deg',
            'd_kpc',
            'wd1_mass',
            'wd1_rad',
            'wd2_mass',
            'wd2_rad',
            'age_yr',
            'wd1_cool_time',
            'wd2_cool_time',
            'cos_i',
            'SeBa_id',
            'phi',
            'theta',
            'l_ecl',
            'b_ecl',
            'M3',
            'a3',
            'Phi3',
            'iota3', 
            'e3'
        ]
        dtype = np.dtype([(key, '<f8') for key in keys])
        data_new = np.asarray([tuple(data[i]) for i in range(len(data))], dtype=dtype)

        return data_new

    def get_sampling_data(self, indicator: str) -> np.ndarray:

        search_input_fp = self.dir_info["triples_setup_directory"] + self.dir_info["base_string"] + f"_pop_for_search_{indicator}.dat"
        data = np.genfromtxt(
                search_input_fp,
                delimiter=",",
                names=True,
                dtype=None,
            )

        # transform to sampling basis
        data["Amp"] = np.log(data["Amp"])
        data["f0"] = data["f0"] * 1e3
        data["iota"] = np.cos(data["iota"])
        data["beta"] = np.sin(data["beta"])
        data["A2"] = np.log(data["A2"])
        data["T2"] = data["T2"] / data["P2"]

        return data

    @classmethod
    def form_special_file_indicator(self, orig_id: int):
        raise NotImplementedError

    def check_if_source_has_been_run(self, indicator: str, group_name: str) -> bool:

        out_fp = self.get_out_fp(indicator)

        if os.path.exists(out_fp):
            with h5py.File(out_fp, "r") as fp:
                already_run = group_name in fp
        else:
            already_run = False

        return already_run

    def check_if_parameters_are_within_lims(self, indicator: str, orig_id: int, m3: float, e2: float, ll_diff: float, opt_snr: float, ll_cut=0.0) -> bool:
        good = True

        for key in ["m3", "e2", "opt_snr", "ll_diff"]:    
            if not good:
                continue
            if f"{key}_lims" in self.limits_info:
                limits = self.limits_info[f"{key}_lims"]
                val = locals()[key]
                if (val > limits[1] or val < limits[0]):
                    good = False
                    if self.verbose:
                        print(f"Not running id {orig_id} in model {indicator} because {key} is not inside its defined limits: {val} not in {limits}")

        if ll_diff > ll_cut:
            #VEE  or orig_id < 28577371 and orig_id > 28577360
            # might be some numerical error. 
            # This is making sure it is less than zero.
            assert ll_diff < 1e-5  
            good = False
            if self.verbose:
                print(f"Not running id {orig_id} in model {indicator} because ll_diff is > -2: {ll_diff}")

        return good

    def write_indicator_to_bad_file(self, indicator: str, reason: str="None"):

        if not os.path.exists(self.dir_info["bad_file"]):
            with open(self.dir_info["bad_file"], "w") as fp:
                pass
        
        with open(self.dir_info["bad_file"], "a") as fp:
            fp.write(f"{indicator}: {reason}\n")

    def determine_N(self, injection_params: np.ndarray) -> int:

        amp = np.exp(injection_params[0])
        f0 = injection_params[1] * 1e-3

        N_found_base = get_N(amp, f0, self.waveform_kwargs["T"], oversample=self.waveform_kwargs["oversample"]).item()

        A2 = np.exp(injection_params[-5])
        varpi  = injection_params[-4]
        e2  = injection_params[-3]
        P2 = injection_params[-2]
        T2 = injection_params[-1] * P2

        N_found_third = self.third_info.template_gen.special_get_N(
            amp, 
            f0, 
            self.waveform_kwargs["T"], 
            A2,
            varpi,
            e2,
            P2,
            T2, 
            oversample=self.waveform_kwargs["oversample"]
        )

        N_found = np.max([N_found_base, N_found_third.item()])

        return N_found
        
    def get_lims(self, injection_params: np.ndarray, return_P2_lims: bool=False) -> Tuple:

        f0 = injection_params[1] * 1e-3
        fdot0 = injection_params[2]

        chirp_mass_lims = self.limits_info["chirp_mass_lims"]
        fdot_min = get_fdot(f0, Mc=chirp_mass_lims[0])
        fdot_max = get_fdot(f0, Mc=chirp_mass_lims[1])

        # adjust for the few that are over 1 solar mass
        try:
            assert fdot0 >= fdot_min and fdot0 <= fdot_max
        except AssertionError:
            fdot_max = fdot0 * 1.05
            assert fdot0 >= fdot_min and fdot0 <= fdot_max

        f_min = f0 * 0.999 * 1e3
        f_max = f0 * 1.001 * 1e3
        f_lims = [f_min, f_max]

        fdot_lims = [fdot_min, fdot_max]

        if not return_P2_lims:
            return f_lims, fdot_lims

        else:
            P2 = injection_params[-2]

            min_P2 = P2 * 1e-1 if P2 * 1e-1 < 32.0 else 32.0
            max_P2 = 1e1 * P2 if 1e1 * P2 > 32.0 else 32.0

            if P2 < 2.0:
                max_P2 = 2.2

            P2_lims = [min_P2, max_P2]

            return f_lims, fdot_lims, P2_lims


    def get_injection_data(self, injection_params_with_third: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:

        params_in = self.third_info.transform_fn["gb"].both_transforms(injection_params_with_third[self.third_info.transform_fn["gb"].fill_dict["test_inds"]])
        A_inj, E_inj = self.third_info.template_gen.inject_signal(*params_in, **self.waveform_kwargs)
        return (A_inj, E_inj)

    def get_start_points(self, gb_info: TemplateSetup, injection_params: np.ndarray, f_lims: List, fdot_lims: List, data_channels_tmp: List, psd_tmp: List, start_freq_ind: int, N: int, P2_lims: List=None) -> np.ndarray:

        # [[index of parameter, limits]]
        lims_in = [[1, f_lims], [2, fdot_lims]]
        if P2_lims is not None:
            lims_in.append([-2, P2_lims])

        factor = 1e-5
        cov = np.ones(gb_info.ndim) * 1e-3
        cov[1] = 1e-7
        max_iter = 2000
        start_like = np.zeros((self.ntemps, self.nwalkers))
        while np.std(start_like[0]) < 7.0:
            # TECHNICALLY THIS PRIOR COMPUTATION IS NOT CORRECT FOR LOGPDF
            # DOES NOT MATTER, IT IS START POINTS
            logp = np.full_like(start_like, -np.inf).flatten()
            tmp_fs = np.zeros((self.ntemps * self.nwalkers, gb_info.ndim))
            fix = np.ones((self.ntemps * self.nwalkers), dtype=bool)
            jj = 0
            while jj < max_iter and np.any(fix):
                # left off here. need to fix 
                # - transform function for prior needs to transform output points as well
                tmp_fs[fix] = (injection_params[gb_info.transform_fn["gb"].fill_dict["test_inds"]] * (1. + factor * cov * np.random.randn(self.nwalkers * self.ntemps, gb_info.ndim)))[fix]

                tmp = tmp_fs.copy()
                
                # map points
                for ind, lims in lims_in:
                    tmp[:, ind] = (tmp[:, ind] - lims[0]) / (lims[1] - lims[0])

                # if np.any(tmp[:, 1] < 0.0):
                    # breakpoint()
                logp = gb_info.priors["gb"].logpdf(tmp).get()
                fix = np.isinf(logp)
                jj += 1

            if "N" in self.waveform_kwargs:
                self.waveform_kwargs.pop("N")

            tmp_fs_in = gb_info.transform_fn["gb"].both_transforms(tmp_fs)

            N = int(N)
            ##########
            ### NEED TO START BY COMPARING THE LIKELIHOODS AT THE SAME VALUES OF THE BASE TEMPLATE AND THE THIRD TEMPLATE
            ###########
            # breakpoint()
            start_like = gb_info.template_gen.get_ll(tmp_fs_in, data_channels_tmp, psd_tmp, start_freq_ind=start_freq_ind, N=N, **self.waveform_kwargs)
            
            if np.any(np.isnan(start_like)):
                breakpoint()
            tmp_fs = tmp_fs.reshape(self.ntemps, self.nwalkers, gb_info.ndim)
            start_like = start_like.reshape(self.ntemps, self.nwalkers)
            logp = logp.reshape(self.ntemps, self.nwalkers)
            
            factor *= 1.5
            # print(np.std(start_like[0]))
        # breakpoint()
        return tmp_fs


class TriplesSetupGuide(Guide):

    def __init__(self, *args, N=16384, batch_size=1000, **kwargs):
        name = "triples_setup"
        self.batch_size = batch_size
        self.N = N
        super().__init__(name, *args, **kwargs)

        self.waveform_kwargs["use_c_implementation"] = False

    def build_sampling_parameter_set(self, data_input):
        
        m1 = data_input['wd1_mass']
        m2 = data_input['wd2_mass']
        period = data_input['P'] * 24.0 * 3600.0 # days to seconds
        d_kpc = data_input['d_kpc']
        index_out = np.arange(len(m1))

        f0 = 2 / period
        Amp = get_amplitude(m1, m2, f0, d_kpc)
        fdot = get_fdot(f0, m1=m1, m2=m2)
        fddot = 11/3 * fdot ** 2 / f0
        phi0 = np.random.uniform(0.0, 2* np.pi, size=len(f0))
        iota = np.arccos(data_input['cos_i'])
        psi = np.random.uniform(0, np.pi, size=len(f0))
        lam = data_input['l_ecl']
        beta =data_input['b_ecl']

        # third body comps
        m3 = data_input['M3']
        a2 = data_input['a3'] * AU
        phi2 = data_input['Phi3']
        i2 = data_input['iota3']
        e2 = data_input['e3']
        M_tot_kg = (m1 + m2) * MSUN + m3 * Mjup
        P2 = (2 * np.pi * np.sqrt(a2 ** 3 / (G * M_tot_kg))) / YEAR
        # breakpoint()
        Omega2 = np.random.rand(len(f0)) * 2 * np.pi
        omega2 = np.random.rand(len(f0)) * 2 * np.pi

        A2, varpi, T2 = third_body_factors(
            m1 + m2,
            m3.copy(),
            P2.copy(),
            e2.copy(),
            i2.copy(),
            Omega2,
            omega2, 
            phi2,
            lam,
            beta,
            third_mass_unit="Mjup",
            third_period_unit="yrs",
        )
        out = np.array([
            Amp, 
            f0, 
            fdot, 
            fddot,
            phi0, 
            iota, 
            psi,
            lam,
            beta,
            A2,
            varpi,
            e2,
            P2,
            T2,
            i2,
            phi2,
            Omega2,
            omega2,
        ]).T

        return out

    def build_waveforms(self, gb_gen: Callable, params: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

        waveform_kwargs = self.waveform_kwargs.copy()
        if "N" in waveform_kwargs:
            waveform_kwargs.pop("N")

        if "oversample" in waveform_kwargs:
            waveform_kwargs.pop("oversample")

        gb_gen.run_wave(*params.T, N=self.N, **waveform_kwargs)
        return gb_gen.A, gb_gen.E, gb_gen.freqs

    def get_likelihood_information(self, params: np.ndarray):

        A_third, E_third, freqs_third = self.build_waveforms(self.third_info.template_gen, params[:, :14])
        A_base, E_base, freqs_base = self.build_waveforms(self.base_info.template_gen, params[:, :9])
        assert self.xp.all(freqs_base == freqs_third)

        freqs_third_in_psd = freqs_third if not self.use_gpu else freqs_third.get()
        psd = self.xp.asarray(get_sensitivity(freqs_third_in_psd.flatten(), sens_fn="noisepsd_AE", model="SciRDv1", includewd=self.waveform_kwargs["T"] / YEAR).reshape(freqs_third.shape))

        third_third = 4 * self.df * self.xp.sum((A_third.conj() * A_third + E_third.conj() * E_third) / psd, axis=-1)
        third_base = 4 * self.df * self.xp.sum((A_third.conj() * A_base + E_third.conj() * E_base) / psd, axis=-1)
        base_base = 4 * self.df * self.xp.sum((A_base.conj() * A_base + E_base.conj() * E_base) / psd, axis=-1)
        
        phase_change = self.xp.angle(third_base)
        opt_snr = self.xp.sqrt(third_third.real)
        ll_diff = -1/2 * (third_third + base_base - 2 * self.xp.abs(third_base))
        
        try:
            # get back from gpu if needed
            phase_change = phase_change.get()
            opt_snr = opt_snr.get()
            ll_diff = ll_diff.get()

        except AttributeError:
            pass
        return (ll_diff, opt_snr, phase_change)

    def run_single_model(self, directory: str, indicator: str, fp: str):
        
        if self.verbose:
            print("Start triples setup:", directory, indicator, fp)

        data_input = self.get_input_data(directory + fp)
        
        params_input = self.build_sampling_parameter_set(data_input)
        # breakpoint()
        num_sources = params_input.shape[0]
        inds_tmp = np.arange(0, num_sources, self.batch_size)

        if inds_tmp[-1] != num_sources - 1:
            inds_tmp = np.concatenate([inds_tmp, np.array([num_sources])])

        ll_info, snr_info, phase_change_info = np.zeros(num_sources), np.zeros(num_sources), np.zeros(num_sources)
        for iteration, (start, end) in enumerate(zip(inds_tmp[:-1], inds_tmp[1:])):
            ll_tmp, snr_tmp, phase_change_tmp = self.get_likelihood_information(params_input[start:end]) # used to be start:end, but it was missing a source.

            ll_info[start:end] = ll_tmp
            snr_info[start:end] = snr_tmp
            phase_change_info[start:end] = phase_change_tmp

            if self.verbose:
                print(f"{iteration + 1} of {len(inds_tmp[:-1])}")

        original_id = data_input["id"]
        self.readout(indicator, original_id, params_input, ll_info, snr_info, phase_change_info)

        self.add_to_status_file(indicator)
        if self.verbose:
            print("End triples setup:", directory, indicator, fp)

    def get_out_fp(self, indicator: str) -> str:
        return self.get_triples_setup_file(indicator)

    def readout(self, indicator: str, original_id: np.ndarray, params: np.ndarray, ll_info: np.ndarray, opt_snr: np.ndarray, phase_change: np.ndarray):

        out_fp = self.get_out_fp(indicator)
        # print(np.max(ll_info))
        # breakpoint()
        inds = np.where(ll_info < self.settings["cut_info"]["first_cut_ll_diff_lim"])[0]
        ll_good = ll_info[inds]
        snr_good = opt_snr[inds]
        phase_change_good = phase_change[inds]

        tmp = np.array([ll_good, snr_good, phase_change_good]).T
        params_in = params[inds]

        readout = np.concatenate([params_in, tmp, inds[:, np.newaxis], original_id[inds][:, np.newaxis]], axis=1)

        header = ""

        for key in [
            "Amp", 
            'f0', 
            'fdot', 
            'fddot', 
            'phi0', 
            'iota', 
            'psi',
            'lam',
            'beta',
            'A2',
            'omegabar',
            'e2',
            'P2',
            'T2',
            'i2',
            'phi2',
            'Omega2',
            'omega2',
            'lldiff_base_marg',
            'snr',
            'phase_shift_base',
            'index',
            "orig_id"
            ]:
            header += key + ','

        header = header[:-1]

        if self.verbose:
            print("Saving", out_fp)

        np.savetxt(out_fp, readout, header=header, delimiter=',')

        # test it immediately to make sure it reads in
        check = np.genfromtxt(out_fp, names=True, delimiter=',', dtype=None)


class SearchRuns(Guide):

    def __init__(self, *args, **kwargs):

        name = "search"
        super().__init__(name, *args, **kwargs)

        # store sampler settings as attributes
        for key, val in self.sampler_settings["search"].items():
            setattr(self, key, val)
            
        self.ndim = self.base_info.ndim

        self.initialize_sampler_routine()

    def initialize_sampler_routine(self):

        self.tempering_kwargs = {"ntemps": self.ntemps, "Tmax": np.inf}

        self.prior_transform_fn = PriorTransformFn(self.xp.zeros(self.ngroups), self.xp.ones(self.ngroups), self.xp.zeros(self.ngroups), self.xp.ones(self.ngroups))

        data_channels = [self.xp.zeros(self.data_length * self.ngroups, dtype=complex), self.xp.zeros(self.data_length * self.ngroups, dtype=complex)]
        psds = [self.xp.ones(self.data_length * self.ngroups, dtype=float), self.xp.ones(self.data_length * self.ngroups, dtype=float)]
        start_freq = self.xp.full(self.ngroups, int(1e-3 / self.df), dtype=np.int32)
        N_vals_in = self.xp.zeros(self.ngroups, dtype=int)
        d_d_all = self.xp.zeros(self.ngroups, dtype=float)
        self.N_max = int(self.data_length / 4)
        self.log_like_fn = LogLikeFn(self.base_info.template_gen, data_channels, psds, start_freq, self.df, self.base_info.transform_fn, N_vals_in, self.data_length, d_d_all, **self.waveform_kwargs)
        
        self.currently_running_index_orig_id = [None for _ in range(self.ngroups)]
        
        # initialize sampler
        self.sampler = ParaEnsembleSampler(
            self.base_info.ndim,
            self.nwalkers,
            self.ngroups,
            self.log_like_fn,
            self.base_info.priors,
            tempering_kwargs=self.tempering_kwargs,
            args=[],
            kwargs={},
            gpu=self.gpu,
            periodic=self.base_info.periodic,
            backend=None,
            update_fn=None,
            update_iterations=-1,
            stopping_fn=None,
            stopping_iterations=-1,
            name="gb",
            prior_transform_fn=self.prior_transform_fn,
            provide_supplemental=True,
        )

        coords = self.xp.zeros((self.ngroups, self.ntemps, self.nwalkers, self.ndim))

        branch_supp_base_shape = (self.ngroups, self.ntemps, self.nwalkers)

        data_inds = self.xp.repeat(self.xp.arange(self.ngroups, dtype=np.int32)[:, None], self.ntemps * self.nwalkers, axis=-1).reshape(self.ngroups, self.ntemps, self.nwalkers) 
        branch_supps = {"gb": BranchSupplemental(
            {"data_inds": data_inds}, base_shape=branch_supp_base_shape, copy=True
        )}

        groups_running = self.xp.zeros(self.ngroups, dtype=bool)
        self.start_state = ParaState({"gb": coords}, groups_running=groups_running, branch_supplemental=branch_supps)
        self.start_state.log_prior = self.xp.zeros((self.ngroups, self.ntemps, self.nwalkers))
        self.start_state.log_like = self.xp.zeros((self.ngroups, self.ntemps, self.nwalkers))
        self.start_state.betas = self.xp.ones((self.ngroups, self.ntemps))

    def run_single_model(self, directory: str, indicator: str, fp: str):
        
        if self.verbose:
            print(f"Start {self.name}:", directory, indicator, fp)

        data_input = self.get_input_data(directory + fp)
        data_search = self.get_sampling_data(indicator)

        self.run_mcmc(indicator, data_search, data_input)

        self.add_to_status_file(indicator)

    def get_out_fp(self, indicator: str) -> str:
        return self.get_search_file(indicator)

    def information_generator(self, indicator: str, data_search: np.ndarray, data_orig: np.ndarray):
        nbin = len(data_search)
        for i in range(nbin):
            index = int(data_search["index"][i])
            orig_id = int(data_search["orig_id"][i])

            already_run = self.check_if_source_has_been_run(indicator, str(orig_id))

            if already_run:
                if self.verbose:
                    print(f"{int(orig_id)} already in file {self.get_out_fp(indicator)} so not running.")

                continue
        
            # check limits
            m3 = data_orig["M3"][int(index)]
            e2 = data_search["e2"][i]
            ll_diff_setup = data_search["lldiff_base_marg"][i]
            opt_snr = data_search["snr"][i]
            phase_shift = data_search["phase_shift_base"][i]

            inside_limits = self.check_if_parameters_are_within_lims(indicator, orig_id, m3, e2, ll_diff_setup, opt_snr, ll_cut=self.settings["cut_info"]["first_cut_ll_diff_lim"])
            if not inside_limits:
                continue

            injection_params = np.array([data_search[key][i] for key in data_search.dtype.names[:14]])
            injection_params[7] = injection_params[7] % (2 * np.pi)

            template_params = injection_params[:9].copy()
            
            # maximized phase shift
            template_params[4] = (template_params[4] + phase_shift) % (2 * np.pi)

            N_found = self.determine_N(injection_params)
            # breakpoint()
            if N_found > self.N_max:
                if self.verbose:
                    print(f"ID {int(orig_id)} (index: {int(index)}) has too high of N value so not running.")
                self.write_indicator_to_bad_file(indicator, reason="too high of N")
                continue

            waveform_kwargs = self.waveform_kwargs.copy()

            waveform_kwargs.pop("oversample")
            waveform_kwargs["N"] = N_found

            f_lims, fdot_lims = self.get_lims(injection_params)

            A_inj, E_inj = self.get_injection_data(injection_params)

            f0 = injection_params[1] / 1e3
            start_freq = int(int(f0 / self.df) - self.data_length / 2)
            fd = np.arange(start_freq, start_freq + self.data_length) * self.df

            data_channels = [A_inj[start_freq:start_freq + self.data_length].copy(), E_inj[start_freq:start_freq + self.data_length].copy()]

            AE_psd = get_sensitivity(fd, sens_fn="noisepsd_AE", model="SciRDv1", includewd=self.waveform_kwargs["T"] / YEAR)
            psd = [AE_psd, AE_psd]

            info_out = {name: value for name, value in zip(data_search.dtype.names, data_search[i])}
            yield (orig_id, index, injection_params, fd, data_channels, psd, start_freq, f_lims, fdot_lims, N_found, info_out)
        
    def setup_next_source(self, info_iterator: Callable, indicator: str, data_search: np.ndarray, data_orig: np.ndarray):
        try:
            (orig_id, index, injection_params, fd, data_channels_tmp, psd_tmp, start_freq_ind, f_lims, fdot_lims, N_val, keep_info) = next(info_iterator)
            
            data_channels_tmp = [self.xp.asarray(tmp) for tmp in data_channels_tmp]
            psd_tmp = [self.xp.asarray(tmp) for tmp in psd_tmp]

            # get injection inner product 
            d_d = 4.0 * self.df * self.xp.sum(self.xp.asarray(data_channels_tmp).conj() * self.xp.asarray(data_channels_tmp) / self.xp.asarray(psd_tmp)).item().real
            # breakpoint()
            self.base_info.template_gen.d_d = d_d
            
            start_points = self.get_start_points(self.base_info, injection_params, f_lims, fdot_lims, data_channels_tmp, psd_tmp, start_freq_ind, N_val)
            # setup in ParaState

            # get first group not running
            new_group_ind = self.start_state.groups_running.argmin().item()
            self.start_state.groups_running[new_group_ind] = True

            self.start_state.branches["gb"].coords[new_group_ind] = self.xp.asarray(start_points)
            # self.start_state.log_prior[new_group_ind] = xp.asarray(logp)
            # self.start_state.log_like[new_group_ind] = xp.asarray(start_like)
            self.start_state.betas[new_group_ind] = self.xp.asarray(self.sampler.base_temperature_control.betas)

            inds_slice = slice((new_group_ind) * self.data_length, (new_group_ind + 1) * self.data_length, 1)
            self.sampler.log_like_fn.data[0][inds_slice] = data_channels_tmp[0]
            self.sampler.log_like_fn.data[1][inds_slice] = data_channels_tmp[1]
            self.sampler.log_like_fn.psd[0][inds_slice] = psd_tmp[0]
            self.sampler.log_like_fn.psd[1][inds_slice] = psd_tmp[1]
            self.sampler.log_like_fn.start_freq[new_group_ind] = start_freq_ind

            self.sampler.prior_transform_fn.f_min[new_group_ind] = f_lims[0]
            self.sampler.prior_transform_fn.f_max[new_group_ind] = f_lims[1]
            self.sampler.prior_transform_fn.fdot_min[new_group_ind] = fdot_lims[0]
            self.sampler.prior_transform_fn.fdot_max[new_group_ind] = fdot_lims[1]
            self.sampler.log_like_fn.N_vals[new_group_ind] = N_val
            self.currently_running_index_orig_id[new_group_ind] = orig_id

            self.sampler.log_like_fn.d_d_all[new_group_ind] = d_d

            self.output_info_store[new_group_ind] = keep_info

            return False

        except StopIteration:
            return True

    def readout(self, end_i: int, indicator: str):
        output_state = State({"gb": self.start_state.branches["gb"].coords[end_i].get()}, log_like=self.start_state.log_like[end_i].get(), log_prior=self.start_state.log_prior[end_i].get(), betas=self.start_state.betas[end_i].get(), random_state=np.random.get_state())

        orig_id = self.currently_running_index_orig_id[end_i]
                    
        backend_tmp = HDFBackend(self.get_out_fp(indicator), name=str(int(orig_id)))
        
        backend_tmp.reset(
            self.nwalkers,
            self.base_info.ndim,
            ntemps=self.ntemps,
            branch_names=["gb"],
        )
        backend_tmp.grow(1, None)
        
        # meaningless array needed for saving input
        accepted = np.zeros((self.ntemps, self.nwalkers), dtype=bool)
        backend_tmp.save_step(output_state, accepted)

        # add other information to file
        with h5py.File(backend_tmp.filename, "a") as fp:
            group_new = fp[str(int(orig_id))].create_group("keep_info")
            for key, value in self.output_info_store[end_i].items():
                group_new.attrs[key] = value
            group_new.attrs["logl_max_mcmc"] = output_state.log_like.max()

        return
       
    def run_mcmc(self, indicator: str, data_search: np.ndarray, data_input: np.ndarray):

        info_iterator = self.information_generator(indicator, data_search, data_input)

        max_log_like = self.xp.full((self.ngroups,), -np.inf)
        now_max_log_like = self.xp.full((self.ngroups,), -np.inf)
        iters_at_max = self.xp.zeros((self.ngroups,), dtype=int)
        self.output_info_store = [None for _ in range(self.ngroups)]
        
        convergence_iter_count = self.sampler_settings["search"]["convergence_iter_count"]
        
        run = True
        finish_up = False

        while run:
            finish_up = self.setup_next_source(info_iterator, indicator, data_search, data_input)

            # end if all are done
            if finish_up and np.all(~self.start_state.groups_running):
                run = False
                return

            started_run = False
            running_inner = (self.xp.all(self.start_state.groups_running) or finish_up)
            while running_inner:
                started_run = True

                self.start_state.log_like = None
                self.start_state.log_prior = None
                self.start_state = self.sampler.run_mcmc(self.start_state, self.sampler_settings["search"]["nsteps_per_check"], progress=self.sampler_settings["search"]["progress"], store=False)

                now_max_log_like[self.start_state.groups_running] = self.start_state.log_like.max(axis=(1, 2))[(self.start_state.groups_running)]
                improved = (now_max_log_like > max_log_like)

                iters_at_max[(improved) & (self.start_state.groups_running)] = 0
                iters_at_max[(~improved) & (self.start_state.groups_running)] += 1
                max_log_like[(improved) & (self.start_state.groups_running)] = now_max_log_like[(improved) & (self.start_state.groups_running)]

                converged = iters_at_max > convergence_iter_count

                end = converged | (now_max_log_like > -2.0)

                if np.any(end):
                    running_inner = False

                self.start_state.groups_running[end] = False
                # print(iters_at_max, start_state.groups_running.sum().item(), now_max_log_like[:10])
            
            if started_run:
                # which groups ended
                end = np.where(end.get())[0]
                for end_i in end:

                    # reset values
                    max_log_like[end_i] = -np.inf
                    now_max_log_like[end_i] = -np.inf
                    converged[end_i] = False
                    iters_at_max[end_i] = 0

                    # readout the source that finished
                    self.readout(end_i, indicator)

                    # reset running id
                    self.currently_running_index_orig_id[end_i] = None

                    self.xp.get_default_memory_pool().free_all_blocks()

            if self.xp.all(~self.start_state.groups_running) and finish_up:
                run = False

    def form_special_file_indicator(self, orig_id: int, **kwargs):
        return str(orig_id)


class EvidenceRuns(Guide):

    def __init__(self, *args, **kwargs):

        name = "evidence"
        super().__init__(name, *args, **kwargs)

        # store sampler settings as attributes
        for key, val in self.sampler_settings["evidence"].items():
            setattr(self, key, val)
            
        self.ndim = self.base_info.ndim

        self.initialize_sampler_routine()

    def initialize_sampler_routine(self):

        # adjust_betas
        ntemps_over_5 = int(self.ntemps / 5)

        ntemps_1 = 4 * ntemps_over_5
        ntemps_2 = self.ntemps - ntemps_1

        betas_1 = np.logspace(-4, 0, ntemps_1)[::-1]
        betas_2 = np.logspace(-10, -4, ntemps_2 - 1, endpoint=False)[::-1]

        betas = np.concatenate([betas_1, betas_2, np.array([0.0])])
        
        self.tempering_kwargs = {"betas": betas, "adaptive": False}
        
        self.prior_transform_fn = {}
        self.prior_transform_fn["third"] = PriorTransformFn(
            self.xp.zeros(self.ngroups),
            self.xp.ones(self.ngroups), 
            self.xp.zeros(self.ngroups), 
            self.xp.ones(self.ngroups), 
            P2_min=self.xp.ones(self.ngroups), 
            P2_max=self.xp.ones(self.ngroups)
        )

        self.prior_transform_fn["base"] = PriorTransformFn(
            self.xp.zeros(self.ngroups),
            self.xp.ones(self.ngroups), 
            self.xp.zeros(self.ngroups), 
            self.xp.ones(self.ngroups)
        )

        data_channels = [self.xp.zeros(self.data_length * self.ngroups, dtype=complex), self.xp.zeros(self.data_length * self.ngroups, dtype=complex)]
        psds = [self.xp.ones(self.data_length * self.ngroups, dtype=float), self.xp.ones(self.data_length * self.ngroups, dtype=float)]
        start_freq = self.xp.full(self.ngroups, int(1e-3 / self.df), dtype=np.int32)
        N_vals_in = self.xp.zeros(self.ngroups, dtype=int)
        d_d_all = self.xp.zeros(self.ngroups, dtype=float)
        self.N_max = int(self.data_length / 4)
        self.log_like_fn = {
            self.base_info.name: LogLikeFn(self.base_info.template_gen, data_channels, psds, start_freq, self.df, self.base_info.transform_fn, N_vals_in, self.data_length, d_d_all, **self.waveform_kwargs),
            self.third_info.name: LogLikeFn(self.third_info.template_gen, data_channels, psds, start_freq, self.df, self.third_info.transform_fn, N_vals_in, self.data_length, d_d_all, **self.waveform_kwargs)
        }
        self.currently_running_index_orig_id = [None for _ in range(self.ngroups)]
        
        self.currently_running_orig_id = [None for _ in range(self.ngroups)]
        self.currently_running_output_info = [None for _ in range(self.ngroups)]
        
        self.start_state = {}
        self.sampler = {}

        for info in [self.base_info, self.third_info]:
            coords = self.xp.zeros((self.ngroups, self.ntemps, self.nwalkers, info.ndim))

            branch_supp_base_shape = (self.ngroups, self.ntemps, self.nwalkers)

            data_inds = self.xp.repeat(self.xp.arange(self.ngroups, dtype=np.int32)[:, None], self.ntemps * self.nwalkers, axis=-1).reshape(self.ngroups, self.ntemps, self.nwalkers) 
            branch_supps = {"gb": BranchSupplemental(
                {"data_inds": data_inds}, base_shape=branch_supp_base_shape, copy=True
            )}

            groups_running = self.xp.zeros(self.ngroups, dtype=bool)
        
            start_state = ParaState({"gb": coords}, groups_running=groups_running, branch_supplemental=branch_supps)
            start_state.log_prior = self.xp.zeros((self.ngroups, self.ntemps, self.nwalkers))
            start_state.log_like = self.xp.zeros((self.ngroups, self.ntemps, self.nwalkers))
            start_state.betas = self.xp.ones((self.ngroups, self.ntemps))
            self.start_state[info.name] = start_state

            # initialize sampler
            self.sampler[info.name] = ParaEnsembleSampler(
                info.ndim,
                self.nwalkers,
                self.ngroups,
                self.log_like_fn[info.name],
                info.priors,
                tempering_kwargs=self.tempering_kwargs,
                args=[],
                kwargs={},
                gpu=self.gpu,
                periodic=info.periodic,
                backend=None,
                update_fn=None,
                update_iterations=-1,
                stopping_fn=None,
                stopping_iterations=-1,
                name="gb",
                prior_transform_fn=self.prior_transform_fn[info.name],
                provide_supplemental=True,
            )

        self.map_models = {"base": 0, "third": 1}

        self.current_model_index_product_space = self.xp.random.randint(2, size=(self.ngroups, self.ntemps, self.nwalkers))

    def run_single_model(self, directory: str, indicator: str, fp: str):
        
        if self.verbose:
            print(f"Start {self.name}:", directory, indicator, fp)

        data_input = self.get_input_data(directory + fp)
        data_search = self.get_sampling_data(indicator)
        #VEE Data_input = data_input[data_input["id"] > 22451709]
        #VEE data_search  = data_search[data_search["orig_id"] > 22451709]
        self.run_mcmc(indicator, data_search, data_input)

        self.add_to_status_file(indicator)

    def get_out_fp(self, indicator: str) -> str:
        return self.get_evidence_file(indicator)

    def get_last_search_state(self, indicator: str, orig_id: int) -> State:

        reader_search = HDFBackend(self.get_search_file(indicator), name=str(int(orig_id)))
        
        try:
            last_state = reader_search.get_last_sample()
        except KeyError:
            last_state = None

        return last_state

    def get_output_info(self, indicator: str, orig_id: int) -> dict:
        with h5py.File(self.get_search_file(indicator), "r") as f:
            output_info = {name: f[str(int(orig_id))]["keep_info"].attrs[name] for name in f[str(int(orig_id))]["keep_info"].attrs}
        return output_info

    def information_generator(self, indicator: str, data_search: np.ndarray, data_orig: np.ndarray):
        nbin = len(data_search)
        for i in range(nbin):
            index = int(data_search["index"][i])
            orig_id = int(data_search["orig_id"][i])
            
            already_run = self.check_if_source_has_been_run(indicator, str(orig_id) + "_third")
                
            if already_run:
                if self.verbose:
                    print(f"{int(orig_id)} already in file {self.get_out_fp(indicator)} so not running.")

                continue
        
            # check limits
            m3 = data_orig["M3"][int(index)]
            e2 = data_search["e2"][i]
            # ll_diff_setup = data_search["lldiff_base_marg"][i]
            opt_snr = data_search["snr"][i]
            phase_shift = data_search["phase_shift_base"][i]

            last_state = self.get_last_search_state(indicator, orig_id)

            if last_state is None:
                if self.verbose:
                    print(f"{int(orig_id)} not in file {self.get_search_file(indicator)} so not running.")
                continue
            
            ll_diff_after_search = last_state.log_like.max()

            inside_limits = self.check_if_parameters_are_within_lims(indicator, orig_id, m3, e2, ll_diff_after_search, opt_snr, ll_cut=self.settings["cut_info"]["second_cut_ll_diff_lim"])

            if not inside_limits:
                continue

            injection_params = np.array([data_search[key][i] for key in data_search.dtype.names[:14]])
            injection_params[7] = injection_params[7] % (2 * np.pi)

            template_params = injection_params[:9].copy()
            
            # maximized phase shift
            template_params[4] = (template_params[4] + phase_shift) % (2 * np.pi)

            N_found = self.determine_N(injection_params)
            
            if N_found > self.N_max:
                if self.verbose:
                    print(f"ID {int(orig_id)} (index: {int(index)}) has too high of N value so not running.")
                self.write_indicator_to_bad_file(indicator, reason="too high of N")
                continue

            waveform_kwargs = self.waveform_kwargs.copy()

            waveform_kwargs.pop("oversample")
            waveform_kwargs["N"] = N_found

            f_lims, fdot_lims, P2_lims = self.get_lims(injection_params, return_P2_lims=True)

            A_inj, E_inj = self.get_injection_data(injection_params)

            f0 = injection_params[1] / 1e3
            start_freq = int(int(f0 / self.df) - self.data_length / 2)
            fd = np.arange(start_freq, start_freq + self.data_length) * self.df

            data_channels = [A_inj[start_freq:start_freq + self.data_length].copy(), E_inj[start_freq:start_freq + self.data_length].copy()]

            AE_psd = get_sensitivity(fd, sens_fn="noisepsd_AE", model="SciRDv1", includewd=self.waveform_kwargs["T"] / YEAR)
            psd = [AE_psd, AE_psd]

            info_out = self.get_output_info(indicator, orig_id)

            yield (orig_id, index, injection_params, fd, data_channels, psd, start_freq, f_lims, fdot_lims, P2_lims, N_found, last_state, info_out)

    def setup_next_source(self, info_iterator: Callable, indicator: str, data_search: np.ndarray, data_orig: np.ndarray):
        try:
            (orig_id, index, injection_params, fd, data_channels_tmp, psd_tmp, start_freq_ind, f_lims, fdot_lims, 
            P2_lims, N_val, last_state, keep_info) = next(info_iterator)
            
            
            data_channels_tmp = [self.xp.asarray(tmp) for tmp in data_channels_tmp]
            psd_tmp = [self.xp.asarray(tmp) for tmp in psd_tmp]

            # get injection inner product 
            d_d = 4.0 * self.df * self.xp.sum(self.xp.asarray(data_channels_tmp).conj() * self.xp.asarray(data_channels_tmp) / self.xp.asarray(psd_tmp)).item().real

            self.base_info.template_gen.d_d = d_d
            self.third_info.template_gen.d_d = d_d
            
            start_points = self.get_start_points(self.third_info, injection_params, f_lims, fdot_lims, data_channels_tmp, psd_tmp, start_freq_ind, N_val, P2_lims=P2_lims)
            # setup in ParaState

            # get first group not running
            new_group_ind = self.start_state["third"].groups_running.argmin().item()
            self.start_state["third"].groups_running[new_group_ind] = True

            self.start_state["third"].branches["gb"].coords[new_group_ind] = self.xp.asarray(start_points)
            # self.start_state["third"].log_prior[new_group_ind] = xp.asarray(logp)
            # self.start_state["third"].log_like[new_group_ind] = xp.asarray(start_like)
            # self.start_state["third"].betas[new_group_ind] = xp.asarray(betas_in_here)

            # if np.any(np.isnan(self.start_state["third"].log_like)):
            #     breakpoint()

            # data arrays are the same between third and base so no need to adjust base
            # inds_slice indexes into the flattened data arrays
            inds_slice = slice((new_group_ind) * self.data_length, (new_group_ind + 1) * self.data_length, 1)
            self.sampler["third"].log_like_fn.data[0][inds_slice] = data_channels_tmp[0]
            self.sampler["third"].log_like_fn.data[1][inds_slice] = data_channels_tmp[1]
            self.sampler["third"].log_like_fn.psd[0][inds_slice] = psd_tmp[0]
            self.sampler["third"].log_like_fn.psd[1][inds_slice] = psd_tmp[1]
            self.sampler["third"].log_like_fn.start_freq[new_group_ind] = start_freq_ind

            self.sampler["third"].prior_transform_fn.f_min[new_group_ind] = f_lims[0]
            self.sampler["third"].prior_transform_fn.f_max[new_group_ind] = f_lims[1]
            self.sampler["third"].prior_transform_fn.fdot_min[new_group_ind] = fdot_lims[0]
            self.sampler["third"].prior_transform_fn.fdot_max[new_group_ind] = fdot_lims[1]
            self.sampler["third"].prior_transform_fn.P2_min[new_group_ind] = P2_lims[0]
            self.sampler["third"].prior_transform_fn.P2_max[new_group_ind] = P2_lims[1]
            self.sampler["third"].log_like_fn.N_vals[new_group_ind] = N_val
            self.sampler["third"].log_like_fn.d_d_all[new_group_ind] = d_d

            self.sampler["base"].log_like_fn.start_freq[new_group_ind] = start_freq_ind
            self.sampler["base"].prior_transform_fn.f_min[new_group_ind] = f_lims[0]
            self.sampler["base"].prior_transform_fn.f_max[new_group_ind] = f_lims[1]
            self.sampler["base"].prior_transform_fn.fdot_min[new_group_ind] = fdot_lims[0]
            self.sampler["base"].prior_transform_fn.fdot_max[new_group_ind] = fdot_lims[1]
            self.sampler["base"].log_like_fn.N_vals[new_group_ind] = N_val
            self.sampler["base"].log_like_fn.d_d_all[new_group_ind] = d_d

            # get info from search

            # get first group not running
            self.start_state["base"].groups_running[new_group_ind] = True

            self.start_state["base"].branches["gb"].coords[new_group_ind] = self.xp.asarray(last_state.branches["gb"].coords[0, :self.nwalkers, 0][None, :])
            # self.start_state["base"].log_prior[new_group_ind] = xp.asarray(last_search_sample.log_prior[0][None, :self.nwalkers])
            # self.start_state["base"].log_like[new_group_ind] = xp.asarray(last_search_sample.log_like[0][None, :self.nwalkers])
            # self.start_state["base"].betas[new_group_ind] = xp.asarray(betas_in_here)

            # if np.any(np.isnan(self.start_state["base"].log_like)):
            #     breakpoint()

            self.currently_running_orig_id[new_group_ind] = orig_id
            self.currently_running_output_info[new_group_ind] = keep_info

            return False

        except StopIteration:
            return True

    # def readout(self, end_i: int, indicator: str):
    #     output_state = State({"gb": self.start_state.branches["gb"].coords[end_i].get()}, log_like=self.start_state.log_like[end_i].get(), log_prior=self.start_state.log_prior[end_i].get(), betas=self.start_state.betas[end_i].get(), random_state=np.random.get_state())

    #     orig_id = self.currently_running_index_orig_id[end_i]
                    
    #     backend_tmp = HDFBackend(self.get_out_fp(indicator), name=str(int(orig_id)))
        
    #     backend_tmp.reset(
    #         self.nwalkers,
    #         self.base_info.ndim,
    #         ntemps=self.ntemps,
    #         branch_names=["gb"],
    #     )
    #     backend_tmp.grow(1, None)
        
    #     # meaningless array needed for saving input
    #     accepted = np.zeros((self.ntemps, self.nwalkers), dtype=bool)
    #     backend_tmp.save_step(output_state, accepted)

    #     # add other information to file
    #     with h5py.File(backend_tmp.filename, "a") as fp:
    #         group_new = fp[str(int(orig_id))].create_group("keep_info")
    #         for key, value in self.output_info_store[end_i].items():
    #             group_new.attrs[key] = value
    #         group_new.attrs["logl_max_mcmc"] = output_state.log_like.max()

    #     return

    def readout(self, end_i: int, indicator: str, current_evidence_estimate: dict, timed_out, counter_few):

        base_evidence = current_evidence_estimate["base"][end_i].item()
        third_evidence = current_evidence_estimate["third"][end_i].item()

        two_logBF = 2 * (third_evidence - base_evidence)

        orig_id = self.currently_running_orig_id[end_i]

        for template in [self.third_info, self.base_info]:
            group_name = str(int(orig_id)) + "_" + template.name
            backend_tmp = HDFBackend(self.get_evidence_file(indicator), name=group_name)
            backend_tmp.reset( self.nwalkers, template.ndim, ntemps=self.ntemps, branch_names=["gb"],)
            backend_tmp.grow(self.sampler[template.name].backend.iteration, None)

            with h5py.File(backend_tmp.filename, "a") as fp_save:
                fp_save[group_name].attrs["iteration"] = self.sampler[template.name].iteration
                fp_save[group_name]["chain"]["gb"][:] = self.sampler[template.name].get_chain()[:, end_i][:, :, :, None, :]
                fp_save[group_name]["log_like"][:] = self.sampler[template.name].get_log_like()[:, end_i]
                fp_save[group_name]["log_prior"][:] = self.sampler[template.name].get_log_prior()[:, end_i]
                group_new = fp_save[group_name].create_group("keep_info")
                for key, value in self.currently_running_output_info[end_i].items():
                    group_new.attrs[key] = value

                group_new.attrs["base_evidence"] = base_evidence
                group_new.attrs["third_evidence"] = third_evidence
                group_new.attrs["2logBF"] = two_logBF

                group_new.attrs["converged"] = bool(not timed_out[end_i])
                group_new.attrs["counter_few"] = int(counter_few[end_i])
    
    def product_space_operation(self, logP_dict):

        p_base_to_third = self.sampler_settings["evidence"]["p_base_to_third"]
        p_third_to_base = 1.0 - p_base_to_third

        base_to_third_forward_indicator = (2 * (self.current_model_index_product_space != 0).astype(int) - 1)
        # forward should be minus for proposal weight
        ratio = (
            np.log(p_base_to_third) * base_to_third_forward_indicator
            + np.log(p_third_to_base) * -1 * base_to_third_forward_indicator
            + base_to_third_forward_indicator * (logP_dict["third"] - logP_dict["base"])
        )

        accept = self.xp.log(self.xp.random.rand(*ratio.shape)) < ratio

        self.current_model_index_product_space[accept] = (self.current_model_index_product_space[accept] + 1) % 2
             
    def run_mcmc(self, indicator: str, data_search: np.ndarray, data_input: np.ndarray):
        test_burn = True
        info_iterator = self.information_generator(indicator, data_search, data_input)
        run = True
        finish_up = False

        total_steps_for_evidence = self.sampler_settings["evidence"]["total_steps_for_evidence"]
        number_old_evidences = self.sampler_settings["evidence"]["number_old_evidences"]
        thin_by = self.sampler_settings["evidence"]["thin_by"]

        old_bf = np.zeros((self.ngroups, number_old_evidences))
        current_old_bf_ind = np.zeros((self.ngroups,), dtype=int)
        current_old_bf_count = np.zeros((self.ngroups,), dtype=int)

        current_evidence_ind = 0
        current_evidence_estimate = {name: self.xp.full((self.ngroups,), np.nan) for name in ["base", "third"]}
        current_evidence_diff = self.xp.full((self.ngroups,), np.nan)
        nsteps = self.sampler_settings["evidence"]["nsteps"]
                
        evidence_log_like = {name: self.xp.full((self.ngroups, total_steps_for_evidence, self.ntemps, self.nwalkers * nsteps), self.xp.nan) for name in ["base", "third"]}        

        counter_few = np.zeros(self.ngroups, dtype=int)
        timed_out = np.zeros(self.ngroups, dtype=bool)
        max_few_iterations = self.sampler_settings["evidence"].get("max_few_iterations", 150)
        
        while run:
            finish_up = self.setup_next_source(info_iterator, indicator, data_search, data_input)
            # end if all are done
            if finish_up and np.all(~self.start_state["third"].groups_running):
                run = False
                return
            
            # end if only one remaining

            started_run = False
            running_inner = (self.xp.all(self.start_state["third"].groups_running) or finish_up)
            while running_inner:
                # breakpoint()
                started_run = True
                assert nsteps <= total_steps_for_evidence

                current_inds_fill_evidence = (self.xp.arange(nsteps) + current_evidence_ind) % total_steps_for_evidence

                logP = {}
                for template in ["third", "base"]:
                    tmp = self.sampler[template].compute_log_prior(self.start_state[template].branches_coords)
                    self.start_state[template].log_like = None
                    self.start_state[template].log_prior = None
                    self.start_state[template].betas = None
                    self.sampler[template].backend.reset(*self.sampler[template].backend.reset_args, **self.sampler[template].backend.reset_kwargs)
                    # if template == "third":
                    #     breakpoint()
                    #     coords_in = {"gb": self.start_state[template].branches["gb"].coords[-1:]}
                    #     check = self.sampler[template].compute_log_prior(coords_in, groups_running=self.xp.arange(self.ngroups)[np.array([15])])
                    #     coords_in2 = {"gb": self.start_state[template].branches["gb"].coords}
                    #     check = self.sampler[template].compute_log_prior(coords_in, groups_running=self.xp.arange(self.ngroups)[:])
                    #     breakpoint()

                    print(template, self.start_state[template].groups_running.sum())
                    
                    print(self.currently_running_orig_id)
                    # burn = 0
                    # if test_burn:
                    #     burn = 0
                    #     if template == "base":
                    #         test_burn = False
                    burn = 1000 if test_burn else 0
                    if template == "base":
                        test_burn = False

                    self.start_state[template] = self.sampler[template].run_mcmc(self.start_state[template], nsteps, burn=burn, thin_by=thin_by, progress=True, store=True)
                    # np.save(f"sample_check_{template}", self.sampler[template].get_chain())
                    
                    logP[template] = self.start_state[template].log_like * self.start_state[template].betas[:, :, None] + self.start_state[template].log_prior

                    evidence_log_like[template][:, current_evidence_ind] = self.xp.asarray(self.sampler[template].get_log_like(discard=self.sampler[template].backend.iteration - nsteps).transpose(1, 2, 3, 0).reshape(self.ngroups, self.ntemps, self.nwalkers * nsteps))  # self.start_state[template].log_like.mean(axis=-1)[:, None, :]
                    # evidence_log_like[template] = 
                # np.save(file='log_like_third_backend_sampler_new.npy', arr=self.sampler['third'].backend.get_log_like())
                # np.save(file='log_like_base_backend_sampler_new.npy', arr=self.sampler['base'].backend.get_log_like())
                # np.save(file='betas_third_backend_sampler_new.npy', arr=self.sampler['third'].backend.get_betas())
                # np.save(file='betas_base_backend_sampler_new.npy', arr=self.sampler['base'].backend.get_betas())
                # breakpoint()
                #access likelihood points in self.sampler[template] - call self.sampler[template].get_log_like for both templates aand call get
                # temperature. store likelihood and temperature for each step walker. 
                # exit()
                # self.product_space_operation(logP)
                
                current_evidence_ind += 1
                current_evidence_ind %= total_steps_for_evidence

                # adjust
                adjust = self.xp.all(~self.xp.isnan(evidence_log_like["third"]), axis=(1, 2, 3))
                if self.xp.any(adjust):
                    for template in ["third", "base"]:
                        
                        assert self.xp.all(self.start_state["base"].betas == self.start_state["third"].betas)
                        for grp in range(self.ngroups):
                            if adjust[grp]:
                                betas = self.start_state["third"].betas[grp]
                                logls = evidence_log_like[template][grp].mean(axis=(0, 2))
                                logZ, dlogZ = thermodynamic_integration_log_evidence(betas.get(), logls.get())  # , block_len=50, repeats=100)
                                current_evidence_estimate[template][grp] = logZ.item()

                    current_evidence_diff[adjust] = (current_evidence_estimate["third"] - current_evidence_estimate["base"])[adjust]
                    print(current_evidence_diff)

                    old_bf[adjust.get(), current_old_bf_ind[adjust.get()]] = 2 * current_evidence_diff[adjust.get()].get()
                    current_old_bf_ind[adjust.get()] = (current_old_bf_ind[adjust.get()] + 1) % old_bf.shape[1]
                    current_old_bf_count[adjust.get()] += 1

                # if self.start_state["base"].groups_running.sum() < 3 and self.start_state["third"].groups_running.sum() < 3:
                #     counter_few += 1 
                #     check_convergence = (current_old_bf_count >= number_old_evidences) & (np.asarray(self.currently_running_orig_id) != None)
                #     end = np.full(self.ngroups, False)
                #     end[check_convergence] = np.all(np.abs(old_bf[check_convergence] - np.mean(old_bf[check_convergence], axis=-1)[:, None]) < 0.03, axis=-1) & (np.abs(np.sign(np.diff(old_bf[check_convergence], axis=-1)).sum(axis=-1)) < number_old_evidences - 1)
                #     print(end, current_evidence_diff)
                #     print(self.currently_running_orig_id)
                #     if counter_few > 150:
                #         print("counter_few", counter_few)
                #         groups_ind = np.where(self.start_state["base"].groups_running)[0]
                #         # groups_val = [orig for orig in self.currently_running_orig_id if orig is not None]
                #         for ind in groups_ind:
                #             current_evidence_estimate["base"][int(ind)] = np.nan
                #             current_evidence_estimate["third"][int(ind)] = np.nan
                #             print(ind)
                #             end[int(ind)] = True
                #     else:
                #         print("counter_few", counter_few)
                # else:
                #     counter_rest +=1
                #     check_convergence = (current_old_bf_count >= number_old_evidences) & (np.asarray(self.currently_running_orig_id) != None)
                #     end = np.full(self.ngroups, False)
                #     end[check_convergence] = np.all(np.abs(old_bf[check_convergence] - np.mean(old_bf[check_convergence], axis=-1)[:, None]) < 0.03, axis=-1) & (np.abs(np.sign(np.diff(old_bf[check_convergence], axis=-1)).sum(axis=-1)) < number_old_evidences - 1)
                #     print(end, current_evidence_diff)
                #     print("counter_rest", counter_rest)
                running = self.start_state["third"].groups_running.get()
                few_left = (
                    self.start_state["base"].groups_running.sum() < 3
                    and self.start_state["third"].groups_running.sum() < 3
                )
                if few_left:
                    counter_few[running] += 1
                    print(counter_few)

                check_convergence = (
                    (current_old_bf_count >= number_old_evidences)
                    & (np.asarray(self.currently_running_orig_id) != None)
                )
                end = np.full(self.ngroups, False)
                end[check_convergence] = (
                    np.all(
                        np.abs(
                            old_bf[check_convergence]
                            - np.mean(old_bf[check_convergence], axis=-1)[:, None]
                        ) < 0.03,
                        axis=-1,
                    )
                    & (
                        np.abs(np.sign(np.diff(old_bf[check_convergence], axis=-1)).sum(axis=-1))
                        < number_old_evidences - 1
                    )
                )

                timed_out = (
                    (counter_few > max_few_iterations)
                    & running
                    & (np.asarray(self.currently_running_orig_id) != None)
                )
                end = end | timed_out

                print(end, current_evidence_diff)
                print(self.currently_running_orig_id)
                if np.any(end) > 0:
                    running_inner = False
                # print(iters_at_max, start_state.groups_running.sum().item(), now_max_log_like[:10])
            
            if started_run:
                # which groups ended
                end = np.where(end)[0]
                for end_i in end:
                    
                    self.readout(end_i, indicator, current_evidence_estimate, timed_out, counter_few)

                    # reset everything for the one that has finished
                    old_bf[end_i] = np.nan
                    current_old_bf_ind[end_i] = 0
                    current_old_bf_count[end_i] = 0

                    counter_few[end_i] = 0  # added here

                    for template in ["base", "third"]:
                        current_evidence_estimate[template][end_i] = np.nan
                        evidence_log_like[template][end_i] = np.nan
                        self.start_state[template].groups_running[end_i] = False
                    
                    current_evidence_diff[end_i] = np.nan
        
                    self.currently_running_orig_id[end_i] = None
                    self.xp.get_default_memory_pool().free_all_blocks()
                    # test_burn = True   # added here 

            if self.xp.all(~self.start_state["third"].groups_running) and finish_up:
                run = False

    def form_special_file_indicator(self, orig_id: int, template: str="", **kwargs):
        return str(orig_id) + template



class PosteriorRuns(Guide):

    def __init__(self, *args, **kwargs):

        name = "pe"
        super().__init__(name, *args, **kwargs)

        # store sampler settings as attributes
        for key, val in self.sampler_settings["pe"].items():
            setattr(self, key, val)
            
        self.ndim = self.third_info.ndim
        # self.ndim = self.base_info.ndim # For Mismatched PE
        self.initialize_sampler_routine()

    def initialize_sampler_routine(self):

        self.tempering_kwargs = {"ntemps": self.ntemps, "Tmax": np.inf}

        self.prior_transform_fn = PriorTransformFn(self.xp.zeros(self.ngroups), self.xp.ones(self.ngroups), self.xp.zeros(self.ngroups), self.xp.ones(self.ngroups), P2_min=self.xp.ones(self.ngroups), P2_max=self.xp.ones(self.ngroups))

        data_channels = [self.xp.zeros(self.data_length * self.ngroups, dtype=complex), self.xp.zeros(self.data_length * self.ngroups, dtype=complex)]
        psds = [self.xp.ones(self.data_length * self.ngroups, dtype=float), self.xp.ones(self.data_length * self.ngroups, dtype=float)]
        start_freq = self.xp.full(self.ngroups, int(1e-3 / self.df), dtype=np.int32)
        N_vals_in = self.xp.zeros(self.ngroups, dtype=int)
        d_d_all = self.xp.zeros(self.ngroups, dtype=float)
        self.N_max = int(self.data_length / 4)
        self.log_like_fn = LogLikeFn(self.third_info.template_gen, data_channels, psds, start_freq, self.df, self.third_info.transform_fn, N_vals_in, self.data_length, d_d_all, **self.waveform_kwargs)
        # self.log_like_fn = LogLikeFn(self.base_info.template_gen, data_channels, psds, start_freq, self.df, self.base_info.transform_fn, N_vals_in, self.data_length, d_d_all, **self.waveform_kwargs) # For Mismatched PE
        
        self.currently_running_orig_id = [None for _ in range(self.ngroups)]
        
        # initialize sampler
        self.sampler = ParaEnsembleSampler(
            self.third_info.ndim,
            # self.base_info.ndim,
            self.nwalkers,
            self.ngroups,
            self.log_like_fn,
            self.third_info.priors,
            # self.base_info.priors,
            tempering_kwargs=self.tempering_kwargs,
            args=[],
            kwargs={},
            gpu=self.gpu,
            periodic=self.third_info.periodic,
            # periodic=self.base_info.periodic,
            backend=None,
            update_fn=None,
            update_iterations=-1,
            stopping_fn=None,
            stopping_iterations=-1,
            name="gb",
            prior_transform_fn=self.prior_transform_fn,
            provide_supplemental=True,
        )

        coords = self.xp.zeros((self.ngroups, self.ntemps, self.nwalkers, self.ndim))

        branch_supp_base_shape = (self.ngroups, self.ntemps, self.nwalkers)

        data_inds = self.xp.repeat(self.xp.arange(self.ngroups, dtype=np.int32)[:, None], self.ntemps * self.nwalkers, axis=-1).reshape(self.ngroups, self.ntemps, self.nwalkers) 
        branch_supps = {"gb": BranchSupplemental(
            {"data_inds": data_inds}, base_shape=branch_supp_base_shape, copy=True
        )}

        groups_running = self.xp.zeros(self.ngroups, dtype=bool)
        self.start_state = ParaState({"gb": coords}, groups_running=groups_running, branch_supplemental=branch_supps)
        self.start_state.log_prior = self.xp.zeros((self.ngroups, self.ntemps, self.nwalkers))
        self.start_state.log_like = self.xp.zeros((self.ngroups, self.ntemps, self.nwalkers))
        self.start_state.betas = self.xp.ones((self.ngroups, self.ntemps))
        self.currently_running_output_info = [None for _ in range(self.ngroups)]
        
    def get_last_evidence_state(self, indicator: str, orig_id: int) -> State:

        reader_evidence = HDFBackend(self.get_evidence_file(indicator), name=str(int(orig_id)) + "_third")
        # reader_evidence = HDFBackend(self.get_evidence_file(indicator), name=str(int(orig_id)) + "_base") # For Mismatched PE
        
        try:
            last_state = reader_evidence.get_last_sample()
        except KeyError:
            last_state = None

        return last_state

    def run_single_model(self, directory: str, indicator: str, fp: str):
        
        if self.verbose:
            print(f"Start {self.name}:", directory, indicator, fp)

        data_input = self.get_input_data(directory + fp)
        data_search = self.get_sampling_data(indicator)

        self.run_mcmc(indicator, data_search, data_input)

        self.add_to_status_file(indicator)

    def get_out_fp(self, indicator: str) -> str:
        return self.get_pe_file(indicator)

    def information_generator(self, indicator: str, data_search: np.ndarray, data_orig: np.ndarray):
        nbin = len(data_search)
        for i in range(nbin):
            index = int(data_search["index"][i])
            orig_id = int(data_search["orig_id"][i])

            already_run = self.check_if_source_has_been_run(indicator, str(orig_id) + "_third_posterior")
            # already_run = self.check_if_source_has_been_run(indicator, str(orig_id) + "_base_posterior") # For Mismatched PE
            if already_run:
                if self.verbose:
                    print(f"{int(orig_id)} already in file {self.get_out_fp(indicator)} so not running.")

                continue
        
            injection_params = np.array([data_search[key][i] for key in data_search.dtype.names[:14]])
            injection_params[7] = injection_params[7] % (2 * np.pi)

            N_found = self.determine_N(injection_params)
            
            if N_found > self.N_max:
                if self.verbose:
                    print(f"ID {int(orig_id)} (index: {int(index)}) has too high of N value so not running.")
                self.write_indicator_to_bad_file(indicator, reason="too high of N")
                continue

            waveform_kwargs = self.waveform_kwargs.copy()

            waveform_kwargs.pop("oversample")
            waveform_kwargs["N"] = N_found

            f_lims, fdot_lims, P2_lims = self.get_lims(injection_params, return_P2_lims=True)

            A_inj, E_inj = self.get_injection_data(injection_params)

            f0 = injection_params[1] / 1e3
            start_freq = int(int(f0 / self.df) - self.data_length / 2)
            fd = np.arange(start_freq, start_freq + self.data_length) * self.df

            data_channels = [A_inj[start_freq:start_freq + self.data_length].copy(), E_inj[start_freq:start_freq + self.data_length].copy()]

            AE_psd = get_sensitivity(fd, sens_fn="noisepsd_AE", model="SciRDv1", includewd=self.waveform_kwargs["T"] / YEAR)
            psd = [AE_psd, AE_psd]

            last_state = self.get_last_evidence_state(indicator, orig_id)

            if last_state is None:
                if self.verbose:
                    print(f"{int(orig_id)} not in file {self.get_search_file(indicator)} so not running.")
                continue

            info_out = {name: value for name, value in zip(data_search.dtype.names, data_search[i])}
            yield (orig_id, index, injection_params, fd, data_channels, psd, start_freq, f_lims, fdot_lims, P2_lims, N_found, last_state, info_out)
        
    def setup_next_source(self, info_iterator: Callable, indicator: str, data_search: np.ndarray, data_orig: np.ndarray):
        try:
            (orig_id, index, injection_params, fd, data_channels_tmp, psd_tmp, start_freq_ind, f_lims, fdot_lims, P2_lims, N_val, last_state, keep_info) = next(info_iterator)
            
            data_channels_tmp = [self.xp.asarray(tmp) for tmp in data_channels_tmp]
            psd_tmp = [self.xp.asarray(tmp) for tmp in psd_tmp]

            # get injection inner product 
            d_d = 4.0 * self.df * self.xp.sum(self.xp.asarray(data_channels_tmp).conj() * self.xp.asarray(data_channels_tmp) / self.xp.asarray(psd_tmp)).item().real

            self.third_info.template_gen.d_d = d_d
            # self.base_info.template_gen.d_d = d_d # For Mismatched PE
            start_points = last_state.branches["gb"].coords.copy()
            # setup in ParaState

            # get first group not running
            new_group_ind = self.start_state.groups_running.argmin().item()
            self.start_state.groups_running[new_group_ind] = True
            
            last_evidence_sample_nwalkers = last_state.branches["gb"].shape[1]

            inds_start = np.random.choice(np.arange(last_evidence_sample_nwalkers), size=(self.ntemps, self.nwalkers), replace=True)
            self.start_state.branches["gb"].coords[new_group_ind] = self.xp.asarray(last_state.branches["gb"].coords[0, inds_start, 0][None, :])

            inds_slice = slice((new_group_ind) * self.data_length, (new_group_ind + 1) * self.data_length, 1)
            self.sampler.log_like_fn.data[0][inds_slice] = data_channels_tmp[0]
            self.sampler.log_like_fn.data[1][inds_slice] = data_channels_tmp[1]
            self.sampler.log_like_fn.psd[0][inds_slice] = psd_tmp[0]
            self.sampler.log_like_fn.psd[1][inds_slice] = psd_tmp[1]
            self.sampler.log_like_fn.start_freq[new_group_ind] = start_freq_ind

            self.sampler.prior_transform_fn.f_min[new_group_ind] = f_lims[0]
            self.sampler.prior_transform_fn.f_max[new_group_ind] = f_lims[1]
            self.sampler.prior_transform_fn.fdot_min[new_group_ind] = fdot_lims[0]
            self.sampler.prior_transform_fn.fdot_max[new_group_ind] = fdot_lims[1]
            self.sampler.prior_transform_fn.P2_min[new_group_ind] = P2_lims[0]
            self.sampler.prior_transform_fn.P2_max[new_group_ind] = P2_lims[1]
            
            self.sampler.log_like_fn.N_vals[new_group_ind] = N_val
            self.currently_running_orig_id[new_group_ind] = orig_id

            self.sampler.log_like_fn.d_d_all[new_group_ind] = d_d

            self.currently_running_output_info[new_group_ind] = keep_info
            
            return False

        except StopIteration:
            return True

    def readout(self, end_i: int, indicator: str):

        orig_id = self.currently_running_orig_id[end_i]
                 
        if orig_id is None:
            return

        output_state = State({"gb": self.start_state.branches["gb"].coords[end_i].get()}, log_like=self.start_state.log_like[end_i].get(), log_prior=self.start_state.log_prior[end_i].get(), betas=self.start_state.betas[end_i].get(), random_state=np.random.get_state())

        group_name = str(int(orig_id)) + "_" + self.third_info.name + "_posterior"
        # group_name = str(int(orig_id)) + "_" + self.base_info.name + "_posterior" # For Mismatched PE
        backend_tmp = HDFBackend(self.get_out_fp(indicator), name=group_name)
        backend_tmp.reset(
            self.nwalkers,
            self.third_info.ndim,
            # self.base_info.ndim,
            ntemps=self.ntemps,
            branch_names=["gb"],
        )
        backend_tmp.grow(self.sampler.backend.iteration, None)
        with h5py.File(backend_tmp.filename, "a") as fp_save:
            fp_save[group_name].attrs["iteration"] = self.sampler.iteration
            fp_save[group_name]["chain"]["gb"][:] = self.sampler.get_chain()[:, end_i][:, :, :, None]
            fp_save[group_name]["log_like"][:] = self.sampler.get_log_like()[:, end_i]
            fp_save[group_name]["log_prior"][:] = self.sampler.get_log_prior()[:, end_i]
            group_new = fp_save[group_name].create_group("keep_info")
            for key, value in self.currently_running_output_info[end_i].items():
                group_new.attrs[key] = value
            
        return
       
    def run_mcmc(self, indicator: str, data_search: np.ndarray, data_input: np.ndarray):

        info_iterator = self.information_generator(indicator, data_search, data_input)

        run = True
        finish_up = False

        burn = self.sampler_settings["pe"]["burn"]
        thin_by = self.sampler_settings["pe"]["thin_by"]
        nsteps = self.sampler_settings["pe"]["nsteps"]
        progress = self.sampler_settings["pe"]["progress"]

        while run:
            finish_up = self.setup_next_source(info_iterator, indicator, data_search, data_input)

            if not finish_up and np.any(~self.start_state.groups_running):
                continue
            # end if all are done
            if finish_up and np.all(~self.start_state.groups_running):
                run = False
                return
                
            self.start_state.log_like = None
            self.start_state.log_prior = None
            self.start_state.betas = None

            self.sampler.backend.reset(*self.sampler.backend.reset_args, **self.sampler.backend.reset_kwargs)
            self.start_state = self.sampler.run_mcmc(self.start_state, nsteps, burn=burn, thin_by=thin_by, progress=progress, store=True)

            template = self.third_info
            # template = self.base_info

            for end_i in range(self.ngroups):
                orig_id = self.currently_running_orig_id[end_i]
                if orig_id is None:
                    continue
                group_name = str(int(orig_id)) + "_" + template.name + "_posterior"
                backend_tmp = HDFBackend(self.get_out_fp(indicator), name=group_name)
                backend_tmp.reset(
                    self.nwalkers,
                    template.ndim,
                    ntemps=self.ntemps,
                    branch_names=["gb"],
                )
                backend_tmp.grow(self.sampler.backend.iteration, None)
                with h5py.File(backend_tmp.filename, "a") as fp_save:
                    fp_save[group_name].attrs["iteration"] = self.sampler.iteration
                    fp_save[group_name]["chain"]["gb"][:] = self.sampler.get_chain()[:, end_i][:, :, :, None]
                    fp_save[group_name]["log_like"][:] = self.sampler.get_log_like()[:, end_i]
                    fp_save[group_name]["log_prior"][:] = self.sampler.get_log_prior()[:, end_i]
                    group_new = fp_save[group_name].create_group("keep_info")
                    for key, value in self.currently_running_output_info[end_i].items():
                        group_new.attrs[key] = value
                    

                self.currently_running_output_info[end_i] = None
                    
                self.start_state.groups_running[end_i] = False
                
                self.currently_running_orig_id[end_i] = None
                
                if self.use_gpu:
                    self.xp.get_default_memory_pool().free_all_blocks()

            if self.xp.all(~self.start_state.groups_running) and finish_up:
                run = False

    def form_special_file_indicator(self, orig_id: int, **kwargs):
        return str(orig_id)


if __name__ == "__main__":

    settings = get_settings()
    gpu = 0
    # runner = TriplesSetupGuide(settings, batch_size=1000, gpu=gpu)
    
    # runner = SearchRuns(settings, gpu=gpu)
    
    # runner = EvidenceRuns(settings, gpu=gpu) 
    
    runner = PosteriorRuns(settings, gpu=gpu) 
    runner.run_directory_list()
    breakpoint()