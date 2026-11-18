import numpy as np
import pandas as pd
import h5py
import os
from gbmodels import ThirdBodyTemplateSetup
from settings import get_settings

settings = get_settings()

gb = ThirdBodyTemplateSetup()

key_list = [
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
    'T2'
]

seed_map = {2: "s442", 3: "s352", 4: "s444", 5: "s555", 6: "s633"}
base_string = "FR_3_YEAR_1"
letter = "B"
num = 2
# realization = [1,2,3]

for realization in [2, 3, 4, 5, 6]:
    for letter in "ABCD":
        for num in [1, 2]:
            search_dir = f"/expanse/lustre/projects/umn131/vgupta1/GBGPU/population_paper/final_run_3/one_year/search_info"
            file_path = f"{search_dir}/{base_string}_search_info_{letter}{num}_{seed_map[realization]}.h5"
            if not os.path.exists(file_path):
                print(f"Cannot find {file_path}.")
                continue

            with h5py.File(file_path, "r") as fp:
                for i, tmp in enumerate(list(fp)):
                    attrs_dict = fp[tmp]["keep_info"].attrs
                    if i == 0:
                        out = {key: [val] for key, val in attrs_dict.items()}
                    else:
                        for key, value in attrs_dict.items():
                            assert key in out
                            out[key].append(value)
                    
                    if (i + 1) % 500 == 0:
                        print(f"Real{realization}-{letter}{num}: {i + 1} of {len(list(fp))}")

            out = {key: np.asarray(val) for key, val in out.items()}  

            inj = np.asarray([out[key]for key in key_list]).T

            inj_good = gb.transform_fn["gb"].transform_base_parameters(inj)

            for i, key in enumerate(key_list):
                out[key] = inj_good[:, i]

            df = pd.DataFrame(out)

            df.index = df.pop("orig_id").astype(int)
            real_dir = "/expanse/lustre/projects/umn131/vgupta1/GBGPU/population_paper/Realizations_all/"
            path_to_input_file = f"{real_dir}Realization_{realization}/data_{letter}{num}_{seed_map[realization]}.dat"
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
            data_ids = data_new["id"].astype(int)

            index_order = -np.ones_like(df.index)
            for i, index in enumerate(df.index):
                ind = np.where(data_ids == index)[0][0]
                index_order[i] = ind

            assert np.all(index_order >= 0)

            keep = pd.DataFrame(data_new[index_order])
            keep.index = keep.pop("id").astype(int)
            assert np.all(keep.index == df.index)
            df_out = pd.concat([df, keep], axis=1)
            df_out.to_hdf(f"FINAL_RUN_3/{base_string}_search_info_output_5_realizations_translated.h5", f"Real{realization}-{letter}{num}", complevel=9)



