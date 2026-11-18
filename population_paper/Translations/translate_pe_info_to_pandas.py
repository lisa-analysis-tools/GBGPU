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

seed_map = {1: "s433", 2: "s442", 3: "s352", 4: "s444", 5: "s555", 6: "s633"}
base_string = settings["dir_info"]["base_string"]
letter = "B"
num = 2
realization = 1
