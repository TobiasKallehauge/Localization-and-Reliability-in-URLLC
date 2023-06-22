# -*- coding: utf-8 -*-
"""
Created on Mon May 22 16:48:17 2023

@author: Tobias Kallehauge
"""

import sys
sys.path.insert(0, '../../../library')
import data_generator_quadriga
import numpy as np
import pickle

# settings (see print(mod) for other settings)
data_index = 6
delta = 5e-2
multiprocessing = True
N_workers = 40
peb_rng = [1,5,10,15,20,30,40,50,100] # constant position error bound

# load data
with open(f'../data/mod_idx_{data_index}.pickle','rb') as f:
    mod = pickle.load(f)

# overwrite localization variance or model
for peb in peb_rng:
    Cov = np.diag([peb**2/2,peb**2/2])
    mod.cov_loc_cal = np.repeat(Cov,mod.N_sim).\
                        reshape(mod.N_sim,2,2, order = 'F')
    mod.cov_loc_eval = np.repeat(Cov,mod.N_eval).\
                        reshape(mod.N_eval,2,2, order = 'F')
    mod.PEB = np.sqrt(np.trace(mod.cov_loc_cal, axis1=1, axis2=2))
    
    # save new model
    mod.save(f'data/mod_idx_{data_index}_peb_{peb}.pickle')
