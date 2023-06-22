# -*- coding: utf-8 -*-
"""
Created on Mon May  1 11:27:12 2023

@author: Tobias Kallehauge
"""

import sys
sys.path.insert(0, '../../../library')
import data_generator_quadriga
import numpy as np
import pickle
from scipy.optimize import bisect
import multiprocessing as mp
from tqdm import tqdm

# settings (see print(mod) for other settings)
data_index = 6
delta = 5e-2
multiprocessing = True
N_workers = 40
peb_rng = [1,5,10,15,20,30,40,50,100]

# load data

# =============================================================================
#%% Select rate with backoff approach
# =============================================================================

def get_R_interval(q, mod):
    
    R = np.zeros(mod.N_sim)
    cov_inv = np.linalg.inv(mod.cov_loc_cal[0]) # the same for all locations
    X = mod.x_cal[:,:2].reshape(mod.N_sim, 2,1)
    
    
    for i, x in enumerate(mod.x_cal):
        mean = x[:2].reshape(1,2,1)
       
        # calculate dispersion for confidence interval (ellipse)
        disp = (np.transpose(X - mean, axes = (0,2,1)) @ cov_inv @ (X - mean)).flatten()
        
        I_idx = disp < q
        R[i]  = mod.R_eps[I_idx].min()
    return(R)

def f_bisect_interval(q, mod):
    R = get_R_interval(q, mod)
    p_max = mod.get_meta_probability(R).max()
    diff = p_max - delta
    print(q,f'{diff:.2f}')
    return(diff)

for peb in peb_rng:
    print(f'\nPEB : {peb}')
    with open(f'data/mod_idx_{data_index}_peb_{peb}.pickle','rb') as f:
        mod = pickle.load(f)
        
    q = bisect(f_bisect_interval, 0.1, 20, xtol = 1e-2, args = (mod,))
    print(f'q : {q:.2f}')
    
    
    R = get_R_interval(q,mod)
    p =  mod.get_meta_probability(R)
    
    # get throughput at selected rate
    T = mod.get_throughput(R)
    
    res = {'R': R, 'p': p, 'T': T, 
            'q': q, 'delta': delta}
    
    with open(f'data/interval_{data_index}_peb_{peb}.pickle', 'wb') as f:  # Overwrites any existing file.
        pickle.dump(res, f, pickle.HIGHEST_PROTOCOL)




