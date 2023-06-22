# -*- coding: utf-8 -*-
"""
Created on Mon May  1 11:27:12 2023

@author: Tobias Kallehauge
"""

import sys
sys.path.insert(0, '../../../library')
import data_generator_quadriga
import numpy as np
import matplotlib.pyplot as plt
import pickle
from matplotlib.colors import SymLogNorm, LogNorm
from scipy.optimize import bisect
from scipy.interpolate import interp1d
from statsmodels.distributions.empirical_distribution import ECDF

# settings (see print(mod) for other settings)
data_index = 6
delta = 5e-2
peb_rng = [1,5,10,15,20,30,40,50,100]

# load data
with open(f'data/mod_idx_{data_index}_const_loc.pickle','rb') as f:
    mod = pickle.load(f)

# =============================================================================
#%% Select rate with backoff approach
# =============================================================================

# setup bisection method
def get_R_backoff(k, mod):
    R = k*mod.R_eps
    return(R)

f_bisect_backoff = lambda k, mod : mod.get_meta_probability(get_R_backoff(k, mod)).max() - delta


for peb in peb_rng:
    print(f'\nPEB : {peb}')
    with open(f'data/mod_idx_{data_index}_peb_{peb}.pickle','rb') as f:
        mod = pickle.load(f)
        
    k = bisect(f_bisect_backoff, 0, 2, args = (mod,))
    print(f'k : {k:.2f}')
    
    
    R = get_R_backoff(k, mod )
    p =  mod.get_meta_probability(R)
    
    # get throughput at selected rate
    T = mod.get_throughput(R)
    
    
    # =============================================================================
    #%% save results
    # =============================================================================
    
    res = {'R': R, 'p': p, 'T': T, 
            'k': k, 'delta': delta}
    
    with open(f'data/backoff_{data_index}_peb_{peb}.pickle', 'wb') as f:  # Overwrites any existing file.
        pickle.dump(res, f, pickle.HIGHEST_PROTOCOL)
