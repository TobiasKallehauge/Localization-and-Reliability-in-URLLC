# -*- coding: utf-8 -*-
"""
Created on Mon May  1 11:27:12 2023

@author: Tobias Kallehauge
"""

import sys
sys.path.insert(0, '../library')
import data_generator_quadriga
import numpy as np
import pickle
from scipy.optimize import bisect

# settings (see print(mod) for other settings)
data_index = 6
delta = 5e-2
multiprocessing = True
N_workers = 40

# load data
with open(f'data/mod_idx_{data_index}.pickle','rb') as f:
    mod = pickle.load(f)
    
peb = np.sqrt(np.trace(mod.cov_loc_cal, axis1=1, axis2=2))

# =============================================================================
#%% Select rate with backoff approach
# =============================================================================


def get_R_interval(q):
    
    R = np.zeros(mod.N_sim)
    
    for i, x in enumerate(mod.x_cal):
        d = np.linalg.norm(mod.x_cal[i] - mod.x_cal, axis = 1)
        
        I_idx = d < q
        R[i]  = mod.R_eps[I_idx].min()
    return(R)

def f_bisect_interval(q):
    R = get_R_interval(q)
    p_max = mod.get_meta_probability(R).max()
    diff = p_max - delta
    print(q,f'{diff:.2f}')
    return(diff)

q = bisect(f_bisect_interval, 0.1, 20)
print(f'q : {q:.2f}')


R = get_R_interval(q)
p =  mod.get_meta_probability(R)

# get throughput at selected rate
T = mod.get_throughput(R)

res = {'R': R, 'p': p, 'T': T, 
        'q': q, 'delta': delta}

with open(f'data/distance_{data_index}.pickle', 'wb') as f:  # Overwrites any existing file.
    pickle.dump(res, f, pickle.HIGHEST_PROTOCOL)



