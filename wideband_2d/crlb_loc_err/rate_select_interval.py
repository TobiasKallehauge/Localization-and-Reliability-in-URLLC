# -*- coding: utf-8 -*-
"""
Created on Mon May  1 11:27:12 2023

@author: Tobias Kallehauge
"""

import sys
sys.path.insert(0, '../../library')
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
N_workers = 4

# load data
with open(f'data/mod_idx_{data_index}.pickle','rb') as f:
    mod = pickle.load(f)
    
peb = np.sqrt(np.trace(mod.cov_loc_cal, axis1=1, axis2=2))

# =============================================================================
#%% Select rate with backoff approach
# =============================================================================

def get_R_interval_core(q, idx_eval, worker_idx = 0):
    """
    Get the rate selected for each estimated location for the true location
    idx_eval for given value of q. 
    """
    
    N_core = len(idx_eval)

    # tqdm_text = "#" + "{}".format(worker_idx).zfill(3)
    # progress_bar = tqdm(total=N_core, desc=tqdm_text,
    #                     position=worker_idx, ascii=True,
    #                     mininterval=1)

    R = np.zeros((N_core,mod.N_sim))
    X = mod.x_cal[:,:2].reshape(mod.N_sim, 2,1)
    
    # with progress_bar as pbar:
    for i, idx in enumerate(idx_eval):
        cov_inv = np.linalg.inv(mod.cov_loc_eval[idx])
        
        disp0 = cov_inv @ X
        disp1  = (np.transpose(X, axes = (0,2,1)) @ disp0).flatten()
        disp2 = 2*disp0[:,:,0] 
    
        for j, x in enumerate(mod.x_cal[:,:2]):
    
            
            # finalize dispersion calculations
            disp3 = disp2 @ x
            
            # calculate dispersion for confidence interval (ellipse)
            disp = disp1 + disp1[j] - disp3
    
            
            I_idx = disp < q
            R[i,j]  = mod.R_eps[I_idx].min()
            # pbar.update(1)
    return(R)

def get_R_interval(q, multiprocessing, N_workers = 4):
    """
    Get the rate selected for each estimated location for each the true 
    location for a given value of q. Hence it is a matrix where each row
    corrosponds to a true location and each column corroesponds to an estimated
    location. 
    
    Parameters
    ----------
    q : float
        quantile parameter
    multiprocessing : bool
        If True, multiprocessing is used. The default is False
    N_workers : int, optional
        If multiprocessing. The number of workers used. 
    
    Returns
    -------
    R : np.ndarray
        Size mod.N_eval x mod.N_cal matrix with selected rates where each row
        corrosponds to a true location and each column corroesponds to an 
        estimated location. 
    """
    
    if not multiprocessing: 
        idx_eval = np.arange(mod.N_eval)
        R = get_R_interval_core(q, idx_eval, worker_idx = 0)
    
    else:
        pool = mp.Pool(processes=N_workers,
                       initargs=(mp.RLock(),),
                       initializer=tqdm.set_lock)

        # setup arguments
        idx_split = np.array_split(np.arange(mod.N_eval), N_workers)
        

        # run multiprocessing
        jobs = [pool.apply_async(get_R_interval_core,
                                 args=(q, idx_split[i]),
                                 kwds={'worker_idx': i})
                for i in range(N_workers)]
        pool.close()
        pool.join()

        # stack results
        R = np.vstack([job.get() for job in jobs])
        
    return(R)


def f_bisect_interval(q):
    print(f'{q:.2f}, diff : ', end = '')
    R = get_R_interval(q,multiprocessing, N_workers)
    p_max = mod.get_meta_probability(R, matrix = True).max()
    diff = p_max - delta
    print(f'{diff:.2f}')
    return(diff)
    

if __name__ == '__main__':
    q = bisect(f_bisect_interval, 0.1, 500, xtol = 0.1)
    
    print(f'q : {q}')


    R = get_R_interval(q, multiprocessing, N_workers = 4)
    p =  mod.get_meta_probability(R, matrix = True)

    # get throughput at selected rate
    T = mod.get_throughput(R, matrix = True)


    
    res = {'R': R, 'p': p, 'T': T, 
            'q': q, 'delta': delta}

    with open(f'data/interval_{data_index}.pickle', 'wb') as f:  # Overwrites any existing file.
        pickle.dump(res, f, pickle.HIGHEST_PROTOCOL)


