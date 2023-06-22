# -*- coding: utf-8 -*-
"""
Created on Tue May 23 09:46:59 2023

@author: Tobias Kallehauge
"""

import matplotlib.pyplot as plt
import pickle
import numpy as np
import sys
sys.path.insert(0, '../library')
import data_generator_quadriga


data_index = 6
delta = 0.05
log = False
peb_rng = [5,10,15,20,30,40,50,100]
whis = [5,95]
p = 0.9

# load data
with open(f'data/mod_idx_{data_index}_const_loc.pickle', 'rb') as f:
    mod = pickle.load(f)


res_interval = []
res_backoff = []

for peb in peb_rng:
    with open(f'data/backoff_{data_index}_peb_{peb}.pickle', 'rb') as f:
        res_backoff.append(pickle.load(f))
    with open(f'data/interval_{data_index}_peb_{peb}.pickle', 'rb') as f:
        res_interval.append(pickle.load(f))        
        
# stack data
k_all = [res['k'] for res in res_backoff]

p_backoff_all = np.vstack(res['p'] for res in res_backoff)
p_interval_all = np.vstack(res['p'] for res in res_interval)
 

T_backoff_all = np.vstack(res['T'] for res in res_backoff)
T_interval_all = np.vstack(res['T'] for res in res_interval)
 
# =============================================================================
#%% Associate points with either peak or valey
# =============================================================================


# load peaks and valleys from file
idx_peaks = np.load(f'data/peaks_manual_idx_{data_index}.npy')
idx_valeys = np.load(f'data/valleys_manual_idx_{data_index}.npy')

peaks = mod.x_cal[idx_peaks]
valeys = mod.x_cal[idx_valeys]

points = np.vstack((peaks,valeys))
peaks_idx = np.hstack((np.ones(len(peaks)),np.zeros(len(valeys))))

peaks_idx_eval = np.where(idx_peaks[mod.idx_eval])[0]
peaks_eval = mod.x_eval[peaks_idx_eval]

valeys_idx_eval = np.where(idx_valeys[mod.idx_eval])[0]
valeys_eval = mod.x_eval[valeys_idx_eval]

# =============================================================================
#%% Get coherence distance for peaks and valleys
# =============================================================================
steps = np.linspace(1,100, num = 200)



def get_coherence(peaks_idx_eval, p, log = False):
    N = len(peaks_idx_eval)
    if log: 
        R = np.log(mod.R_eps)
    else:
        R = mod.R_eps
        
    
    D = np.zeros(N) # coherence distances
    
    for i, idx in enumerate(peaks_idx_eval):
        
        
        for d in steps:
            x = mod.x_eval[idx]
            
            dist = np.linalg.norm(x - mod.x_cal, axis = 1)
            I_idx = dist < d
        
        
            R_x = R[mod.idx_eval[idx]]
            delta = np.abs(R_x  - R[I_idx]).max()/np.abs(R_x)
            # delta = np.abs(R_x  - R[I_idx].min())/np.abs(R_x)
            
            if delta > p:
                break
        D[i] = d
        
    return(D)



D_peaks = get_coherence(peaks_idx_eval, p, log = log)
D_valleys = get_coherence(valeys_idx_eval,p, log = log)

# =============================================================================
#%% Plot just selected data correlations
# =============================================================================


# throughput ratio

# interval peaks
res = res_interval
D = D_peaks
idx = peaks_idx_eval
rho_T = []
fig, ax = plt.subplots(figsize = (6,3))
for k, peb in enumerate(peb_rng):
    
    rho_T.append(np.corrcoef(D,res[k]['T'][idx])[0,1])

    ax.scatter(D,res[k]['T'][idx], s = 5)
    
    # fit lines and plot
    a_T, b_T = np.polyfit(D, res[k]['T'][idx], 1)
    rng = np.linspace(D.min(), D.max() , 100)
    ax.plot(rng, a_T*rng + b_T, label = f'{peb} m ')
ax.legend(title = 'PEB', fontsize = 8)
ax.set_xlabel('Coherence distance [m]')
ax.set_ylabel(r'Throughput ratio $\overline{T}$')
fig.savefig('plots/fig14.pdf', bbox_inches = 'tight')


