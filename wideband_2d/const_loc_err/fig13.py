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
peb_rng = [5,10,15,20,30,40,50,100]
whis = [5,95]

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
q_all = [res['q'] for res in res_interval]

p_backoff_all = np.vstack(res['p'] for res in res_backoff)
p_interval_all = np.vstack(res['p'] for res in res_interval)
 

T_backoff_all = np.vstack(res['T'] for res in res_backoff)
T_interval_all = np.vstack(res['T'] for res in res_interval)
 
# =============================================================================
#%% plot meta-probability vs peb        
# =============================================================================

fig, ax = plt.subplots(ncols = 2)
fig.subplots_adjust(wspace = 0.05)
ax[0].boxplot(p_backoff_all.T, showfliers = False, labels = peb_rng, whis = whis)
ax[0].set_ylabel(r'Meta-probability $\tilde{p}_{\epsilon}$')
ax[0].set_yscale('symlog',linthresh = 1e-15)
ax[0].axhline(delta, c = 'r')
ax[0].set_ylim(ax[0].get_ylim()[0],0.1)
ax[0].set_title('Backoff')

ax[1].boxplot(p_interval_all .T, showfliers = False, labels = peb_rng, whis =  whis)
ax[1].set_yscale('symlog',linthresh = 1e-15)
ax[1].axhline(delta, label = r'$\delta$', c = 'r')
ax[1].legend(loc = 'upper right')
ax[1].set_ylim(ax[1].get_ylim()[0],0.1)
ax[1].set_title('Interval')
ax[1].set_yticks([])

fig.supxlabel('Position error bound (PEB) [m]')
fig.savefig('plots/fig13a.pdf', bbox_inches = 'tight')





# =============================================================================
#%% plot throughtput vs peb        
# =============================================================================

fig, ax = plt.subplots(ncols = 2)
fig.subplots_adjust(wspace = 0.05)
ax[0].set_title('Backoff')
ax[0].boxplot(T_backoff_all.T, showfliers = False, labels = peb_rng, whis = whis)
ax[0].plot(range(1,len(peb_rng)+1),k_all, 'o', label = r'$\beta$')
ax[0].set_ylabel(r'Throughput ratio $\overline{T}$')
ax[0].legend()

ax[1].boxplot(T_interval_all.T, showfliers = False, labels = peb_rng, whis= whis)
ax[1].yaxis.tick_right()
ax[1].set_title('Interval')

yticks = np.hstack([np.arange(0.1,1 + 0.1, step = 0.1),
                    np.arange(0.01,0.1 + 0.01, step = 0.01),
                    np.arange(0.001,0.01 + 0.001, step = 0.001)])

ax[1].set_ylim(1e-3,0.7)
ax[0].set_ylim(1e-3,0.7)
ax[0].set_yscale('log')
ax[1].set_yscale('log')
ax[0].set_yticks(yticks)
ax[1].set_yticks(yticks)
ax[1].yaxis.set_visible(False)

fig.supxlabel('Position error bound (PEB) [m]')
fig.savefig('plots/fig13b.pdf', bbox_inches = 'tight')
