# -*- coding: utf-8 -*-
"""
Created on Tue May 23 09:46:59 2023

@author: Tobias Kallehauge
"""

from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import pickle
import numpy as np
import sys
sys.path.insert(0, '../library')
import data_generator_quadriga
import matplotlib.patheffects as PathEffects

data_index = 6
delta = 0.05
log = False
p = 0.9
peak_method = 'manual'

# load data
with open(f'data/mod_idx_{data_index}_const_loc.pickle', 'rb') as f:
    mod = pickle.load(f)

with open(f'data/backoff_{data_index}_const_loc.pickle', 'rb') as f:
    res_backoff = pickle.load(f)

with open(f'data/interval_{data_index}_const_loc.pickle', 'rb') as f:
    res_interval = pickle.load(f)


# load peaks and valleys from file
idx_peaks = np.load(f'data/peaks_manual_idx_{data_index}.npy')
idx_valeys = np.load(f'data/valleys_manual_idx_{data_index}.npy')


# =============================================================================
#%% Associate points with either peak or valey
# =============================================================================

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



def get_coherence(peaks_idx_eval, p, log = True):
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
#%% plot correlations at found p
# =============================================================================

rho_p = []
rho_T = []
names = ['Backoff', 'Interval']
rotations = [[[13,0],[6,30]],
             [[-26,-4],[0,-5]]]
text_h = [[[20,0.015],[20,0.02]],
          [[-55,0],[-20,-0.15]]]

fig, ax = plt.subplots(nrows = 2, ncols= 2)
fig.subplots_adjust(hspace = 0.05, wspace = 0.05)
for i, (D, idx) in enumerate(((D_peaks, peaks_idx_eval), (D_valleys,valeys_idx_eval))):
    for j, res in enumerate((res_backoff, res_interval)): 
        idx_nonzero = res['p'][idx] != 0
        p_nonzero = res['p'][idx][idx_nonzero]
        
        
        rho_p_ij = np.corrcoef(D[idx_nonzero],np.log(p_nonzero))[0,1]
        rho_T_ij = np.corrcoef(D,res['T'][idx])[0,1]
        rho_p.append(rho_p_ij)
        rho_T.append(rho_T_ij)
    
        ax[0,i].scatter(D,res['p'][idx], c = f'C{j}', s = 5, label = names[j])
        ax[1,i].scatter(D,res['T'][idx], c = f'C{j}', s = 5)
        
        # fit lines and plot
        a_p, b_p = np.polyfit(D[idx_nonzero], np.log(p_nonzero), 1)
        a_T, b_T = np.polyfit(D, res['T'][idx], 1)
        rng = np.linspace(D.min(), D.max() , 3)
        ax[0,i].plot(rng, np.exp(a_p*rng + b_p),  c = f'C{j}')
        ax[1,i].plot(rng, a_T*rng + b_T,  c = f'C{j}')
        
        # add text for lines
        
        # for rho
        label = r'$\rho$' + f' : {rho_p_ij:.2f}'
        midpoint = D[1]
        txt = ax[0,i].text(midpoint, np.exp(midpoint*a_p + b_p + text_h[i][j][0]), label,
                     rotation = rotations[i][j][0],
                     c = 'w',
                     fontsize = 10)
        txt.set_path_effects([PathEffects.withStroke(linewidth=1, foreground='k')])
        
        # for tau
        label = r'$\rho$' + f' : {rho_T_ij:.2f}'
        midpoint = D[1]
        txt = ax[1,i].text(midpoint, midpoint*a_T + b_T + text_h[i][j][1], label,
                     rotation = rotations[i][j][1],
                     c = 'w',
                     fontsize = 10)
        txt.set_path_effects([PathEffects.withStroke(linewidth=1, foreground='k')])
                
        
    
    ax[i,1].yaxis.tick_right()


ax[0,1].legend(loc = 'lower left')
ax[0,0].set_title('Peaks')
ax[0,1].set_title('Valleys')
ax[0,0].set_ylabel(r'Meta probability $\tilde{p}_{\epsilon}$')
ax[1,0].set_ylabel(r'Throughput ratio $\overline{T}$')
ax[0,0].set_xticks([])
ax[0,1].set_xticks([])
ax[0,0].set_yscale('log')
ax[0,1].set_yscale('log')    
ax[1,0].set_xlabel('Coherence distance [m]')
ax[1,1].set_xlabel('Coherence distance [m]')

fig.savefig('plots/fig12.pdf', bbox_inches = 'tight')

print(
f"""Correlations

Meta probability

Peak backoff: {rho_p[0]:.2f}
Peak interval: {rho_p[1]:.2f}
Valley backoff: {rho_p[2]:.2f}
Valley interval: {rho_p[3]:.2f}

Througput ratio
Peak backoff: {rho_T[0]:.2f}
Peak interval: {rho_T[1]:.2f}
Valley backoff: {rho_T[2]:.2f}
Valley interval: {rho_T[3]:.2f}
""")


