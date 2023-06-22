# -*- coding: utf-8 -*-
"""
Created on Mon May  1 11:21:28 2023

@author: Tobias Kallehauge
"""

import sys
sys.path.insert(0, '/library')
from data_generator_quadriga import data_generator_quadriga
import numpy as np



# settings
SNR_dB = 60
data_index = 6
eps = 1e-3
oversample = 100
multiprocessing = True
N_workers = 4
N_CDF = 5000
l_edge = 20 
N_sub = 601
W_MHz = 20
CRLB_monte_carlo_sims = 1000
cal_fading_stats = True
cal_loc_stats = True
res_path = f'data/stats_idx_{data_index}.pickle'
crlb_path = f'data/CRLB_{data_index}_SNRdB{SNR_dB}.pickle'


if __name__ == '__main__':
    np.random.seed(72) # for reproducibility
    mod = data_generator_quadriga(data_index, data_path = '../../Quadriga/Stored/',
                                  eps = eps, 
                                  multiprocessing= multiprocessing, 
                                  N_workers = N_workers,
                                  N_CDF = N_CDF,
                                  l_edge = l_edge,
                                  N_sub = N_sub,
                                  W_MHz = W_MHz,
                                  SNR_dB = SNR_dB,
                                  oversample = 100,
                                  cal_fading_stats = cal_fading_stats,
                                  cal_loc_stats = cal_loc_stats,
                                  res_path = res_path,
                                  crlb_path = crlb_path,
                                  CRLB_monte_carlo_sims = CRLB_monte_carlo_sims)
    
    # save model
    mod.save(f'data/mod_idx_{data_index}.pickle')

# =============================================================================
#%% Some visualizations
# =============================================================================
if True and __name__ == '__main__':
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm, Normalize
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    import matplotlib.patches as patches

    # ### Channel powermap ###
    fig, ax = plt.subplots(figsize = (4,4))
    divider = make_axes_locatable(ax)
    im = ax.imshow(mod.R_eps.reshape(mod.N_side_cal,mod.N_side_cal),
              origin = 'lower',
              norm = LogNorm(),
              cmap = 'jet',
              extent = mod.extent)
    ax.plot([mod.x_BS[mod.BS_data,0]],mod.x_BS[mod.BS_data,1], markersize = 15,
            marker = 'X', markerfacecolor='w', 
            markeredgecolor='k', markeredgewidth=2.0, linewidth = 0,
            label = 'BS')
    rect = patches.Rectangle((-50, -50), 100, 100, linewidth=1, edgecolor='k', facecolor='none',
                              linestyle = '--', label = 'Cell area')
    ax.add_patch(rect)
    cax = divider.append_axes("right", size="7%", pad=0.09)
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label('Outage capacity [bits/s/Hz]')
    ax.legend()
    ax.set_xticks(np.arange(-mod.extent[0],mod.extent[1] + 20, step = 20))
    ax.set_yticks(np.arange(-mod.extent[2],mod.extent[3] + 20, step = 20))
    ax.set_ylabel('UE location 2nd coordinate [m]')
    ax.set_xlabel('UE location 1st coordinate [m]')
    ax.set_title('Channel powermap')


    ### PEB map ###
    peb = np.sqrt(np.trace(mod.cov_loc_eval, axis1=1, axis2=2))

    fig, ax = plt.subplots(figsize = (4,4))
    divider = make_axes_locatable(ax)
    im = ax.imshow(peb.reshape(mod.N_side_eval,mod.N_side_eval),
              origin = 'lower',
                norm = Normalize(vmax  = 10),
              cmap = 'jet',
              extent = mod.extent_eval)
    labels = ['BS',None,None,None]
    for i in range(4):
        ax.plot([mod.x_BS[i,0]],mod.x_BS[i,1], markersize = 15,
                marker = 'X', markerfacecolor='w', 
                markeredgecolor='k', markeredgewidth=2.0, linewidth = 0,
                label = labels[i])
    cax = divider.append_axes("right", size="7%", pad=0.09)
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label('Position Error Bound [m]')
    ax.legend(loc = 'lower right')
    ax.set_ylabel('UE location 2nd coordinate [m]')
    ax.set_xlabel('UE location 1st coordinate [m]')
    ax.set_title('Localization error')
    ax.set_xlim(-60,60)
    ax.set_ylim(-60,60)

    
    print(f'Percentage PEB above 10 m: {np.mean(peb > 10)*100:.2f}%')

