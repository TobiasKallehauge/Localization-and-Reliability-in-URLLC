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
peb = 5 # constant position error bound

# load data
with open(f'../data/mod_idx_{data_index}.pickle','rb') as f:
    mod = pickle.load(f)

# overwrite localization variance or model
Cov = np.diag([peb**2/2,peb**2/2])
mod.cov_loc_cal = np.repeat(Cov,mod.N_sim).\
                    reshape(mod.N_sim,2,2, order = 'F')
mod.cov_loc_eval = np.repeat(Cov,mod.N_eval).\
                    reshape(mod.N_eval,2,2, order = 'F')
mod.PEB = np.sqrt(np.trace(mod.cov_loc_cal, axis1=1, axis2=2))

# save new model
mod.save(f'data/mod_idx_{data_index}_const_loc.pickle')


# =============================================================================
#%% Plot data
# =============================================================================


import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable

# ### Channel powermap ###
fig, ax = plt.subplots(figsize = (4,4))
divider = make_axes_locatable(ax)
im = ax.imshow(mod.R_eps[mod.idx_eval].reshape(mod.N_side_eval,mod.N_side_eval),
          origin = 'lower',
          norm = LogNorm(),
          cmap = 'jet',
          extent = mod.extent_eval)
ax.plot([mod.x_BS[mod.BS_data,0]],mod.x_BS[mod.BS_data,1], markersize = 15,
        marker = 'X', markerfacecolor='w', 
        markeredgecolor='k', markeredgewidth=2.0, linewidth = 0,
        label = 'BS')
cax = divider.append_axes("right", size="7%", pad=0.09)
cbar = fig.colorbar(im, cax=cax)
cbar.set_label('Outage capacity [bits/s/Hz]')
ax.legend()
ax.set_ylabel('UE location 2nd coordinate [m]')
ax.set_xlabel('UE location 1st coordinate [m]')
ax.set_title('Channel powermap')
fig.savefig('plots/R_eps_eval.pdf', bbox_inches = 'tight')