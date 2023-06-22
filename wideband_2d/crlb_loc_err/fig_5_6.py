# -*- coding: utf-8 -*-
"""
Created on Fri May 12 11:00:52 2023

@author: Tobias Kallehauge
"""

import sys
sys.path.insert(0, '../library')
from data_generator_quadriga import data_generator_quadriga
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pickle

data_index = 6
with open(f'data/mod_idx_{data_index}.pickle','rb') as f:
    mod = pickle.load(f)

# =============================================================================
#%% Outage capacity
# =============================================================================
fig, ax = plt.subplots(figsize = (4,4))
divider = make_axes_locatable(ax)
im = ax.imshow(mod.R_eps[mod.idx_eval].reshape(mod.N_side_eval,mod.N_side_eval),
          origin = 'lower',
          norm = LogNorm(),
          cmap = 'jet',
          extent = mod.extent_eval)
ax.plot([mod.x_BS[mod.BS_data,0]],mod.x_BS[mod.BS_data,1], markersize = 10,
        marker = 'X', markerfacecolor='w', 
        markeredgecolor='k', markeredgewidth=1.0, linewidth = 0,
        label = 'BS')
cax = divider.append_axes("right", size="7%", pad=0.09)
cbar = fig.colorbar(im, cax=cax)
cbar.set_label(r'$C_{\epsilon}$ [bits/s/Hz]')
ax.legend()
ax.set_ylabel('UE 2nd coordinate [m]')
ax.set_xlabel('UE 1st coordinate [m]')
ax.set_xlim(-55,55)
ax.set_ylim(-55,55)
ax.set_title(r'$\epsilon$-outage capacity ')
fig.savefig('plots/fig5.pdf', bbox_inches = 'tight')

# =============================================================================
#%% Localization statistics 
# =============================================================================

### PEB map ###

fig, ax = plt.subplots(figsize = (4,4))
divider = make_axes_locatable(ax)
im = ax.imshow(mod.PEB.reshape(mod.N_side_eval,mod.N_side_eval),
          origin = 'lower',
           norm = Normalize(vmax  = 7),
          cmap = 'jet',
          extent = mod.extent_eval)
labels = ['BS',None,None,None]
for i in range(4):
    ax.plot([mod.x_BS[i,0]],mod.x_BS[i,1], markersize = 10,
            marker = 'X', markerfacecolor='w', 
            markeredgecolor='k', markeredgewidth=1.0, linewidth = 0,
            label = labels[i])
cax = divider.append_axes("right", size="7%", pad=0.09)
cbar = fig.colorbar(im, cax=cax, extend = 'max')
cbar.set_label('PEB [m]')
ax.legend(bbox_to_anchor=(1.0,.92))
ax.set_ylabel('UE 2nd coordinate [m]')
ax.set_xlabel('UE 1st coordinate [m]')
ax.set_title('Position Error Bound')
ax.set_xlim(-55,55)
ax.set_ylim(-55,55)
fig.savefig('plots/fig6.pdf', bbox_inches = 'tight')

