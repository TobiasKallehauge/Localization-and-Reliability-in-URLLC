# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 08:31:25 2023

@author: Tobias Kallehauge
"""

import sys
sys.path.insert(0,'../library')
import data_generator_quadriga
import pickle
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm


data_index = 6
delta = 0.05

# load data
with open(f'data/mod_idx_{data_index}_const_loc.pickle','rb') as f:
    mod = pickle.load(f)

with open(f'data/backoff_{data_index}_const_loc.pickle','rb') as f:
    res_backoff = pickle.load(f)
    
with open(f'data/interval_{data_index}_const_loc.pickle','rb') as f:
    res_interval = pickle.load(f)

        
norm = SymLogNorm(vmin = 0, vmax = delta, linthresh=1e-15)
        
fig, ax = plt.subplots(ncols = 2, figsize = (8,4))
fig.subplots_adjust(wspace = .0)


# plot backoff
ax[0].set_title(r'Backoff $\beta$' + f': {res_backoff["k"]:.3f}')
im = ax[0].imshow(res_backoff['p'].reshape(mod.N_side_eval,mod.N_side_eval), 
          extent = mod.extent_eval, norm = norm,
          cmap = 'jet',
          origin = 'lower')
ax[0].set_xlabel(r'UE location 1st coordinate [m]')
ax[0].set_ylabel(r'UE location 2nd coordinate [m]')

# plot interval
ax[1].set_title(r'Interval, $q^2$' + f' : {res_interval["q"]:.2f}')
im = ax[1].imshow(res_interval['p'].reshape(mod.N_side_eval,mod.N_side_eval), 
          extent = mod.extent_eval, norm = norm,
          cmap = 'jet',
          origin = 'lower')
ax[1].set_xlabel(r'Estimated location 1st coordinate [m]')
ax[1].set_yticks([])

cax = fig.add_axes([ax[1].get_position().x1+0.01,ax[1].get_position().y0,0.02,ax[1].get_position().height])
cbar = fig.colorbar(im, cax = cax)
cbar.set_label(r'$\tilde{p}_{\epsilon}$')
fig.savefig('plots/fig8.pdf', bbox_inches = 'tight')