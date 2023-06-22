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
from matplotlib.colors import Normalize


data_index = 6
delta = 0.05

# load data
with open(f'data/mod_idx_{data_index}_const_loc.pickle','rb') as f:
    mod = pickle.load(f)

with open(f'data/backoff_{data_index}_const_loc.pickle','rb') as f:
    res_backoff = pickle.load(f)
    
with open(f'data/interval_{data_index}_const_loc.pickle','rb') as f:
    res_interval = pickle.load(f)

        
        
fig, ax = plt.subplots(ncols = 2, figsize = (8,4))
fig.subplots_adjust(wspace = .3)

# plot backoff
ax[0].set_title(r'Backoff, $\beta$' + f': {res_backoff["k"]:.3f}')
im1 = ax[0].imshow(res_backoff['T'].reshape(mod.N_side_eval,mod.N_side_eval), 
          extent = mod.extent_eval,
          cmap = 'jet',
          origin = 'lower',
          norm = Normalize(vmin = 0, vmax = 0.15))
ax[0].set_xlabel(r'UE 1st coordinate [m]')
ax[0].set_ylabel(r'UE 2nd coordinate [m]')
cax = fig.add_axes([ax[0].get_position().x1+0.01,ax[0].get_position().y0,0.02,ax[0].get_position().height])
cbar = fig.colorbar(im1, cax = cax, extend = 'max')


# plot intervalneu
ax[1].set_title(r'Interval, $q^2$' + f' : {res_interval["q"]:.2f}')
im2 = ax[1].imshow(res_interval['T'].reshape(mod.N_side_eval,mod.N_side_eval), 
          extent = mod.extent_eval,
          cmap = 'jet',
          origin = 'lower',
          norm = Normalize(vmin = 0, vmax = 1))
ax[1].set_xlabel(r'UE 1st coordinate [m]')
ax[1].set_yticks([])


cax = fig.add_axes([ax[1].get_position().x1+0.01,ax[1].get_position().y0,0.02,ax[1].get_position().height])
cbar = fig.colorbar(im2, cax = cax)
cbar.set_label(r'$\overline{T}$')
fig.savefig('plots/fig9.pdf')