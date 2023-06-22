# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 08:31:25 2023

@author: Tobias Kallehauge
"""

import sys
sys.path.insert(0, '../library')
import data_generator_quadriga
import numpy as np
import pickle
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.patches import Ellipse, Rectangle
import matplotlib.patches as mpatches



# settings (see print(mod) for other settings)
data_index = 6
delta = 5e-2
loc_idx = 1626




with open(f'data/backoff_{data_index}.pickle','rb') as f:
    res_backoff = pickle.load(f)
    
with open(f'data/distance_{data_index}.pickle','rb') as f:
    res_interval_blind = pickle.load(f)
        
with open(f'data/interval_{data_index}.pickle','rb') as f:
    res_interval = pickle.load(f)

# load data
with open(f'data/mod_idx_{data_index}.pickle','rb') as f:
    mod = pickle.load(f)

C  =mod.cov_loc_eval[loc_idx]
eig_val, eig_vec = np.linalg.eig(C)
angle_rad =  np.arctan2(eig_vec[1,0],eig_vec[0,0]) # angle of rotation
angle_deg = angle_rad*180/np.pi

E = Ellipse((40,42),
            2*np.sqrt(eig_val[0]*res_interval["q"]),
            
            angle = angle_deg,
            fill = False, 
            edgecolor = 'k', 
            linestyle = '-', 
            linewidth = 1.5,
            label = 'Interval')
R = Rectangle((30, 30), 20, 20, linewidth=1, edgecolor='none', facecolor='w')

        
R_all = np.hstack([res_backoff['R'], res_interval['R'][loc_idx],res_interval_blind['R']])
norm = LogNorm(vmin = R_all.min(), vmax = 50)
        
        
fig, ax = plt.subplots(ncols = 3, figsize = (12,4))

# plot backoff
ax[0].set_title(r'Backoff, $\beta$' + f' : {res_backoff["k"]:.3f}')
im = ax[0].imshow(res_backoff['R'][mod.idx_eval].reshape(mod.N_side_eval,mod.N_side_eval), 
          extent = mod.extent_eval, norm = norm,
          cmap = 'jet',
          origin = 'lower')
ax[0].set_xlabel(r'Estimated location 1st coordinate [m]')
ax[0].set_ylabel(r'Estimated location 2nd coordinate [m]')

# plot intervalneu
ax[1].set_title(r'Interval, $q^2$' +f' : {res_interval["q"]:.2f}')
im = ax[1].imshow(res_interval['R'][loc_idx][mod.idx_eval].reshape(mod.N_side_eval,mod.N_side_eval), 
          extent = mod.extent_eval, norm = norm,
          cmap = 'jet',
          origin = 'lower')
ax[1].set_xlabel(r'Estimated location 1st coordinate [m]')
ax[1].set_yticks([])
ax[1].add_patch(R)
ax[1].add_patch(E)
ax[1].text(40, 31.8, r'$I(\hat{\mathbf{x}})$', horizontalalignment = 'center')

ax[2].set_title(r'Distance, $d^2$' + f' : {res_interval_blind["q"]**2:.0f}')
im = ax[2].imshow(res_interval_blind['R'][mod.idx_eval].reshape(mod.N_side_eval,mod.N_side_eval), 
          extent = mod.extent_eval, norm = norm,
          cmap = 'jet',
          origin = 'lower')
ax[2].set_xlabel(r'Estimated location 1st coordinate [m]')
ax[2].set_yticks([])


fig.subplots_adjust(wspace = 0)
cax = fig.add_axes([ax[2].get_position().x1+0.01,ax[2].get_position().y0,0.02,ax[2].get_position().height])
cbar = fig.colorbar(im, cax = cax, extend = 'max')
cbar.set_label(r'Selected rate $R$ [bits/sec/Hz]')
# fig.suptitle('Selected rate')

fig.savefig('plots/fig15.pdf', bbox_inches = 'tight')
