# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 15:41:53 2023

@author: Tobias Kallehauge
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.patches import Ellipse
import matplotlib.patches as mpatches
from scipy.stats import chi2
from matplotlib.legend_handler import HandlerPatch

import sys
sys.path.insert(0,'../library')
import data_generator_quadriga
import numpy as np
from matplotlib.colors import LogNorm, SymLogNorm, LinearSegmentedColormap
from scipy.stats import multivariate_normal
from scipy.optimize import bisect
import pickle

gray = 0.75 # intensity of checker pattern color
cm = LinearSegmentedColormap.from_list('gray',((gray,gray,gray),(1,1,1)))

# =============================================================================
# Setup drawing ellipses
# =============================================================================

class HandlerEllipse(HandlerPatch):
    def create_artists(self, legend, orig_handle,
                       xdescent, ydescent, width, height, fontsize, trans):
        center = 0.5 * width - 0.5 * xdescent, 0.5 * height - 0.5 * ydescent
        p = Ellipse(xy=center, width=width + xdescent,
                             height=height + ydescent)
        self.update_prop(p, orig_handle, legend)
        p.set_transform(trans)
        return [p]

# =============================================================================
# Run simulation
# =============================================================================

# settings (see print(mod) for other settings)
data_index = 6
k = .5
alpha = 0.05
q = chi2.ppf(1-alpha, df = 2)

# load data
with open(f'data/mod_idx_{data_index}.pickle','rb') as f:
    mod = pickle.load(f)



def get_R_backoff(k):
    R = k*mod.R_eps
    return(R)
    
x = (10,-15, 1.5)
idx = np.linalg.norm(mod.x_cal - x, axis = 1).argmin()
idx_eval = np.linalg.norm(mod.x_eval - x, axis = 1).argmin()
x = mod.x_cal[idx]


C = mod.cov_loc_eval[idx_eval]

eig_val, eig_vec = np.linalg.eig(C)
angle_rad =  np.arctan2(eig_vec[1,0],eig_vec[0,0]) # angle of rotation
angle_deg = angle_rad*180/np.pi

E = Ellipse(x[:2], 2*np.sqrt(eig_val[0]*q),2*np.sqrt(eig_val[1]*q), 
            angle = angle_deg,
            fill = False, 
            edgecolor = 'r', 
            linestyle = '-', 
            linewidth = 1.5,
            label = '95% conf.\ninterval')

f = multivariate_normal.pdf(mod.x_cal[:,:2], mean = x[:2], cov = C )
f_max = f.max()

R = get_R_backoff(k)

p = mod.get_meta_probability(R,[idx_eval])[0]
print(f'p_meta : {p*100:.2f}%')
print(mod.cov_loc_eval[idx_eval].round(1))
print(f'PEB : {np.sqrt(np.trace(C)):.1f} m')

S = mod.R_eps[idx] < R
f[~S] = np.nan

x_max = mod.x_eval.max()
extent = [-x_max,x_max,-x_max,x_max]

# =============================================================================
# Plot
# =============================================================================

fig, ax = plt.subplots(figsize = (4,4))

n = 40
Z1 = np.add.outer(range(n), range(n)) % 2  # chessboard
ax.imshow(Z1, interpolation='nearest', extent=extent, cmap = cm)

im2 = ax.imshow(f[mod.idx_eval].reshape(mod.N_side_eval,mod.N_side_eval), cmap = 'jet',
          label = 'Outage region',
          origin = 'lower',
          extent = extent,
          norm = SymLogNorm(linthresh = 1e-15))


# UE and BS locations
BS = mod.x_BS[mod.BS_data][:2]
ax.plot(BS[0],BS[1], markersize = 10,
        marker = 'X', markerfacecolor='w', 
        markeredgecolor='k', markeredgewidth=1.0, linewidth = 0,
        label = 'BS')
ax.plot(x[0],x[1], markersize = 10,
        marker = '1', markerfacecolor='w', 
        markeredgecolor='k', markeredgewidth=1.5, linewidth = 0,
        label = 'UE')

# colorbars
divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="7%", pad=0.09)
cbar = fig.colorbar(im2, cax=cax)
cbar.set_label('Localization PDF')

# confidence interval
ax.add_patch(E)

# legends
handles, labels = ax.get_legend_handles_labels()
c = mpatches.Circle((0.5, 0.5), 0.25, facecolor="none",
                    edgecolor="red", linewidth=1.5)
handles[2] = c
ax.legend(handles = handles, labels = labels, handler_map={mpatches.Circle: HandlerEllipse()})


# titles and labels
ax.set_xlabel('1st cordinate [m]')
ax.set_ylabel('2nd cordinate [m]')

# ax limits
ax.set_xlim(-55,55)
ax.set_ylim(-55,55)
# # various
fig.subplots_adjust(wspace = 0.25)
fig.savefig('plots/fig7.pdf', bbox_inches = 'tight')