# -*- coding: utf-8 -*-
"""
Created on Tue May 23 09:46:59 2023

@author: Tobias Kallehauge
"""

from matplotlib.colors import LogNorm
import matplotlib.pyplot as plt
from findpeaks import findpeaks
import pickle
import numpy as np
import sys
sys.path.insert(0, '../library')
import data_generator_quadriga


data_index = 6
delta = 0.05
peak_method = 'manual'
whis = [5,95]

# load data
with open(f'data/mod_idx_{data_index}_const_loc.pickle', 'rb') as f:
    mod = pickle.load(f)

with open(f'data/backoff_{data_index}_const_loc.pickle', 'rb') as f:
    res_backoff = pickle.load(f)

with open(f'data/interval_{data_index}_const_loc.pickle', 'rb') as f:
    res_interval = pickle.load(f)

# =============================================================================
# Find peaks and valleys
# =============================================================================

if peak_method == 'automatic':
    # 2D array example
    X  = np.log(mod.R_eps.reshape(mod.N_side_cal,mod.N_side_cal))
    # X = np.flipud(X)
    
    im_length = mod.N_side_cal # size of pre-processed data - keep same as size of X
    
    window_size = 9
    
    # first find peaks
    fp = findpeaks(method='mask', denoise='mean', window= window_size, 
                   imsize=(im_length,im_length), 
                   togray = False)
    results_peaks = fp.fit(X)
    # fp.plot(cmap = 'jet')
    
    # then valeys
    fp = findpeaks(method='mask', denoise='mean', window= window_size, 
                   imsize=(im_length,im_length), 
                   togray = False)
    results_valeys = fp.fit(-X)
    idx_peaks = results_peaks['Xranked'].flatten() != 0
    
    idx_valeys = results_valeys['Xranked'].flatten() != 0

elif peak_method == 'manual':
    # load from file
    idx_peaks = np.load(f'data/peaks_manual_idx_{data_index}.npy')
    idx_valeys = np.load(f'data/valleys_manual_idx_{data_index}.npy')
    

# =============================================================================
#%% Associate points with either peak or valey
# =============================================================================



peaks = mod.x_cal[idx_peaks]

valeys = mod.x_cal[idx_valeys]

points = np.vstack((peaks,valeys))
peaks_idx = np.hstack((np.ones(len(peaks)),np.zeros(len(valeys)))).astype('bool')


n_peaks = sum(idx_peaks[mod.idx_eval])
n_valeys = sum(idx_valeys[mod.idx_eval])
print(f'Number of peaks: {n_peaks}\nNumber of valyes : {n_valeys}')
    
# =============================================================================
#%% plot
# =============================================================================

def set_box_color(bp, color):
    plt.setp(bp['boxes'], color=color)
    plt.setp(bp['whiskers'], color=color)
    plt.setp(bp['caps'], color='k')
    plt.setp(bp['medians'], color='k')
    plt.setp(bp['fliers'], markeredgecolor = color)

fig, ax = plt.subplots(ncols = 4, width_ratios=[2, 1,1,0.5],
                       height_ratios = [1],
                       figsize = (10,14))
for i in range(1,3):
    ax[i].yaxis.tick_right()
    ax[i].set_box_aspect(2)
fig.subplots_adjust(wspace = 0.5)
ax[0].set_title(r'$\epsilon$-outage capacity')
im = ax[0].imshow(mod.R_eps[mod.idx_eval].reshape(mod.N_side_eval,mod.N_side_eval),
            cmap = 'jet', extent = mod.extent_eval, origin = 'lower',
            norm = LogNorm())
ax[0].scatter(points[:,0][peaks_idx], points[:,1][peaks_idx], c = 'w',
            cmap = 'binary', label = 'Peaks', edgecolor = 'k')
ax[0].scatter(points[:,0][~peaks_idx], points[:,1][~peaks_idx], c = 'k',
            cmap = 'binary', label = 'Valleys')
ax[0].set_xlim(-50,50)
ax[0].set_ylim(-50,50)
ax[0].legend(loc = 'upper right')
ax[0].set_ylabel('UE 2nd coordinate [m]')
ax[0].set_xlabel('UE 1st coordinate [m]')

cax = fig.add_axes([ax[0].get_position().x1+0.01,ax[0].get_position().y0,0.02,ax[0].get_position().height])
cbar = fig.colorbar(im, cax = cax)

n_peaks = sum(idx_peaks[mod.idx_eval])
n_valeys = sum(idx_valeys[mod.idx_eval])


spacing = 0.15
ax[1].set_yscale('symlog', linthresh = 1e-15)
bp = ax[1].boxplot(res_backoff['p'][idx_peaks[mod.idx_eval]], positions = [-spacing], labels = ['Peaks'], whis = whis); set_box_color(bp,'C0')
bp = ax[1].boxplot(res_interval['p'][idx_peaks[mod.idx_eval]], positions = [+spacing], labels = ['Peaks'], whis = whis); set_box_color(bp,'C1')
bp = ax[1].boxplot(res_backoff['p'][idx_valeys[mod.idx_eval]],positions = [1 - spacing], labels = ['Valeys'],whis = whis); set_box_color(bp,'C0')
bp = ax[1].boxplot(res_interval['p'][idx_valeys[mod.idx_eval]],positions = [1 + spacing], labels = ['Valeys'], whis = whis); set_box_color(bp,'C1')
# ax[1].set_ylabel('Meta probability')
ax[1].set_xticks([0,1], ['Peaks', 'Valleys'])
ax[1].plot([], c='C0', label='Backoff')
ax[1].plot([], c='C1', label='Interval')
ax[1].legend(loc = 'upper left', fontsize = 7)
ax[1].set_title('Meta probability')


spacing = 0.15
bp = ax[2].boxplot(res_backoff['T'][idx_peaks[mod.idx_eval]], positions = [-spacing], labels = ['Peaks']); set_box_color(bp,'C0')
bp = ax[2].boxplot(res_interval['T'][idx_peaks[mod.idx_eval]], positions = [+spacing], labels = ['Peaks']); set_box_color(bp,'C1')
bp = ax[2].boxplot(res_backoff['T'][idx_valeys[mod.idx_eval]],positions = [1 - spacing], labels = ['Valeys']); set_box_color(bp,'C0')
bp = ax[2].boxplot(res_interval['T'][idx_valeys[mod.idx_eval]],positions = [1 + spacing], labels = ['Valeys']); set_box_color(bp,'C1')
# ax[1].set_ylabel('Meta probability')
ax[2].set_xticks([0,1], ['Peaks', 'Valleys'])
ax[2].plot([], c='C0', label='Backoff')
ax[2].plot([], c='C1', label='Interval')
ax[2].legend(loc = 'upper left', fontsize = 7)
ax[2].set_title('Throughput ratio')

# add boxplot explanation
np.random.seed(72)
# data_dummy = np.random.standard_t(df = 20, size = 50)
data_dummy = np.random.normal(size = 200)
ax[3].set_box_aspect(4)
ax[3].get_xaxis().set_visible(False)
ax[3].get_yaxis().set_visible(False)
ax[3].set_frame_on(False)
bp = ax[3].boxplot(data_dummy, whis = [1,99], positions = [1])
plt.setp(bp['medians'], color='k')
ax[3].text(1.2,0, 'Median', ha = 'left', va = 'center')
ax[3].text(1.2,np.quantile(data_dummy,0.25), 'Q1', ha = 'left', va = 'center')
ax[3].text(1.2,np.quantile(data_dummy,0.75), 'Q3', ha = 'left', va = 'center')
ax[3].text(1.2,np.quantile(data_dummy,0.01), r'$q_{\alpha}$', ha = 'left', va = 'center')
ax[3].text(1.2,np.quantile(data_dummy,0.99), r'$q_{1- \alpha}$', ha = 'left', va = 'center')
ax[3].text(1.2,max(data_dummy), r'Outliers', ha = 'left', va = 'center')
ax[3].text(1.2,min(data_dummy), r'Outliers', ha = 'left', va = 'center')


fig.savefig('plots/fig11.pdf', bbox_inches = 'tight')
