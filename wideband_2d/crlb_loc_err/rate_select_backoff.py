# -*- coding: utf-8 -*-
"""
Created on Mon May  1 11:27:12 2023

@author: Tobias Kallehauge
"""

import sys
sys.path.insert(0, '..')
import data_generator_quadriga
import numpy as np
import matplotlib.pyplot as plt
import pickle
from matplotlib.colors import SymLogNorm, LogNorm
from scipy.optimize import bisect
from statsmodels.distributions.empirical_distribution import ECDF

# settings (see print(mod) for other settings)
data_index = 6
delta = 5e-2

# load data
with open(f'data/mod_idx_{data_index}.pickle','rb') as f:
    mod = pickle.load(f)
    



# =============================================================================
#%% Select rate with backoff approach
# =============================================================================

# setup bisection method
def get_R_backoff(k):
    R = k*mod.R_eps
    return(R)

f_bisect_backoff = lambda k : mod.get_meta_probability(get_R_backoff(k)).max() - delta

k = bisect(f_bisect_backoff, 0, 2)
# k = 0.03501305991994741
print(f'k : {k:.2f}')


R = get_R_backoff(k)
p =  mod.get_meta_probability(R)

# get throughput at selected rate
T = mod.get_throughput(R)

# =============================================================================
#%% some plots
# =============================================================================


fig, ax = plt.subplots()
F = ECDF(p)
x_rng = 10**(np.linspace(-300,np.log10(p.max()), 1000))
ax.semilogx(x_rng,F(x_rng))
ax.axvline(delta, c = 'r')
ax.set_xlabel('Meta probability')
ax.set_title('Meta probability')
ax.set_ylabel('CDF')

# thorughput ratio
fig, ax = plt.subplots()
F = ECDF(T)
x_rng = np.linspace(0,1,1000)
ax.plot(x_rng,F(x_rng))
ax.set_xlabel('Throughput')
ax.set_title('Throughput')
ax.set_ylabel('CDF')
ax.axvline(k, c = 'r')


fig, ax = plt.subplots(figsize = (4,4))
ax.set_title('Rate backoff')
im = ax.imshow(R.reshape(mod.N_side_cal,mod.N_side_cal), 
          extent =mod.extent, norm = LogNorm(),
          cmap = 'jet',
          origin = 'lower')
# plt.semilogy(mod.x_cal, mod.R_eps, label = 'Epsilon outage capacity')
ax.set_xlabel(r'Location $x_0$ [m]')
ax.set_ylabel(r'Location $x_1$ [m]')
fig.colorbar(im)

# # Plot achieved meta-probability
fig, ax = plt.subplots(figsize = (4,4))
ax.set_title('Meta probability  backoff')
im = ax.imshow(p.reshape(mod.N_side_eval,mod.N_side_eval), 
          extent =mod.extent, norm = SymLogNorm(linthresh = 1e-8,vmax = delta),
          cmap = 'jet',
          origin = 'lower')
# plt.semilogy(mod.x_cal, mod.R_eps, label = 'Epsilon outage capacity')
ax.set_xlabel(r'Location $x_0$ [m]')
ax.set_ylabel(r'Location $x_1$ [m]')
fig.colorbar(im)


fig, ax = plt.subplots(figsize = (4,4))
ax.set_title('Throughput backoff')
im = ax.imshow(T.reshape(mod.N_side_eval,mod.N_side_eval), 
          extent =mod.extent,
          cmap = 'jet',
          origin = 'lower')
ax.set_xlabel(r'Location $x_0$ [m]')
ax.set_ylabel(r'Location $x_1$ [m]')
fig.colorbar(im)


# =============================================================================
#%% save results
# =============================================================================

res = {'R': R, 'p': p, 'T': T, 
        'k': k, 'delta': delta}

with open(f'data/backoff_{data_index}.pickle', 'wb') as f:  # Overwrites any existing file.
    pickle.dump(res, f, pickle.HIGHEST_PROTOCOL)
