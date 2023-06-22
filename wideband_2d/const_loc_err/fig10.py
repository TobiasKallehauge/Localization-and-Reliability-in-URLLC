# -*- coding: utf-8 -*-
"""
Created on Mon May 15 14:01:30 2023

@author: Tobias Kallehauge
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle
from statsmodels.distributions.empirical_distribution import ECDF

data_index = 6
delta = 0.05

# load data
with open(f'data/backoff_{data_index}_const_loc.pickle','rb') as f:
    res_backoff = pickle.load(f)
    
with open(f'data/interval_{data_index}_const_loc.pickle','rb') as f:
    res_interval = pickle.load(f)


# =============================================================================
#%% Plot meta probability CDF
# =============================================================================


fig, ax = plt.subplots(nrows = 2)
fig.subplots_adjust(hspace = 0.6)

# meta probability
F_backoff = ECDF(res_backoff['p'])
F_interval = ECDF(res_interval['p'])
x_rng = 10**(np.linspace(-15,0, 1000))
ax[0].semilogx(x_rng,F_backoff(x_rng), label = r'Backoff, $\beta$' + f' : {res_backoff["k"]:.3f}')
ax[0].semilogx(x_rng,F_interval(x_rng), label = r'Interval, $q^2$' + f' : {res_interval["q"]:.2f}')
ax[0].axvline(res_backoff['delta'], c = 'r', label = r'$\delta : 5\%$')
ax[0].set_xlabel(r'$\tilde{p}_{\epsilon}$')
ax[0].set_title('Meta probability distribution within the cell')
ax[0].set_ylabel('CDF')
ax[0].legend(loc = 'lower left')

# throughput ratio
F_backoff = ECDF(res_backoff['T'])
F_interval = ECDF(res_interval['T'])
x_rng = np.linspace(0,1,1000)
ax[1].plot(x_rng,F_backoff(x_rng), label = r'Backoff, $\beta$' + f' : {res_backoff["k"]:.3f}')
ax[1].axvline(res_backoff['k'], c = 'C0', linestyle = ':', label = r'$\beta$')
ax[1].plot(x_rng,F_interval(x_rng), label = r'Interval, $q^2$' + f' : {res_interval["q"]:.2f}')
ax[1].set_xlabel(r'$\overline{T}$')
ax[1].set_title('Throughput ratio distribution within the cell')
ax[1].set_ylabel('CDF')
ax[1].legend(loc = 'lower right')

fig.savefig('plots/fig10.pdf', bbox_inches = 'tight')

# =============================================================================
#%% Throughput
# =============================================================================

print(f'T average, backoff: {res_backoff["T"].mean():.3}')
print(f'T average, interval: {res_interval["T"].mean():.3f}')
