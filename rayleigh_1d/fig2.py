# -*- coding: utf-8 -*-
"""
Created on Tue Mar 28 13:06:46 2023

@author: Tobias Kallehauge
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.integrate import quad
import matplotlib.text as mtext


eta = 2
x_max = 100
x_min = 20
x_rng = np.linspace(x_min,x_max, 810)
std_loc = 4
G0 = 1
eps = 1e-3
SNR = 10**(30/10)
delta = [1e-1,1e-3,1e-5]
compute = False

def throughput_backoff(x_hat,x,k):
    R = np.log2(1 - SNR*k*G0*x_hat**(-eta)*np.log(1-eps))
    c_p_out = (1-eps)**(k*(x/x_hat)**eta)
    return(R*c_p_out)

def throughput_interval(x_hat,x, q):
    R = np.log2(1 - SNR*G0*(x_hat + q*std_loc)**(-eta)*np.log(1-eps))
    c_p_out = (1-eps)**((x/(x_hat + q*std_loc))**eta)
    return(R*c_p_out)

def avg_th_ratio_ratio_num_integrate(x_rng, delta,
                                     conf = 1 - 1e-14, 
                                     method = 'backoff'):
    
    
    
    omega = np.zeros(x_rng.size)
    if method == 'backoff':
        throughput = throughput_backoff
        param = (1 - norm.ppf(1-delta)*std_loc/x_min)**eta # k
    elif method == 'interval':
        param  = -norm.ppf(delta)
        throughput = throughput_interval
    
    for i, x in enumerate(x_rng):
        if i % 100 == 0:
            print(i)
        
        # evaluate throughput at x
        f = norm(loc = x, scale = std_loc)
        func_integrate = lambda x_hat : f.pdf(x_hat)*throughput(x_hat,x,param)
        
        # get interval for integration
        a,b = f.interval(confidence = conf)
        
        th_avg = quad(func_integrate, a,b)[0]
        
        # get normalization
        R_eps = np.log2(1 - SNR*G0*x**(-eta)*np.log(1-eps))

        omega[i] = th_avg/(R_eps*(1-eps))
        
        
    return(omega)
    
if compute:
    
    omega_backoff = [avg_th_ratio_ratio_num_integrate(x_rng,d, method = 'backoff') for d in delta]
    omega_interval = [avg_th_ratio_ratio_num_integrate(x_rng,d, method = 'interval') for d in delta]
    with open('throughput_results.npy', 'wb') as f:
        np.save(f,np.array(omega_backoff))
        np.save(f,np.array(omega_interval))
else:
    with open('throughput_results.npy', 'rb') as f:
        omega_backoff = np.load(f)
        omega_interval = np.load(f)
         


#%%

class LegendTitle(object):
    def __init__(self, text_props=None):
        self.text_props = text_props or {}
        super(LegendTitle, self).__init__()

    def legend_artist(self, legend, orig_handle, fontsize, handlebox):
        x0, y0 = handlebox.xdescent, handlebox.ydescent
        title = mtext.Text(x0, y0, orig_handle, usetex=True, **self.text_props)
        handlebox.add_artist(title)
        return title



linestyles = ['-', '-','-']
markers = ['o','^','s']
markevery = 101
markersize = 8
fig, ax = plt.subplots()
for i,d in enumerate(delta):
    k = (1 - norm.ppf(1-d)*std_loc/x_min)**eta # k
    ax.plot(x_rng,omega_backoff[i], c = 'C0', linestyle = linestyles[i],
            label = r'$\delta : 10^{%d}$' %int(np.log10(d)),
            marker = markers[i], markevery = markevery,
            markersize = markersize)
    ax.plot(x_rng,omega_interval[i],c = 'C1', linestyle = linestyles[i],
            label = r'$\delta : 10^{%d}$' %int(np.log10(d)),
            marker = markers[i], markevery = markevery,
            markersize = markersize,
            fillstyle= 'none')
    if i == 0:
        ax.axhline(k, linestyle = ':', label = r'$\beta$')
    else:
        ax.axhline(k, linestyle = ':')
order = [0,3,5,2,1,4,6]
handles, labels = ax.get_legend_handles_labels()
handles = [handles[idx] for idx in order]
labels = [labels[idx] for idx in order]
handles.insert(0, 'Backoff')
labels.insert(0, '')
handles.insert(5, 'Interval')
labels.insert(5, '')
ax.legend(handles, labels, handler_map={str: LegendTitle({'fontsize': 12})},
          loc = 'lower right')
ax.grid()
ax.set_xlabel('UE location x [m]')
ax.set_ylabel(r'Throughput ratio $\overline{T}(x)$')
ax.set_title('Throughput ratio')
fig.savefig('plots/throughput_ratio.pdf', bbox_inches = 'tight')