# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 14:28:58 2023

@author: Tobias Kallehauge
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import (OffsetImage, AnnotationBbox)
import matplotlib.image as image
from scipy.stats import norm
from matplotlib.patches import Rectangle


UE = image.imread('UE.png')
BS = image.imread('BS.png')

def add_image(ax, im, coor, zoom):
    im_offset = OffsetImage(im,zoom = zoom)
    box = AnnotationBbox(im_offset,coor, frameon= False)
    ax.add_artist(box)

eta = 2
x_max = 100
std_loc = 4
x = 50
G0 = 1
k = 0.5
eps = 1e-5
SNR = 10**(30/10)

x_rng = np.linspace(1,x_max, 1000)
R_est = np.log2(1 - SNR*k*G0*x_rng**(-eta)*np.log(1-eps))
dx = x*(1-k**(1/eta)) 
R_eps = np.log2(1 - SNR*G0*x**(-eta)*np.log(1-eps))


fig, ax = plt.subplots()


ax.semilogy(x_rng, R_est, label = r'$R(\hat{x})$', 
        color = 'C0', marker = 'o', markevery = 50)
# ax.axvline(x - dx, color = 'r')
ax.axvspan(-5,x-dx, color = 'r', alpha = 0.3, label = 'Outage region')
ax.axhline(R_eps, label = r'$C_{\epsilon}(x)$', c = 'C3',linestyle = '--')
ax.set_xlabel(r'Estimated location $\hat{x}$ [m]')
ax.set_ylabel(r'Rate [bits/s/Hz]')
ax.set_xlim(-5,105)


ax2 = ax.twinx()
col2 = 'C2'
ax2.plot(x_rng, norm.pdf(x_rng, loc = x, scale = std_loc),color = col2, 
         label = r'$p(\hat{x})$', marker = '^', markevery = 49)
ax2.set_ylabel('Localization density', color = col2)
ax2.tick_params(axis='y', labelcolor=col2)

# add dX
height = 0.035
width = 18
ax2.annotate('', xy=(x,height), xytext=(x-dx,height), arrowprops=dict(arrowstyle='<->'))
ax2.add_patch(Rectangle((x - dx/2 - width/2, height+0.002), width, .006,
             facecolor = 'w',
             fill=True,
             lw=5,
             zorder=2
             ))
ax2.text(x - dx/2, height+0.003,r'$x(1-\beta^{1/\eta})$', horizontalalignment = 'center')


# add images
add_image(ax2,UE,(x,0.005), 0.02)
add_image(ax2,BS,(0,0.011), 0.045)

# fix legends in same box
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
lines = lines1 + lines2
labels = labels1 + labels2
order = [0,2,3,1]
labels = [labels[i] for i in order]
lines = [lines[i] for i in order]
ax2.legend(lines,labels , loc=0)
fig.savefig('plots/fig1.pdf', bbox_inches = 'tight')

p_meta =  norm.cdf(x - dx, loc = x, scale = std_loc)
print(f'Meta probability: {p_meta*100:.5f}%')
