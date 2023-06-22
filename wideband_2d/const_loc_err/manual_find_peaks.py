# -*- coding: utf-8 -*-
"""
Created on Tue May 23 09:46:59 2023

@author: Tobias Kallehauge
"""

from matplotlib.colors import  LogNorm
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import pickle
import numpy as np
import sys
sys.path.insert(0, '../../../library')
import data_generator_quadriga
import PySimpleGUI as sg
# import various funtions from matplotlib



data_index = 6
delta = 0.05
log = True
im_pixels = 849
version = 2

# load data
with open(f'data/mod_idx_{data_index}_const_loc.pickle', 'rb') as f:
    mod = pickle.load(f)

with open(f'data/backoff_{data_index}_const_loc.pickle', 'rb') as f:
    res_backoff = pickle.load(f)

with open(f'data/interval_{data_index}_const_loc.pickle', 'rb') as f:
    res_interval = pickle.load(f)


# =============================================================================
# Save map figure as image
# =============================================================================

# 2D array example
X  = np.log(mod.R_eps.reshape(mod.N_side_cal,mod.N_side_cal))
# X = np.flipud(X)

fig, ax = plt.subplots()
im = ax.imshow(mod.R_eps.reshape(mod.N_side_cal,mod.N_side_cal),
            cmap = 'jet', extent = mod.extent, origin = 'lower',
            norm = LogNorm())

rec = Rectangle((-50,-50),100,100, edgecolor = 'k', linestyle = '--',
                facecolor = 'none')
ax.add_patch(rec)
ax.set_xticks([])
ax.set_yticks([])
ax.set_axis_off()
fig.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
fig.savefig('plots/find_peaks.png', dpi=200, bbox_inches = 'tight')


size = 900
graph = sg.Graph((size,size), graph_bottom_left= (0,0), 
                  graph_top_right=(size,size),
                  background_color = 'white',
                  key = '-graph-',
                  enable_events = True)


# =============================================================================
# Setup pysimple gui
# =============================================================================

button_names = ['-peak-','-valley-','-erase-']


layout = [[sg.Text('Locate UPPER LEFT corner of the cell', key = '-msg-')],
          [graph],
          [sg.Checkbox('Peak', True, enable_events=True, key = '-peak-'),
           sg.Checkbox('Valley', False, enable_events =True, key = '-valley-'),
           sg.Checkbox('Erase', False, enable_events =True, key = '-erase-')],
          [sg.Button('Exit', key = '-exit-')]]


window = sg.Window('Find peaks',
                    layout = layout,
                    finalize = True,
                    element_justification='center')
graph = window['-graph-']

graph.draw_image('plots/find_peaks.png', 
                             location = (size/2 - im_pixels/2,size/2 + im_pixels/2))
pos_type = '-peak-'

# =============================================================================
# first locate corners of the cell (already done)
# =============================================================================

# upper left corner
# pos_ul = window.read()[1]['-graph-']
# print(pos_ul)

# window['-msg-'].update('Locate LOWER RIGHT corner of the cell')
# pos_lr = window.read()[1]['-graph-']
# print(pos_lr)

pos_ul = (50,852)
pos_lr = (842,60)


# =============================================================================
# run 
# =============================================================================
  
window['-msg-'].update('Locate peaks and valeys')

rng_x = np.linspace(pos_ul[0],pos_lr[0], mod.N_side_cal)
rng_y = np.linspace(pos_lr[1],pos_ul[1], mod.N_side_cal)
points = {'-peak-' : [],
          '-valley-': []}


while True: 
    event, values = window.read()
    print(event,values)  
    
    if event == '-graph-':
        pos = values['-graph-']
        
        # fix position to grid
        idx_x = np.argmin(np.abs(pos[0] - rng_x))
        idx_y = np.argmin(np.abs(pos[1] - rng_y))
        pos_grid = (rng_x[idx_x],rng_y[idx_y])
        
        figures = graph.get_figures_at_location(pos_grid)
        
        if pos_type in ('-peak-','-valley-'):
            if len(figures) == 1: # only draw when there is not point 
                #draw
                if pos_type == '-peak-':
                    fill_color = 'white'
                elif pos_type == '-valley-':
                    fill_color = 'black'
                graph.draw_circle(pos_grid, 4, line_color = 'black',
                                              fill_color = fill_color,
                                              line_width = 2)
                # then add to dictionary
                points[pos_type].append(pos_grid)
                
                
        else:
            figures = graph.get_figures_at_location(pos) # a maximum of 1
            
            for figure in figures[1:]: # first one is always the background image
                # delete from graph
                graph.delete_figure(figure)
            
                # delete from point list
                if pos_grid in points['-peak-']:
                    idx = points['-peak-'].index(pos_grid)
                    points['-peak-'].pop(idx)
                if pos_grid in points['-valley-']:
                    idx = points['-valley-'].index(pos_grid)
                    points['-valley-'].pop(idx)
        
    if event in button_names:
        for name in button_names:
            if event != name:
                window[name].update(False)
        pos_type = event
    
    if event in (sg.WIN_CLOSED, '-exit-'):
        break
    
window.close()

# =============================================================================
# converent gui coordinates to meters within the cell
# =============================================================================

def convert_to_meters(point_gui):
    x_gui_norm = (point_gui[0] - pos_ul[0])/(pos_lr[0] - pos_ul[0])
    y_gui_norm = (point_gui[1] - pos_lr[1])/(pos_ul[1] - pos_lr[1])
    
    x_meter = x_gui_norm*(mod.extent[1] - mod.extent[0]) + mod.extent[0]
    y_meter = y_gui_norm*(mod.extent[3] - mod.extent[2]) + mod.extent[2]
    return(x_meter,y_meter)

peaks = np.array([convert_to_meters(peak) for peak in points['-peak-']])
valleys = np.array([convert_to_meters(valley) for valley in points['-valley-']])
    
# =============================================================================
#%% Show peaks and valleys
# =============================================================================

# 2D array example

fig, ax = plt.subplots()
im = ax.imshow(mod.R_eps.reshape(mod.N_side_cal,mod.N_side_cal),
            cmap = 'jet', extent = mod.extent, origin = 'lower',
            norm = LogNorm())

rec = Rectangle((-50,-50),100,100, edgecolor = 'k', linestyle = '--',
                facecolor = 'none')
ax.add_patch(rec)
if len(valleys) > 0:
    ax.scatter(valleys[:,0], valleys[:,1], c = 'k', s = 3)
if len(peaks) > 0:
    ax.scatter(peaks[:,0], peaks[:,1], c = 'w', s = 6, edgecolors= 'k',
               linewidth = 0.5)

# =============================================================================
#%% save as indicies
# =============================================================================

idx_peaks = np.zeros(mod.N_sim, dtype = 'bool')
if len(idx_peaks) > 0:
    idx = np.argmin(np.linalg.norm((peaks[:,None,:] - mod.x_cal[None, : , :2]), axis = 2), axis = 1)
    idx_peaks[idx] = True

idx_valleys = np.zeros(mod.N_sim, dtype = 'bool')
if len(idx_valleys) > 0:
    idx = np.argmin(np.linalg.norm((valleys[:,None,:] - mod.x_cal[None, : , :2]), axis = 2), axis = 1)
    idx_valleys[idx] = True

np.save(f'data/peaks_manual_idx_{data_index}_v{version}.npy',idx_peaks)
np.save(f'data/valleys_manual_idx_{data_index}_v{version}.npy',idx_valleys)

