# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 07:54:02 2019

@author: rkrylov1
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from numpy import pi
from numpy import concatenate as concat
from SinglePedestrianModel import vvmodel
from PedestrianBridgeModel import pedestrian_bridge 

#params = (w,h,a)
#vvmodel(phi, t, params)
#params=  (mu, w, g, l, lambda, a, k, h, p)
#pedestrian_bridge(phi_x, t, params)

#params = (mu, 1, 10, 1, , 0.2, 1, 0.05, 1)

def graph_specifications(graph_type = None, xlim = (0, 1), ylim = (0,1), title = '', param_tuple = (), \
                         xlabel = '', ylabel = ''):
    if graph_type == 'phase plane': 
        plt.rcParams.update({'font.size':20})
        plt.figure(figsize=(12,8), dpi=80)
        plt.xlim(xlim)
        plt.ylim(ylim)
        plt.xlabel(r'${}$'.format(xlabel), labelpad=200, fontweight=1000)
        plt.ylabel(r'${}$'.format(ylabel), labelpad = 310, fontweight = 1000)
        plt.title(title)
        try:
            parameters = r'${} = {}$,' * (len(param_tuple)/2 - 1) + '{} = {}'
            plt.suptitle(parameters.format(*param_tuple), y = 0.965)
        except:
            pass
        #moving axis
        # Move left y-axis and bottim x-axis to centre, passing through (0,0)
        ax = plt.subplot()
        ax.spines['left'].set_position('center')
        ax.spines['bottom'].set_position('center')
        ax.xaxis.get_ticklabels
        # Eliminate upper and right axes
        ax.spines['right'].set_color('none')
        ax.spines['top'].set_color('none')
        ax.axhline(linewidth=4, color='k')
        ax.axvline(linewidth=4, color='k')
        xticks = ax.xaxis.get_major_ticks()
        xticks[4].label1.set_visible(False)
        #yticks = ax.yaxis.get_major_ticks()
        #yticks[3].update_position(200)
        plt.xticks(fontweight=700)
        plt.yticks(fontweight=700)
    else: 
        plt.rcParams.update({'font.size':20})
        plt.figure(figsize=(12,8), dpi=80)
        plt.xlabel(r'${}$'.format(xlabel), fontweight=1000)
        plt.ylabel(r'${}$'.format(ylabel), fontweight = 1000)
        plt.title(title)
    
    return


#creates a square array of points from the passed bottom left corner(x, y)
#up to the top right corner (x+r, y+r) with a sampling rate of sampling param. 
#Returns an np.array of the form [[x1, y1], [x1, y2], ..., [xn,y_(n-1)], [xn, yn]]
def neighborhood(corner, r, sampling):
    x, y = corner
    x_arr = np.linspace(x, x+r, sampling)
    y_arr = np.linspace(y, y+r, sampling)

    xy_matrix = [[]]
    for x in x_arr:
        for y in y_arr:
            xy_matrix += [[x, y]]
        
    return np.array(xy_matrix[1:])


#difference - a 1d array consisting of length N, where columns are paired
#as [x1,y1,x2,y2,x3,y3,...]. 
#returns scaled normalized difference to make the phase plane diagram legible
def diff_normalization(difference):
    size = difference.size
    diff_2d = difference.reshape((int(size/2), 2))
    row_norms = np.array([np.linalg.norm(row) for row in diff_2d])
    diff_normalized = diff_2d/(row_norms[:, None])
    
    return diff_normalized


#creates and saves the plot of phase plane vector field with few selected
#trajectories in the current directory, based on the model used in odeint(...)
#params - parameters for model we are researching
    


def selected_traj(func, params, coords, interval_cond):
    t = np.linspace(*interval_cond)
    params = tuple(params)
    
    try:
        trajectories = odeint(func, coords[0], t, args=params)
    except:
        return (0, 0)
        
    for xy in coords[1:]: 
        trajectories = concat((trajectories, odeint(func, xy, t, args=params)))
        t = np.concatenate((t,t))
    
    return (trajectories, t)

#ax - axis for plotting
#bounds = {(x, y), radius, sampling} - for the phase plane window, used for neighborhood
#coords = [..., (x_i, y_i), ...] - 
#func - model to be integrated over
#params - parameters for the model
    
def time_series(traj, t, xlabel, ylabel, fig_name):
   
    graph_specifications(xlabel = xlabel, ylabel = ylabel, title = fig_name)
    
    plt.plot(t, traj)
    plt.savefig("./{}.pdf".format(fig_name))
    plt.show()
    
    return

def phase_plane_traj(func, coords, params, interval_cond, func_vars = [0, 1], traj_type = 'reg', mod_param = 0):
    print('plotting_trajectories')
    start, end, sampling = interval_cond
    t = np.linspace(*interval_cond)

    if traj_type == 'reg':
        for coord in coords:
            traj = odeint(func, coord, t, args=params)[:, func_vars]
            plt.plot(traj[end-100:, 0], traj[end-100:, 1], color='b')
            
    elif traj_type == 'modulo' or traj_type == 'mod':
        for coord in coords:
            traj = odeint(func, coord, t, args=params)[:, func_vars]
            quotient, remainder = np.divmod(traj[:, 0], mod_param)
            rem_odd_div = np.where(np.mod(quotient, 2) == 1)
            remainder[rem_odd_div] -= 0.5
        
            plt.plot(remainder[end-100:], traj[end-100:, 1], color='b')
            
    return

#vector_field(bounds, func, params, fig_name, coords=(), func_vars = [0,1], \
#                 param_tuple = (), fig_title = '', arrows = True, interval_cond = (0, 5000, 50000), 
#                 traj_type = 'reg', mod_param = 0, **kwargs
def vector_field(bounds, func, params, fig_name, func_vals, coords=(), func_vars = [0,1], \
                 param_tuple = (), fig_title = '', arrows = True, interval_cond = (0, 5000, 100000), 
                 traj_type = 'reg', mod_param = 0, **kwargs):
    corner, r, sampling = bounds        
    try:
        xlim = kwargs['xlim']
        ylim = kwargs['ylim']
    except:
        x, y = corner
        xlim = (x, x+r)
        ylim = (y, y+r)
    
    try:
        xlabel = kwargs['xlabel']
        ylabel = kwargs['ylabel']
    except:
        xlabel = 'x'
        ylabel = 'y'
        
    xy = neighborhood(*bounds)
    t = np.linspace(0, 1, 10)
    phi_integrated_arr = np.empty((10, 2))
    
    extra_vals = np.repeat(func_vals, xy.shape[0], axis=0)
    xy = concat((extra_vals, xy), axis=1)
    print(xy)
    
    for phi in xy:
        
        solution  = odeint(func, phi, t, args=params)[:, func_vars]
        phi_integrated_arr = concat((phi_integrated_arr, solution), axis=1)

    sol_arr = phi_integrated_arr[:, 2:]
    difference = sol_arr - np.roll(sol_arr, 1, axis=0)
    sol_arr = sol_arr[0]
    difference = difference[1]
    scaling_factor = r/sampling
    diff_normalized = diff_normalization(difference)*scaling_factor*0.5
    
    graph_specifications(graph_type = 'phase plane', xlabel = xlabel, ylabel = ylabel, param_tuple = param_tuple, \
                          xlim = xlim, ylim = ylim, title = fig_title)
    size = sol_arr.size
    sol = sol_arr.reshape((int(size/2), 2))
    if arrows:
        for start, change in zip(sol, diff_normalized):        
            plt.arrow(start[0], start[1], change[0], change[1], width = scaling_factor*0.1, color="#808080")
        
    if coords != ():
        phase_plane_traj(func, coords, params, interval_cond, func_vars = func_vars, traj_type = traj_type, \
                         mod_param = mod_param)
    
    
    plt.savefig("./{}.pdf".format(fig_name))
    plt.show()
    
    return 'vector_field'
    
#vector_field(bounds, func, params, fig_name, func_vals, coords=(), func_vars = [0,1], \
#                 param_tuple = (), fig_title = '', arrows = True, interval_cond = (0, 5000, 50000), 
#                 traj_type = 'reg', mod_param = 0, **kwargs

params = (0.1, 0.7, 10, 1, 1, 1, 1, 0.05, 0.3)
bounds = ((-0.4, -0.4), 0.5, 100)
coords = [[0.1, 0.1, 0, 0]]
fig_name = 'PhiVectorField'
func_vars = [0, 1]
param_tuple = ('\mu', 0.1, 'w', 0.7, '\lambda', 1, 'a', 1, 'k', 1, 'h', 0.01)
fig_title = r'$\dot{\phi}$ vs $x\phi$'

vector_field(bounds, pedestrian_bridge, params, fig_name, [[0.1,0.1]], coords=coords, func_vars=[0,1], param_tuple=param_tuple, \
             traj_type = 'mod', mod_param = 0.3, arrows = False, xlabel = '\phi', ylabel = '\dot{\phi}', xlim = (-0.6, 0.4), ylim = (0.5, 1.5))

trajectories, t = selected_traj(pedestrian_bridge, params, coords, (0, 2000, 100000))

traj1 = trajectories[99900:100000, 2]
t = t[99900:100000]
time_series(traj1, t, 't', '\phi', 'PHITimeSeries')
    
"""
bounds = ((-0.3, -0.3), 0.6, 60)
#coords_sp1 = (())
#coords_sp2 = 
coords_bp = ()
#params: (mu, w, g, l, lambda, a, k, h, p)
#params = (0.1, 0.7, 10, 1, 0.1, 0.2, 1, 0.1, 0.3)
fig_name = "PhaseDiagramPBPHI_mu_0.1_w_07_lambda_1_a_1_k_072_h_005_p_15"

#vector_field(bounds, coords_bp, fig_name, pedestrian_bridge, params)

#phase_plane_traj(pedestrian_bridge, coords_bp, params, bounds, fig_name)
#time_series(pedestrian_bridge, params, "PHITimeSeries_mu_0.1_w_07_lambda_1_a_1_k_072_h_005_p_15")
single_params = (0.7, 1, 0.72, 0.5)
single_ped_fig = 'SinglePedestrianTest'
single_bounds = ((-1.5, -1.5), 3, 60)
coords_sp = ()
vector_field(single_bounds, coords_sp, single_ped_fig, vvmodel, single_params)
"""

"""
    #phi_lc1 = (-pi/2 + 0.05, 2.5)
    phi_lc1 = (-8, 2.5)
    lc1 = odeint(vvmodel, phi_lc1, t2, args=params)
    
    #phi_lc2 = (pi/2 - 0.05, -2.5)
    phi_lc2 = (8, -2.5)
    lc2 = odeint(vvmodel, phi_lc2, t2, args=params)
    
    phi_npi = (-pi+0.1, 0.05)
    utraj_npi = odeint(vvmodel, phi_npi, t2, args=params)
    
    phi_pi = (pi+0.1, 0.05)
    utraj_pi = odeint(vvmodel, phi_pi, t2, args=params)
    
    phi_ssaddle1 = (-0.025,0.05)
    ssaddle1 = odeint(vvmodel, phi_ssaddle1, t3, args=params)
    
    phi_ssaddle2 = (0.025, -0.05)
    ssaddle2 = odeint(vvmodel, phi_ssaddle2, t3, args=params)
    
    phi_usaddle1 = (0.025, 0.05)
    usaddle1 = odeint(vvmodel, phi_usaddle1, t2, args=params)
    
    phi_usaddle2 = (-0.025, -0.05)
    usaddle2 = odeint(vvmodel, phi_usaddle2, t2, args=params)
"""