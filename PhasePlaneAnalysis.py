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

def graph_specifications(graph_type = None, xlim = None, ylim = None, xlabel = None, ylabel = None, \
                         title = None, param_tuple = None):
    if graph_type == 'phase plane': 
        #specifications to move axis to the center of the graph
        plt.rcParams.update({'font.size':20})
        plt.figure(figsize=(12,8), dpi=80)
        if xlim != None:
            plt.xlim(xlim)
        if xlim != None:
            plt.ylim(ylim)
            
        if xlabel != None:
            plt.xlabel(r'${}$'.format(xlabel), labelpad=200, fontweight=1000)

        if ylabel != None:
            plt.ylabel(r'${}$'.format(ylabel), labelpad = 310, fontweight = 1000)

        if title != None:
            plt.title(title)

        if param_tuple != None:
            parameters = r'${} = {}$,' * int(len(param_tuple)/2 - 1) + '{} = {}'
            plt.suptitle(parameters.format(*param_tuple), y = 0.965)
        #moving axis
        # Move left y-axis and bottim x-axis to centre, passing through (0,0)
        ax = plt.subplot()
        ax.spines['left'].set_position('center')
        ax.spines['bottom'].set_position('center')
        ax.xaxis.get_ticklabels
        # Eliminate upper and right axes
        ax.spines['right'].set_color('none')
        ax.spines['top'].set_color('none')
        #ylim = ax.get_ylim()
        #xlim = ax.get_xlim()
        #ax.axhline(y = (ylim[0] + ylim[1])/2, linewidth=4, color='k')
        #ax.axvline(x = (xlim[0] + xlim[1])/2, linewidth=4, color='k')
        xticks = ax.xaxis.get_major_ticks()
        xticks[4].label1.set_visible(False)
        #yticks = ax.yaxis.get_major_ticks()
        #yticks[3].update_position(200)
        plt.xticks(fontweight=700)
        plt.yticks(fontweight=700)
    else: 
        plt.rcParams.update({'font.size':20})
        plt.figure(figsize=(12,8), dpi=80)
        if xlabel != None:
            plt.xlabel(r'${}$'.format(xlabel), fontweight=1000)
        if xlabel != None:
            plt.ylabel(r'${}$'.format(ylabel), fontweight = 1000)
        if title != None:
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



#interval_cond = conditions for linspace, for t
#coords = [..., (x_i, y_i), ...] - 
#func - model to be integrated over
#params - parameters for the model

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

    
def pltime_series(traj, t, xlabel, ylabel, fig_name, title = None):
   
    graph_specifications(xlabel = xlabel, ylabel = ylabel, title = title)
    
    plt.plot(t, traj)
    plt.savefig("./NewFigs/TimeSeries/TS{}.pdf".format(fig_name))
    plt.show()
    
    return

def mod(val, p):
    quotient, remainder = np.divmod(val, p)
    rem_odd_div = np.where(np.mod(quotient, 2) == 1)
    remainder[rem_odd_div] -= p
    
    return remainder

def phase_plane_traj(func, coords, params, interval_cond, func_vars = [0, 1], traj_type = 'reg', mod_param = 0):
    start, end, sampling = interval_cond
    t = np.linspace(*interval_cond)

    if traj_type == 'reg':
        print('plotting reg traj')
        for coord in coords:
            traj = odeint(func, coord, t, args=params)[:, func_vars]
            plt.plot(traj[sampling-1000:, 0], traj[sampling-1000:, 1], color='b')
            
    elif traj_type == 'modulo' or traj_type == 'mod':
        print('plotting mod traj')
        for coord in coords:
            traj = odeint(func, coord, t, args=params)[:, func_vars]
            remainder = mod(traj[:, 0], mod_param)
            plt.plot(remainder[sampling-1000:], traj[sampling-1000:, 1], color='b')
            
    return

#vector_field(bounds, func, params, fig_name, coords=(), func_vars = [0,1], \
#                 param_tuple = (), fig_title = '', arrows = True, interval_cond = (0, 5000, 50000), 
#                 traj_type = 'reg', mod_param = 0, **kwargs
#bounds - (start, end, sample) used for generating the neighborhood of vector field arrows
#func - model used to generate vector field(passed into odeint), params - parameters for the model
#coords  - 2d array with rows as sets of initial conditions used to generate trajectories within the phase plane
#func_vars = [i, j] the indices for the variables of the model to be used for phase plane
#i - the variable on the x axis, j variable on the y axis. 
#e.g.: odeint returns = [[\phi1, \dot{\phi1}, x1, \dot{x1}]...], if func_vars = [0,1] then the graph 
#generated is for \phi vs \dot{\phi}
#arrows = True - whether or not to include vector field arrows in the graph
#traj_type - used for drawing trajectories in phase plane. 'reg' if nothing needs to be changed about
#output returned from odeint, 'mod' if the output needs to be modded by some parameter mod_param
def phase_plane(func, params, fig_name, coords, func_vars = [0,1], bounds = (), \
                 param_tuple = (), fig_title = '', arrows = False, interval_cond = (0, 5000, 100000), 
                 traj_type = 'reg', mod_param = 0, xlim = None, ylim = None, xlabel = None, ylabel = None,
                 time_series = False):
    graph_specifications(graph_type = 'phase plane', xlabel = xlabel, ylabel = ylabel, param_tuple = param_tuple, \
                          xlim = xlim, ylim = ylim, title = fig_title)
    if arrows:
    #creating an array of points corresponding to the 2d grid where vector field arrows are going to be graphed
        corner, r, sampling = bounds        
        xy = neighborhood(*bounds)
        t = np.linspace(0, 1, 10)
        phi_integrated_arr = np.empty((10, 2))
    
        vector_coords = np.repeat([coords[0]], xy.shape[0], axis=0)
        vector_coords[:, func_vars] = xy
        
        for phi in vector_coords:
            solution  = odeint(func, phi, t, args=params)[:, func_vars]
            phi_integrated_arr = concat((phi_integrated_arr, solution), axis=1)
        
        #finding the direction for the vectors in forms of differences to plot them with plt.arrow
        sol_arr = phi_integrated_arr[:, 2:]
        difference = sol_arr - np.roll(sol_arr, 1, axis=0)
        sol_arr = sol_arr[0]
        difference = difference[1]
        scaling_factor = r/sampling
        diff_normalized = diff_normalization(difference)*scaling_factor*0.5
        
    
        size = sol_arr.size
        sol = sol_arr.reshape((int(size/2), 2))
        for start, change in zip(sol, diff_normalized):        
            plt.arrow(start[0], start[1], change[0], change[1], width = scaling_factor*0.1, color="#808080")
            
    phase_plane_traj(func, coords, params, interval_cond, func_vars = func_vars, traj_type = traj_type, \
                     mod_param = mod_param)
    
    plt.savefig("./NewFigs/PhasePlane/PhasePlane{}.pdf".format(fig_name))
    plt.show()
    
    if time_series:
        start, end, sample = interval_cond
        trajectories, t = selected_traj(func, params, coords, interval_cond)
        if traj_type == 'mod':
            traj1 = mod(trajectories[sample-100:sample, func_vars[0]], mod_param)
        else:
            traj1 = trajectories[sample-100:sample, func_vars[0]]
        t = t[sample-100:sample]
        pltime_series(traj1, t, 't', xlabel, fig_name)
    
    return 'vector_field'

#generates all the phase plane graphs along with time series for every parameter in 
#np.linspace(param_linspace) for the Pedestrian_Bridge Model
#The images are saved in the followin way
#All phase planes are saved in ./NewFigs/Phase/PhasePlane/ with the name of the form
#PhasePlane_PhiDotPhi_all_parameter_specifications
#param_index = (i,j) is a tuple where i is a index for the parameter in param_tuple array
#and j is the index for the parameter in the params array.
#params and param_tuple are passed as arrays and are converted into tuples within the method code
def bridge_paramVar(param_linspace, param_index, params, coords, param_tuple):
    param_tuple_index, params_index = param_index
    param_arr = np.linspace(*param_linspace)

    for param in param_arr:
        params[params_index] = param
        param_tuple[param_tuple_index] = param
        fig_name = "_PhiDotPhi_" + "_{}"*len(param_tuple)
        fig_name = fig_name.format(*tuple(param_tuple))
        phase_plane(pedestrian_bridge, tuple(params), fig_name, coords, func_vars=[0,1], param_tuple=param_tuple, \
                 traj_type = 'mod', mod_param = params[-1], arrows = False, xlabel = '\phi', ylabel = '\dot{\phi}', time_series = True)
        fig_name = "_PhiX_" + "_{}"*len(param_tuple)
        fig_name = fig_name.format(*tuple(param_tuple))
        phase_plane(pedestrian_bridge, tuple(params), fig_name, coords, func_vars=[0,2], param_tuple=param_tuple, \
                 traj_type = 'mod', mod_param = params[-1], arrows = False, xlabel = '\phi', ylabel = 'x', time_series = False)
        fig_name = "_XDotX_" + "_{}"*len(param_tuple)
        fig_name = fig_name.format(*tuple(param_tuple))
        phase_plane(pedestrian_bridge, tuple(params), fig_name, coords, func_vars=[2,3], param_tuple=param_tuple, \
                 traj_type = 'reg', mod_param = params[-1], arrows = False, xlabel = 'x', ylabel = '\dot{x}', time_series = True)
        
    return 
#vector_field(bounds, func, params, fig_name, coords, func_vars = [0,1], \
#                 param_tuple = (), fig_title = '', arrows = True, interval_cond = (0, 5000, 50000), 
#                 traj_type = 'reg', mod_param = 0, **kwargs)
#params=  (mu, w, g, l, lambda, a, k, h, p)

params = [0.1, 0.7, 10, 1, 0.4666666666, 1, 2, 0.002, 0.3]
bounds = ((-0.4, -0.4), 0.5, 100)
coords = [[0.1, 0.1, 0, 0]]
param_tuple = ['mu', 0.1, 'w', 0.7, 'lambda', 0.466666, 'a', 1, 'k', 2, 'h', 0.002, 'p', 0.3]
fig_name = "_PhiDotPhi_" + "_{}"*len(param_tuple)
fig_name = fig_name.format(*tuple(param_tuple))

#phase_plane(pedestrian_bridge, tuple(params), fig_name, coords, func_vars=[0,1], param_tuple = param_tuple, \
#            traj_type = 'mod', mod_param = params[-1], arrows = False, xlabel = '\phi', ylabel='\dot{\phi}')
#p
#p_linspace = (0.1, 1, 10)
#p_index = (13, 8)
#bridge_paramVar(p_linspace, p_index, params, coords, param_tuple)

#h
#params = [0.1, 0.7, 10, 1, 1, 1, 1, 0.002, 0.3]
#h_linspace = (0, 0.0035, 10)
#h_index = (11, 7)
#bridge_paramVar(h_linspace, h_index, params, coords, param_tuple)
##a
#params = [0.1, 0.7, 10, 1, 1, 1, 1, 0.002, 0.3]
#a_linspace = (0.2, 0.4, 8)
#a_index = (7, 5)
#bridge_paramVar(a_linspace, a_index, params, coords, param_tuple)
##k
#params = [0.1, 0.7, 10, 1, 1, 1, 1, 0.002, 0.3]
#k_linspace = (0.1, 2, 10)
#k_index = (9, 6)
#bridge_paramVar(k_linspace, k_index, params, coords, param_tuple)
##lambda
#params = [0.1, 0.7, 10, 1, 1, 1, 1, 0.002, 0.3]
#lmbda_linspace = (0.2, 1, 10)
#lmbda_index = (5, 4)
#bridge_paramVar(lmbda_linspace, lmbda_index, params, coords, param_tuple)
#w


#traj1 = trajectories[99900:100000, 0]
#traj1 = mod(traj1, 0.3)
#t = t[99900:100000]
#time_series(traj1, t, 't', '\phi', 'PHITimeSeries')
    
