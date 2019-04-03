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
def vector_field(bounds, func, params, fig_name, coords, func_vars = [0,1], \
                 param_tuple = (), fig_title = '', arrows = True, interval_cond = (0, 5000, 100000), 
                 traj_type = 'reg', mod_param = 0, xlim = None, ylim = None, xlabel = None, ylabel = None,
                 time_series = False):
    corner, r, sampling = bounds        
    graph_specifications(graph_type = 'phase plane', xlabel = xlabel, ylabel = ylabel, param_tuple = param_tuple, \
                          xlim = xlim, ylim = ylim, title = fig_title)
    if arrows:
    #creating an array of points corresponding to the 2d grid where vector field arrows are going to be graphed
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
        t = t[sample-100:sample]
        pltime_series(traj1, t, 't', xlabel, fig_name)
    
    return 'vector_field'
    
#vector_field(bounds, func, params, fig_name, coords, func_vars = [0,1], \
#                 param_tuple = (), fig_title = '', arrows = True, interval_cond = (0, 5000, 50000), 
#                 traj_type = 'reg', mod_param = 0, **kwargs)

params = [0.1, 0.7, 10, 1, 1, 1, 1, 0.05, 0.3]
bounds = ((-0.4, -0.4), 0.5, 100)
coords = [[0.1, 0.1, 0, 0]]
param_tuple = ('\mu', 0.1, 'w', 0.7, '\lambda', 1, 'a', 1, 'k', 1, 'h', 0.01)
#varying mu:
mu_arr = np.linspace(0.05, 0.2, 10)
for mu in mu_arr:
    fig_name = "_PhiDotPhi_mu_{}_w_0.7_lambda_1_a_1_k_1_h_0.05_p_0.3".format(mu)
    param_tuple = ('\mu', mu, 'w', 0.7, '\lambda', 1, 'a', 1, 'k', 1, 'h', 0.05)
    params[0] = mu
    vector_field(bounds, pedestrian_bridge, tuple(params), fig_name, coords, func_vars=[0,1], param_tuple=param_tuple, \
                 traj_type = 'mod', mod_param = params[-1], arrows = False, xlabel = '\phi', ylabel = '\dot{\phi}', time_series = True)
    fig_name = "_PhiX_mu_{}_w_0.7_lambda_1_a_1_k_1_h_0.05_p_0.3".format(mu)
    vector_field(bounds, pedestrian_bridge, tuple(params), fig_name, coords, func_vars=[0,2], param_tuple=param_tuple, \
                 traj_type = 'mod', mod_param = params[-1], arrows = False, xlabel = '\phi', ylabel = 'x')
    fig_name = "_XDotX_mu_{}_w_0.7_lambda_1_a_1_k_1_h_0.05_p_0.3".format(mu)
    vector_field(bounds, pedestrian_bridge, tuple(params), fig_name, coords, func_vars=[2,3], param_tuple=param_tuple, \
                 traj_type = 'mod', mod_param = params[-1], arrows = False, xlabel = 'x', ylabel = '\dot{x}', time_series = True)
params = [0.1, 0.7, 10, 1, 1, 1, 1, 0.05, 0.3]

h_arr = np.linspace(0, 0.002, 10)
for h in h_arr:
    fig_name = "_PhiDotPhi_mu_0.1_w_0.7_lambda_1_a_1_k_1_h_{}_p_0.3".format(h)
    param_tuple = ('\mu', 0.1, 'w', 0.7, '\lambda', 1, 'a', 1, 'k', 1, 'h', h)

    params[7] = h
    vector_field(bounds, pedestrian_bridge, tuple(params), fig_name, coords, func_vars=[0,1], param_tuple=param_tuple, \
                 traj_type = 'mod', mod_param = params[-1], arrows = False, xlabel = '\phi', ylabel = '\dot{\phi}', time_series = True)
    fig_name = "_PhiX_mu_0.1_w_0.7_lambda_1_a_1_k_1_h_{}_p_0.3".format(h)
    vector_field(bounds, pedestrian_bridge, tuple(params), fig_name, coords, func_vars=[0,2], param_tuple=param_tuple, \
                 traj_type = 'mod', mod_param = params[-1], arrows = False, xlabel = '\phi', ylabel = 'x')
    fig_name = "_XDotX_mu_0.1_w_0.7_lambda_1_a_1_k_1_h_{}_p_0.3".format(h)
    vector_field(bounds, pedestrian_bridge, tuple(params), fig_name, coords, func_vars=[2,3], param_tuple=param_tuple, \
                 traj_type = 'mod', mod_param = params[-1], arrows = False, xlabel = 'x', ylabel = '\dot{x}', time_series = True)

params = [0.1, 0.7, 10, 1, 1, 1, 1, 0.05, 0.3]
lambda_arr = np.linspace(0.5, 0.6, 5)
for lmbda in lambda_arr:
    fig_name = "_PhiDotPhi_mu_0.1_w_0.7_lambda_{}_a_1_k_1_h_0.05_p_0.3".format(lmbda)
    param_tuple = ('\mu', 0.1, 'w', 0.7, '\lambda', lmbda, 'a', 1, 'k', 1, 'h', 0.05)
    params[4] = lmbda
    vector_field(bounds, pedestrian_bridge, tuple(params), fig_name, coords, func_vars=[0,1], param_tuple=param_tuple, \
                 traj_type = 'mod', mod_param = params[-1], arrows = False, xlabel = '\phi', ylabel = '\dot{\phi}', time_series = True)
    fig_name = "_PhiX_mu_0.1_w_0.7_lambda_1_a_1_k_1_h_0.05_p_0.3".format(lmbda)
    vector_field(bounds, pedestrian_bridge, tuple(params), fig_name, coords, func_vars=[0,2], param_tuple=param_tuple, \
                 traj_type = 'mod', mod_param = params[-1], arrows = False, xlabel = '\phi', ylabel = 'x')
    fig_name = "_XDotX_mu_0.1_w_0.7_lambda_1_a_1_k_1_h_0.05_p_0.3".format(lmbda)
    vector_field(bounds, pedestrian_bridge, tuple(params), fig_name, coords, func_vars=[2,3], param_tuple=param_tuple, \
                 traj_type = 'mod', mod_param = params[-1], arrows = False, xlabel = 'x', ylabel = '\dot{x}', time_series = True)

params = [0.1, 0.7, 10, 1, 1, 1, 1, 0.05, 0.3]
a_arr = np.linspace(0.2, 0.35, 10)
for a in a_arr:
    fig_name = "_PhiDotPhi_mu_0.1_w_0.7_lambda_1_a_{}_k_1_h_0.05_p_0.3".format(a)
    param_tuple = ('\mu', 0.1, 'w', 0.7, '\lambda', 1, 'a', a, 'k', 1, 'h', 0.05)
    params[5] = a
    vector_field(bounds, pedestrian_bridge, tuple(params), fig_name, coords, func_vars=[0,1], param_tuple=param_tuple, \
                 traj_type = 'mod', mod_param = params[-1], arrows = False, xlabel = '\phi', ylabel = '\dot{\phi}', time_series = True)
    fig_name = "_PhiX_mu_0.1_w_0.7_lambda_1_a_{}_k_1_h_0.05_p_0.3".format(a)
    vector_field(bounds, pedestrian_bridge, tuple(params), fig_name, coords, func_vars=[0,2], param_tuple=param_tuple, \
                 traj_type = 'mod', mod_param = params[-1], arrows = False, xlabel = '\phi', ylabel = 'x')
    fig_name = "_XDotX_mu_0.1_w_0.7_lambda_1_a_{}_k_1_h_0.05_p_0.3".format(a)
    vector_field(bounds, pedestrian_bridge, tuple(params), fig_name, coords, func_vars=[2,3], param_tuple=param_tuple, \
                 traj_type = 'mod', mod_param = params[-1], arrows = False, xlabel = 'x', ylabel = '\dot{x}', time_series = True)
    



#traj1 = trajectories[99900:100000, 0]
#traj1 = mod(traj1, 0.3)
#t = t[99900:100000]
#time_series(traj1, t, 't', '\phi', 'PHITimeSeries')
    
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