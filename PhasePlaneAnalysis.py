# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 07:54:02 2019

@author: rkrylov1
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from numpy import concatenate as concat
from PedestrianBridgeModel import pedestrian_bridge 
import imageio

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
            plt.xlabel(r'${}$'.format(xlabel), fontweight=1000)

        if ylabel != None:
            plt.ylabel(r'${}$'.format(ylabel), fontweight = 1000)

        if title != None:
            plt.title(title)

        if param_tuple != None:
            parameters = r'${} = {}$,' * int(len(param_tuple)/2 - 1) + '{} = {}'
            plt.suptitle(parameters.format(*param_tuple), y = 0.965)
        #moving axis
        # Move left y-axis and bottim x-axis to centre, passing through (0,0)
        ax = plt.subplot()
        #ax.spines['left'].set_position('center')
        #ax.spines['bottom'].set_position('center')
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
        plt.tight_layout(pad=4)
    else: 
        plt.rcParams.update({'font.size':20})
        plt.figure(figsize=(12,8), dpi=80)
        plt.tight_layout(pad=2)

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
    
def data_overParam(func, params, param_index, param_linspace, coord, coord_linspace, data_name = '', \
                   save = True, directory = './'):
    param_arr = np.linspace(*param_linspace)
    t = np.linspace(*coord_linspace)
    traj_comb = []
    
    for param in param_arr:
        params[param_index] = param
        traj = odeint(func, coord, t, args = tuple(params))
        traj[:, 0] = np.mod(traj[:, 0], 4*params[-1])
        traj_comb.append(traj)
        
    if save:
        traj_name = directory + "traj_" + data_name
        params_name = directory + "params_" + data_name
        full_params = [params, param_index, param_linspace, coord, coord_linspace]
        np.save(traj_name, traj_comb)
        np.save(params_name, full_params)
    
    print('finished')
    return traj_comb
#params=  (mu, w, g, l, lambda, a, k, h, p)

#param_tuple = ['mu', 0.1, 'w', 0.7, 'lambda', 1, 'a', 1, 'k', 1, 'h', 0.002, 'p', 0.3]
#param_name = "_{}"*len(param_tuple)
#param_name = param_name.format(*param_tuple)
#print(param_name)
#coord = [0.1, 0.1, 0.75, 0]
##p
#params = [0.1, 0.7, 10, 1, 1, 1, 1, 0.002, 0.3]
#p_linspace = (0.1, 1, 100)
#data_name = "pvar"+ param_name
#data_overParam(pedestrian_bridge, params, 8, p_linspace, coord, (0, 5000, 100000), data_name = data_name, directory = "./Data/NewData/")
##h
#params = [0.1, 0.7, 10, 1, 1, 1, 1, 0.002, 0.3]
#h_linspace = (0, 0.005, 100)
#data_name = "hvar"+ param_name
#data_overParam(pedestrian_bridge, params, 7, h_linspace, coord, (0, 5000, 100000), data_name = data_name, directory = "./Data/NewData/")
##k
#params = [0.1, 0.7, 10, 1, 1, 1, 1, 0.002, 0.3]
#k_linspace = (0.1, 1.5, 100)
#data_name = "kvar"+ param_name
#data_overParam(pedestrian_bridge, params, 6, k_linspace, coord, (0, 5000, 100000), data_name = data_name, directory = "./Data/NewData/")
##mu
#params = [0.1, 0.7, 10, 1, 1, 1, 1, 0.002, 0.3]
#mu_linspace = (0.05, 0.15, 100)
#data_name = "muvar"+ param_name
#data_overParam(pedestrian_bridge, params, 0, mu_linspace, coord, (0, 5000, 100000), data_name = data_name, directory = "./Data/NewData/")
##lambda
#params = [0.1, 0.7, 10, 1, 1, 1, 1, 0.002, 0.3]
#lambda_linspace = (0.1, 1.5, 100)
#data_name = "lambdavar"+ param_name
#data_overParam(pedestrian_bridge, params, 4, lambda_linspace, coord, (0, 5000, 100000), data_name = data_name, directory = "./Data/NewData/")
##a
#params = [0.1, 0.7, 10, 1, 1, 1, 1, 0.002, 0.3]
#a_linspace = (0.1, 1.5, 100)
#data_name = "avar"+ param_name
#data_overParam(pedestrian_bridge, params, 5, a_linspace, coord, (0, 5000, 100000), data_name = data_name, directory = "./Data/NewData/")
#

def selected_traj(func, params, coords, interval_cond, save = True, directory = ''):
    t = np.linspace(*interval_cond)
    params = tuple(params)
    
    try:
        trajectories = odeint(func, coords[0], t, args=params)
    except:
        return (0, 0)
        
    for xy in coords[1:]: 
        trajectories = concat((trajectories, odeint(func, xy, t, args=params)))
        t = np.concatenate((t,t))
    
    return 

    
def pltime_series(traj, t, xlabel, ylabel, fig_name, title = None, tseries_dir = ''):
   
    graph_specifications(xlabel = xlabel, ylabel = ylabel, title = title)
    
    plt.plot(t, traj)
    tseries_name = tseries_dir + "TimeSeries_" + fig_name
    plt.savefig("./{}".format(tseries_name))
    plt.show()
    
    return

def mod(val, p):
    quotient, remainder = np.divmod(val, p)
    #rem_odd_div = np.where(np.mod(quotient, 2) == 1)
    #remainder[rem_odd_div] -= p
    
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

def generate_GIF(name, filenames, directory = "./"):
    with imageio.get_writer(directory + '{}.mp4'.format(name), mode='I', fps=10) as writer:
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)
    print("generated {}".format(name))
    return

#full_params = [params, param_index, param_linspace, coord, coord_linspace]
def pplane_overData(data_name, fig_name, fig_title = '', func_vars = [0,1], param_tuple = (), ptuple_index = None, xlim = None, ylim = None, \
                    xlabel = None, ylabel = None, directory = './', data_dir = "./", animate = False):
    full_params = np.load(data_dir + "params" + data_name)
    trajectories = np.load(data_dir + "traj" + data_name)
    filenames = []
    
    params, param_index, param_linspace, coord, coord_linspace = full_params
    param_arr = np.linspace(*param_linspace)
    start, end, sampling = coord_linspace
    pstart, pend, psampling = param_linspace
    for traj, param in zip(trajectories, param_arr):
        if ptuple_index != None:
            param_tuple[ptuple_index] = param
        graph_specifications(graph_type = "phase plane", xlabel = xlabel, ylabel = ylabel, xlim = xlim, ylim = ylim, \
                         title = fig_title, param_tuple = param_tuple)
        plt.plot(traj[sampling-1000:, func_vars[0]], traj[sampling-1000:, func_vars[1]], color = 'b')
        filename = directory + fig_name + "{}.png".format(param)
        plt.savefig(filename)
        filenames.append(filename)
        plt.close()
            
    np.save(directory + fig_name, filenames)
    if animate:
        generate_GIF(fig_name, filenames, directory = directory + "/gifs/")
    
    return
#
#directory = "./NewFigs/Phase Planes/animations/p_var/"
#param_tuple = ['mu', 0.1, 'w', 0.7, 'lambda', 1, 'a', 1, 'k', 1, 'h', 0.002, 'p', 0.3]
#data_dir = "./Data/NewData/"
#param_name = "_{}"*len(param_tuple)
#param_name = param_name.format(*param_tuple)
#data_name = "_pvar" + param_name + ".npy"
#fig_name = "pplane_pvar"
#filenames = np.load(directory+fig_name+".npy")
#generate_GIF(fig_name+"2", filenames, directory = directory + "/gifs/")

#p
directory = "./NewFigs/Phase Planes/animations/p_var/"
param_tuple = ['mu', 0.1, 'w', 0.7, 'lambda', 1, 'a', 1, 'k', 1, 'h', 0.002, 'p', 0.3]
data_dir = "./Data/NewData/"
param_name = "_{}"*len(param_tuple)
param_name = param_name.format(*param_tuple)
data_name = "_pvar" + param_name + ".npy"
fig_name = "pplane_pvar"
pplane_overData(data_name, fig_name+"PhiDotPhi", param_tuple = param_tuple, ptuple_index = 13, xlabel = '\phi', ylabel = '\dot{\phi}', \
                directory = directory, data_dir = data_dir, animate = True)
pplane_overData(data_name, fig_name+"PHIX", func_vars = [0, 2], param_tuple = param_tuple, ptuple_index = 13, xlabel = '\phi', ylabel = 'x', \
                directory = directory, data_dir = data_dir, animate = True)
pplane_overData(data_name, fig_name+"XDotX", func_vars = [2, 3], param_tuple = param_tuple, ptuple_index = 13, xlabel = 'x', ylabel = '\dot{x}', \
                directory = directory, data_dir = data_dir, animate = True)

#k
directory = "./NewFigs/Phase Planes/animations/k_var/"
param_tuple = ['mu', 0.1, 'w', 0.7, 'lambda', 1, 'a', 1, 'k', 1, 'h', 0.002, 'p', 0.3]
data_name = "_kvar" + param_name + ".npy"
fig_name = "pplane_pvar"
pplane_overData(data_name, fig_name+"PhiDotPhi", param_tuple = param_tuple, ptuple_index = 9, xlabel = '\phi', ylabel = '\dot{\phi}', \
                directory = directory, data_dir = data_dir, animate = True)
pplane_overData(data_name, fig_name+"PHIX", func_vars = [0, 2], param_tuple = param_tuple, ptuple_index = 9, xlabel = '\phi', ylabel = 'x', \
                directory = directory, data_dir = data_dir, animate = True)
pplane_overData(data_name, fig_name+"XDotX", func_vars = [2, 3], param_tuple = param_tuple, ptuple_index = 9, xlabel = 'x', ylabel = '\dot{x}', \
                directory = directory, data_dir = data_dir, animate = True)
#h
directory = "./NewFigs/Phase Planes/animations/h_var/"
param_tuple = ['mu', 0.1, 'w', 0.7, 'lambda', 1, 'a', 1, 'k', 1, 'h', 0.002, 'p', 0.3]
data_name = "_hvar" + param_name + ".npy"
fig_name = "pplane_pvar"
pplane_overData(data_name, fig_name+"PhiDotPhi", param_tuple = param_tuple, ptuple_index = 11, xlabel = '\phi', ylabel = '\dot{\phi}', \
                directory = directory, data_dir = data_dir, animate = True)
pplane_overData(data_name, fig_name+"PHIX", func_vars = [0, 2], param_tuple = param_tuple, ptuple_index = 11, xlabel = '\phi', ylabel = 'x', \
                directory = directory, data_dir = data_dir, animate = True)
pplane_overData(data_name, fig_name+"XDotX", func_vars = [2, 3], param_tuple = param_tuple, ptuple_index = 11, xlabel = 'x', ylabel = '\dot{x}', \
                directory = directory, data_dir = data_dir, animate = True)
#a
directory = "./NewFigs/Phase Planes/animations/a_var/"
param_tuple = ['mu', 0.1, 'w', 0.7, 'lambda', 1, 'a', 1, 'k', 1, 'h', 0.002, 'p', 0.3]
data_name = "_avar" + param_name + ".npy"
fig_name = "pplane_pvar"
pplane_overData(data_name, fig_name+"PhiDotPhi", param_tuple = param_tuple, ptuple_index = 7, xlabel = '\phi', ylabel = '\dot{\phi}', \
                directory = directory, data_dir = data_dir, animate = True)
pplane_overData(data_name, fig_name+"PHIX", func_vars = [0, 2], param_tuple = param_tuple, ptuple_index = 7, xlabel = '\phi', ylabel = 'x', \
                directory = directory, data_dir = data_dir, animate = True)
pplane_overData(data_name, fig_name+"XDotX", func_vars = [2, 3], param_tuple = param_tuple, ptuple_index = 7, xlabel = 'x', ylabel = '\dot{x}', \
                directory = directory, data_dir = data_dir, animate = True)
#lambda
directory = "./NewFigs/Phase Planes/animations/lambda_var/"
param_tuple = ['mu', 0.1, 'w', 0.7, 'lambda', 1, 'a', 1, 'k', 1, 'h', 0.002, 'p', 0.3]
data_name = "_lambdavar" + param_name + ".npy"
fig_name = "pplane_pvar"
pplane_overData(data_name, fig_name+"PhiDotPhi", param_tuple = param_tuple, ptuple_index = 5, xlabel = '\phi', ylabel = '\dot{\phi}', \
                directory = directory, data_dir = data_dir, animate = True)
pplane_overData(data_name, fig_name+"PHIX", func_vars = [0, 2], param_tuple = param_tuple, ptuple_index = 5, xlabel = '\phi', ylabel = 'x', \
                directory = directory, data_dir = data_dir, animate = True)
pplane_overData(data_name, fig_name+"XDotX", func_vars = [2, 3], param_tuple = param_tuple, ptuple_index = 5, xlabel = 'x', ylabel = '\dot{x}', \
                directory = directory, data_dir = data_dir, animate = True)
#mu
directory = "./NewFigs/Phase Planes/animations/mu_var/"
param_tuple = ['mu', 0.1, 'w', 0.7, 'lambda', 1, 'a', 1, 'k', 1, 'h', 0.002, 'p', 0.3]
data_name = "_muvar" + param_name + ".npy"
fig_name = "pplane_pvar"
pplane_overData(data_name, fig_name+"PhiDotPhi", param_tuple = param_tuple, ptuple_index = 1, xlabel = '\phi', ylabel = '\dot{\phi}', \
                directory = directory, data_dir = data_dir, animate = True)
pplane_overData(data_name, fig_name+"PHIX", func_vars = [0, 2], param_tuple = param_tuple, ptuple_index = 1, xlabel = '\phi', ylabel = 'x', \
                directory = directory, data_dir = data_dir, animate = True)
pplane_overData(data_name, fig_name+"XDotX", func_vars = [2, 3], param_tuple = param_tuple, ptuple_index = 1, xlabel = 'x', ylabel = '\dot{x}', \
                directory = directory, data_dir = data_dir, animate = True)


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
                 time_series = False, pplane_dir = '', tseries_dir = ''):
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
    pplane_name = pplane_dir + "PhasePlane_" +  fig_name
    plt.savefig("./{}".format(pplane_name))
    plt.show()
    
    if time_series:
        start, end, sample = interval_cond
        trajectories, t = selected_traj(func, params, coords, interval_cond)
        if traj_type == 'mod':
            traj1 = mod(trajectories[sample-100:sample, func_vars[0]], mod_param)
        else:
            traj1 = trajectories[sample-100:sample, func_vars[0]]
        t = t[sample-100:sample]
        pltime_series(traj1, t, 't', xlabel, fig_name, tseries_dir = tseries_dir)
    
    return 'vector_field'

#generates all the phase plane graphs along with time series for every parameter in 
#np.linspace(param_linspace) for the Pedestrian_Bridge Model
#The images are saved in the followin way
#All phase planes are saved in ./NewFigs/Phase/PhasePlane/ with the name of the form
#PhasePlane_PhiDotPhi_all_parameter_specifications
#param_index = (i,j) is a tuple where i is a index for the parameter in param_tuple array
#and j is the index for the parameter in the params array.
#params and param_tuple are passed as arrays and are converted into tuples within the method code
def bridge_paramVar(param_linspace, param_index, params, coords, param_tuple, pplane_dir = '', tseries_dir = '', combTSeries = False):
    param_tuple_index, params_index = param_index
    param_arr = np.linspace(*param_linspace)

    for param in param_arr:
        params[params_index] = param
        param_tuple[param_tuple_index] = param
        fig_name = "PhiDotPhi" + "_{}"*len(param_tuple) + ".png"
        fig_name = fig_name.format(*tuple(param_tuple))
        phase_plane(pedestrian_bridge, tuple(params), fig_name, coords, func_vars=[0,1], param_tuple=param_tuple, \
                 traj_type = 'mod', mod_param = params[-1], arrows = False, xlabel = '\phi', ylabel = '\dot{\phi}', time_series = True,
                 tseries_dir = tseries_dir, pplane_dir = pplane_dir)
        fig_name = "PhiX" + "_{}"*len(param_tuple) + ".png"
        fig_name = fig_name.format(*tuple(param_tuple))
        phase_plane(pedestrian_bridge, tuple(params), fig_name, coords, func_vars=[0,2], param_tuple=param_tuple, \
                 traj_type = 'mod', mod_param = params[-1], arrows = False, xlabel = '\phi', ylabel = 'x', time_series = False, 
                 tseries_dir = tseries_dir, pplane_dir = pplane_dir)
        fig_name = "XDotX" + "_{}"*len(param_tuple) + ".png"
        fig_name = fig_name.format(*tuple(param_tuple))
        phase_plane(pedestrian_bridge, tuple(params), fig_name, coords, func_vars=[2,3], param_tuple=param_tuple, \
                 traj_type = 'reg', mod_param = params[-1], arrows = False, xlabel = 'x', ylabel = '\dot{x}', time_series = True, 
                 tseries_dir = tseries_dir, pplane_dir = pplane_dir)
        
    if combTSeries: 
        print('combTSeries')
        
        
    return 
#vector_field(bounds, func, params, fig_name, coords, func_vars = [0,1], \
#                 param_tuple = (), fig_title = '', arrows = True, interval_cond = (0, 5000, 50000), 
#                 traj_type = 'reg', mod_param = 0, **kwargs)
#params=  (mu, w, g, l, lambda, a, k, h, p)
pplane_dir = "NewFigs/PhasePlane/"
tseries_dir = "NewFigs/TimeSeries/"

params = [0.1, 0.7, 10, 1, 1, 1, 1, 0.002, 0.3]
bounds = ((-0.4, -0.4), 0.5, 100)
coords = [[0.1, 0.1, 0, 0]]
param_tuple = ['mu', 0.1, 'w', 0.7, 'lambda', 1, 'a', 1, 'k', 1, 'h', 0.002, 'p', 0.3]
fig_name = "_PhiDotPhi_" + "_{}"*len(param_tuple)
fig_name = fig_name.format(*tuple(param_tuple))

#phase_plane(pedestrian_bridge, tuple(params), fig_name, coords, func_vars=[0,1], param_tuple = param_tuple, \
##            traj_type = 'mod', mod_param = params[-1], arrows = False, xlabel = '\phi', ylabel='\dot{\phi}')
##p
#params = [0.1, 0.7, 10, 1, 1, 1, 1, 0.002, 0.3]
#param_tuple = ['mu', 0.1, 'w', 0.7, 'lambda', 1, 'a', 1, 'k', 1, 'h', 0.002, 'p', 0.3]
#pplane_pdir = pplane_dir + "p_var/"
#tseries_pdir = tseries_dir + "p_var/"
#p_linspace = (0.1, 1, 10)
#p_index = (13, 8)
#bridge_paramVar(p_linspace, p_index, params, coords, param_tuple, pplane_dir = pplane_pdir, tseries_dir = tseries_pdir)
##h
#params = [0.1, 0.7, 10, 1, 1, 1, 1, 0.002, 0.3]
#param_tuple = ['mu', 0.1, 'w', 0.7, 'lambda', 1, 'a', 1, 'k', 1, 'h', 0.002, 'p', 0.3]
#pplane_hdir = pplane_dir + "h_var/"
#tseries_hdir = tseries_dir + "h_var/"
#h_linspace = (0, 0.0035, 10)
#h_index = (11, 7)
#bridge_paramVar(h_linspace, h_index, params, coords, param_tuple, pplane_dir = pplane_hdir, tseries_dir = tseries_hdir)
##a
#params = [0.1, 0.7, 10, 1, 1, 1, 1, 0.002, 0.3]
#param_tuple = ['mu', 0.1, 'w', 0.7, 'lambda', 1, 'a', 1, 'k', 1, 'h', 0.002, 'p', 0.3]
#pplane_adir = pplane_dir + "a_var/"
#tseries_adir = tseries_dir + "a_var/"
#a_linspace = (0.2, 0.4, 8)
#a_index = (7, 5)
#bridge_paramVar(a_linspace, a_index, params, coords, param_tuple, pplane_dir = pplane_adir, tseries_dir = tseries_adir)
##k
#params = [0.1, 0.7, 10, 1, 1, 1, 1, 0.002, 0.3]
#param_tuple = ['mu', 0.1, 'w', 0.7, 'lambda', 1, 'a', 1, 'k', 1, 'h', 0.002, 'p', 0.3]
#pplane_kdir = pplane_dir + "k_var/"
#tseries_kdir = tseries_dir + "k_var/"
#k_linspace = (0.1, 2, 10)
#k_index = (9, 6)
#bridge_paramVar(k_linspace, k_index, params, coords, param_tuple, pplane_dir = pplane_kdir, tseries_dir = tseries_kdir)
##lambda
#params = [0.1, 0.7, 10, 1, 1, 1, 1, 0.002, 0.3]
#param_tuple = ['mu', 0.1, 'w', 0.7, 'lambda', 1, 'a', 1, 'k', 1, 'h', 0.002, 'p', 0.3]
#pplane_lmbdadir = pplane_dir + "lambda_var/"
#tseries_lmbdadir = tseries_dir + "lambda_var/"
#lmbda_linspace = (0.2, 1, 10)
#lmbda_index = (5, 4)
#bridge_paramVar(lmbda_linspace, lmbda_index, params, coords, param_tuple, pplane_dir = pplane_lmbdadir, tseries_dir = tseries_lmbdadir)
##mu
#params = [0.1, 0.7, 10, 1, 1, 1, 1, 0.002, 0.3]
#param_tuple = ['mu', 0.1, 'w', 0.7, 'lambda', 1, 'a', 1, 'k', 1, 'h', 0.002, 'p', 0.3]
#pplane_mudir = pplane_dir + "mu_var/"
#tseries_mudir = tseries_dir + "mu_var/"
#mu_linspace = (0.07, 0.13, 10)
#mu_index = (1, 0)
#bridge_paramVar(mu_linspace, mu_index, params, coords, param_tuple, pplane_dir = pplane_mudir, tseries_dir = tseries_mudir)
#w


#traj1 = trajectories[99900:100000, 0]
#traj1 = mod(traj1, 0.3)
#t = t[99900:100000]
#time_series(traj1, t, 't', '\phi', 'PHITimeSeries')
    
