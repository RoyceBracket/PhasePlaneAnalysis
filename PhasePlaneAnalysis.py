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
    
def neighborhood(center, radius, sampling):
    x, y = center
    distance = radius/sampling
    
    x_arr = []
    y_arr = []
    for i in range(sampling):
        x_arr += [x + i*distance]
        y_arr += [y + i*distance]
        
    xy_matrix = [[]]
    for x in x_arr:
        for y in y_arr:
            xy_matrix += [[0.1, 0.1, x, y]]
        
    return xy_matrix
    
def many_trajectories(w, h, a):
    xy = neighborhood((-6, -6), 12, 120)
    t = np.linspace(0, 1000, 10000)
    
    phi_integrated_arr = np.empty((10000,2))

    for phi in xy[1:]:
        #print(phi)
        solution = odeint(vvmodel, phi, t, args = params)
        phi_integrated_arr = np.concatenate((phi_integrated_arr, solution), axis=1)
        
    sol_arr = phi_integrated_arr[:, 2:]
    
    plt.rcParams.update({'font.size': 26})
    plt.figure(figsize=(12,8), dpi=80)
    plt.xlim(-10, 10)
    plt.ylim(-4,4)
    plt.xlabel('phi')
    plt.ylabel('phi_dot')
    print(phi_integrated_arr)
    
    row, columns = sol_arr.shape
    print(columns/2)
    for i in range(int(columns/2)):
        plt.plot(sol_arr[:, 2*i], sol_arr[:, 2*i+1])
        
    plt.savefig("./PhasePortrait2.pdf")
    plt.show()

#difference - a 1d array consisting of length N, where columns are paired
#as [x1,y1,x2,y2,x3,y3,...]. 
#returns scaled normalized difference to make the phase plane diagram legible
def diff_normalization(difference):
    size = difference.size
    diff_2d = difference.reshape((int(size/2), 2))
    row_norms = np.array([np.linalg.norm(row) for row in diff_2d])
    diff_normalized = diff_2d/(18*row_norms[:, None])
    
    return diff_normalized


#creates and saves the plot of phase plane vector field with few selected
#trajectories in the current directory, based on the model used in odeint(...)
#params - parameters for model we are researching
    


def selected_traj(func, params, coords):
    t2 = np.linspace(0, 100, 1000)
    t3 = np.linspace(0, -100, 1000)
    trajectories = ()
    
    for xy in coords: 
        trajectories += odeint(func, xy, t2, args=(params))
    
    return trajectories

#ax - axis for plotting
#bounds = {(x, y), radius, sampling} - for the phase plane window, used for neighborhood
#coords = [..., (x_i, y_i), ...] - 
#func - model to be integrated over
#params - parameters for the model
    
def time_series(func, params, fig_name):
    p = params[-1]
    t = np.linspace(0, 100, 10000)
    phi_x = [-0.5, 2, 0.1, 0.1]
    sol = odeint(func, phi_x, t, args = params)
    x = sol[:, 2]
    """
    phi = sol[:, 0]
    
    neg_phi = np.where(phi<-1)
    pos_phi = np.where(phi>1)
    phi[neg_phi] = np.mod(phi[neg_phi], p)
    phi[pos_phi] = np.mod(phi[pos_phi], -p)
    """
    plt.rcParams.update({'font.size':20})
    plt.figure(figsize=(12,8), dpi=80)
    
    plt.xlabel("time")
    plt.ylabel(r"$x$")
    
    print(sol[:, 0])
    plt.plot(t, x)
    plt.savefig("./{}.pdf".format(fig_name))
    plt.show()
    
    return

def vector_field(bounds, coords, fig_name, func, params):
    #xy = neighborhood((-pi, -pi), 8, 40)
    corner, radius, sampling = bounds
    p = params[-1]
    
    
    xy = neighborhood(corner, radius, sampling)
    t = np.linspace(0, 1, 10)
    phi_integrated_arr = np.empty((10, 2))
    for phi in xy[1:]:
        solution  = odeint(func, phi, t, args=params)[:, 2:]
        phi_integrated_arr = concat((phi_integrated_arr, solution), axis=1)
        
    sol_arr = phi_integrated_arr[:, 2:]
    difference = sol_arr - np.roll(sol_arr, 1, axis=0)
    sol_arr = sol_arr[0]
    difference = difference[1]
    diff_normalized = diff_normalization(difference)
    
    
    
    plt.rcParams.update({'font.size':20})
    plt.figure(figsize=(12,8), dpi=80)
    #plt.xlim(-pi/2, pi/2)
    plt.xlim(-p, p)
    plt.ylim(-3,3)
    plt.xlabel(r'$x$', labelpad=200, fontweight=1000)
    plt.ylabel(r'$\dot{x}$', labelpad = 310, fontweight = 1000)
    plt.title("Phase Diagram for Pedestrian-Bridge System")
    plt.suptitle(r"w = 1, $\lambda = 0.5$, a = 0.2", y = 0.965)
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
    
    size = sol_arr.size
    sol = sol_arr.reshape((int(size/2), 2))
    
    for start, change in zip(sol, diff_normalized):
        plt.arrow(start[0], start[1], change[0], change[1], width = 0.01, color="#808080")
        
    t2 = np.linspace(0, 100, 10000)
    phi_x = [0.1, 0.1, 0.75, 0]
    traj = odeint(func, phi_x, t2, args=params)
    
    plt.plot(traj[:,2], traj[:, 3], color='b')
    
    """
    #\phi, \dot{\phi} trajectory
    t2 = np.linspace(0, 10, 10000)
    phi_x = [-0.5, -2, 0, 0]
    traj = odeint(func, phi_x, t2, args = params)    
    
    phi = traj[:, 0]
    neg_phi = np.where(phi<-1)
    pos_phi = np.where(phi>1)
    phi[neg_phi] = np.mod(phi[neg_phi], p)
    phi[pos_phi] = np.mod(phi[pos_phi], -p)
    
    plt.plot(phi, traj[:, 1], color='b')                           
    """
    #for traj in selected_traj(func, params, coords):
    #    plt.plot(traj[:,0], traj[:, 1], color='b')
        
    #plt.plot(lc2[:, 0], lc2[:, 1], color='b')
    #plt.plot(utraj_npi[:,0], utraj_npi[:,1], color='b')
    #plt.plot(utraj_pi[:,0], utraj_pi[:,1], color='b')
    #plt.plot(ssaddle1[:,0], ssaddle1[:,1], color='b')
    #plt.plot(ssaddle2[:,0], ssaddle2[:,1], color='b')
    #plt.plot(usaddle1[:,0], usaddle1[:,1], color='b')
    #plt.plot(usaddle2[:,0], usaddle2[:,1], color='b')
    
              
    plt.savefig("./{}.pdf".format(fig_name))
    plt.show()
    
bounds = ((-3, -3), 6, 60)
#coords_sp1 = (())
#coords_sp2 = 
coords_bp = ()
#params: (mu, w, g, l, lambda, a, k, h, p)
params = (0.0001, 1, 10, 1, 0.5, 0.2, 1, 0.05, 1)
fig_name = "PhaseDiagramPBX"

vector_field(bounds, coords_bp, fig_name, pedestrian_bridge, params)

time_series(pedestrian_bridge, params, "XTimeSeries1")


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