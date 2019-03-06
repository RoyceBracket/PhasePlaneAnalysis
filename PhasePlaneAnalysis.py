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


#vvmodel(phi, t, w, h, a)

    
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
            xy_matrix += [[x,y]]
        
    return xy_matrix
    
def many_trajectories(w, h, a):
    xy = neighborhood((-6, -6), 12, 120)
    t = np.linspace(0, 1000, 10000)
    
    phi_integrated_arr = np.empty((10000,2))

    for phi in xy[1:]:
        #print(phi)
        solution = odeint(vvmodel, phi, t, args = (w,h,a))
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
    diff_normalized = diff_2d/(10*row_norms[:, None])
    
    return diff_normalized


#creates and saves the plot of phase plane vector field with few selected
#trajectories in the current directory, based on the model used in odeint(...)
#(w,h,a) - parameters for model we are researching
def vector_field(w, h, a):
    xy = neighborhood((-pi, -pi), 8, 40)
    t = np.linspace(0, 1, 10)
    phi_integrated_arr = np.empty((10, 2))
    for phi in xy[1:]:
        solution  = odeint(vvmodel, phi, t, args = (w,h,a))
        phi_integrated_arr = concat((phi_integrated_arr, solution), axis=1)
        
    sol_arr = phi_integrated_arr[:, 2:]
    difference = sol_arr - np.roll(sol_arr, 1, axis=0)
    sol_arr = sol_arr[0]
    difference = difference[1]
    diff_normalized = diff_normalization(difference)
    
    #generating data for selected trajectories
    t2 = np.linspace(0, 100, 1000)
    t3 = np.linspace(0, -100, 1000)
    phi_lc1 = (-pi/2 + 0.05, 2.5)
    lc1 = odeint(vvmodel, phi_lc1, t2, args=(w,h,a))
    
    phi_lc2 = (pi/2 - 0.05, -2.5)
    lc2 = odeint(vvmodel, phi_lc2, t2, args=(w,h,a))
    """
    phi_npi = (-pi+0.1, 0.05)
    utraj_npi = odeint(vvmodel, phi_npi, t2, args=(w,h,a))
    
    phi_pi = (pi+0.1, 0.05)
    utraj_pi = odeint(vvmodel, phi_pi, t2, args=(w,h,a))
    """
    phi_ssaddle1 = (-0.025,0.05)
    ssaddle1 = odeint(vvmodel, phi_ssaddle1, t3, args=(w,h,a))
    
    phi_ssaddle2 = (0.025, -0.05)
    ssaddle2 = odeint(vvmodel, phi_ssaddle2, t3, args=(w,h,a))
    
    phi_usaddle1 = (0.025, 0.05)
    usaddle1 = odeint(vvmodel, phi_usaddle1, t2, args=(w,h,a))
    
    phi_usaddle2 = (-0.025, -0.05)
    usaddle2 = odeint(vvmodel, phi_usaddle2, t2, args=(w,h,a))
    
    plt.rcParams.update({'font.size':20})
    plt.figure(figsize=(12,8), dpi=80)
    plt.xlim(-pi/2, pi/2)
    plt.ylim(-3,3)
    plt.xlabel(r'$\phi$', labelpad=200, fontweight=1000)
    plt.ylabel(r'$\dot{\phi}$', labelpad = 310, fontweight = 1000)
    plt.title("Phase Diagram for a Single Pedestrian")
    plt.suptitle(r"w = 1, $\lambda = 1$, a = 1", y = 0.965)
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
        plt.arrow(start[0], start[1], change[0], change[1], width=0.025, color="#808080")
        
                  
    plt.plot(lc1[:,0], lc1[:, 1], color='b')
    plt.plot(lc2[:, 0], lc2[:, 1], color='b')
    #plt.plot(utraj_npi[:,0], utraj_npi[:,1], color='b')
    #plt.plot(utraj_pi[:,0], utraj_pi[:,1], color='b')
    plt.plot(ssaddle1[:,0], ssaddle1[:,1], color='b')
    plt.plot(ssaddle2[:,0], ssaddle2[:,1], color='b')
    plt.plot(usaddle1[:,0], usaddle1[:,1], color='b')
    plt.plot(usaddle2[:,0], usaddle2[:,1], color='b')
    
              
    plt.savefig("./PhasePortraitVV_4.pdf")
    plt.show()
    
vector_field(1,1,1)