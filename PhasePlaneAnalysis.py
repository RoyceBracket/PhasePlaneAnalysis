# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 07:54:02 2019

@author: rkrylov1
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from numpy import sin
from numpy import cos
from numpy import pi
from numpy import concatenate as concat
from numpy.random import randint 
from SinglePedestrianModel import vvmodel


#phi= {phi, phi_dot}

p = pi/2


    
def plot(phi): 
    plt.rcParams.update({'font.size': 26})
    plt.figure(figsize=(12,8), dpi=80)
    plt.xlim(-10, 10)
    plt.ylim(-4,4)
    plt.xlabel('phi')
    plt.ylabel('phi_dot')
    plt.plot(phi[:, 0], phi[:, 1])
    #plt.savefig("./{}.pdf".format(filename))
    plt.show()
    
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
    
def diff_normalization(difference):
    size = difference.size
    diff_2d = difference.reshape((int(size/2), 2))
    row_norms = np.array([np.linalg.norm(row) for row in diff_2d])
    diff_normalized = diff_2d/(6*row_norms[:, None])
    
    return diff_normalized
    
def vector_field(w, h, a):
    xy = neighborhood((-8, -8), 17, 60)
    t = np.linspace(0, 1, 10)
    phi_integrated_arr = np.empty((10, 2))
    for phi in xy[1:]:
        solution  = odeint(vvmodel, phi, t, args = (w,h,a))
        phi_integrated_arr = np.concatenate((phi_integrated_arr, solution), axis=1)
        
    sol_arr = phi_integrated_arr[:, 2:]
    difference = sol_arr - np.roll(sol_arr, 1, axis=0)
    sol_arr = sol_arr[0]
    difference = difference[1]
    diff_normalized = diff_normalization(difference)
    
    
    plt.rcParams.update({'font.size':26})
    plt.figure(figsize=(12,8), dpi=80)
    plt.xlim(-8,8)
    plt.ylim(-3,3)
    plt.xlabel('phi')
    plt.ylabel('phi_dot')
    
    size = sol_arr.size
    sol = sol_arr.reshape((int(size/2), 2))
    
    for start, change in zip(sol, diff_normalized):
        plt.arrow(start[0], start[1], change[0], change[1], width=0.025, color="#808080")
        
    plt.savefig("./Test.pdf")
    plt.show()
    
vector_field(1,1,1)