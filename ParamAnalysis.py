# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 15:03:19 2019

@author: rkrylov1
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from numpy import pi
from numpy import concatenate as concat
from SinglePedestrianModel import vvmodel
from PedestrianBridgeModel import pedestrian_bridge 

def parameter_analysis(func, params, index, start, end, sampling, figname, param_dictionary): 
    varied_param = np.linspace(start, end, 100)
    max_x = []
    phi_x = [0.1, 0.1, 0.75, 0]
    if index == 0:
        M = 5650
        mu = np.sqrt(varied_param/(M + varied_param))    
        print(mu)
        varied_param = mu
    
    t = np.linspace(0, 5000, 10000)

    for param in varied_param:
        print(param)
        params[index] = param
        params_tuple = tuple(params)
        x = odeint(func, phi_x, t, args = params_tuple)[8000:, 2]
        max_x += [np.amax(x)]
        
    plt.rcParams.update({'font.size':20})
    plt.figure(figsize=(12,8), dpi=80)
    plt.xlabel(r"${}$".format(param_dictionary[index]))
    plt.ylabel(r"max(x)")
    
    plt.plot(varied_param, max_x)
    
    plt.savefig("./NewFigs/{}.pdf".format(figname))
    plt.show()
    
    np.save("./Data/{}_{}".format(figname, sampling), max_x)
    
    return

#params: (mu, w, g, l, lambda, a, k, h, p)

params = [0.1, 0.7, 10, 1, 1, 1, 1, 0.05, 0.3]
sampling = 60
param_dicitonary = ('\mu', 'w', 'g', 'l', '\lambda', 'a', 'k', 'h', 'p')

parameter_analysis(pedestrian_bridge, params, 1, 1.1, 3.4, sampling, "MAX_X_mu_0.1_w_var1.1_3.4_lambda_1_a_1_k_1_h_0.05_p_0.3", param_dicitonary)
parameter_analysis(pedestrian_bridge, params, 5, 0.25, 0.35, sampling, "MAX_X_mu_0.1_w_0.7_lambda_1_a_var0.25_0.35_k_1_h_0.05_p_0.3", param_dicitonary)
parameter_analysis(pedestrian_bridge, params, 5, 1, 3, sampling, "MAX_X_mu_0.1_w_0.7_lambda_1_a_var1_3_k_1_h_0.05_p_0.3", param_dicitonary)
parameter_analysis(pedestrian_bridge, params, 6, 0, 0.3, sampling, "MAX_X_mu_0.1_w_0.7_lambda_1_a_1_k_var0_0.2_h_0.05_p_0.3", param_dicitonary)
parameter_analysis(pedestrian_bridge, params, 7, 0, 0.004, sampling, "MAX_X_mu_0.1_w_0.7_lambda_1_a_1_k_1_h_var0_0.004_p_0.3", param_dicitonary)
parameter_analysis(pedestrian_bridge, params, 8, 0.2, 0.25, sampling, "MAX_X_mu_0.1_w_0.7_lambda_1_a_1_k_1_h_0.05_p_var0.2_0.25", param_dicitonary)

params = [0.1, 0.7, 10, 1, 1, 1, 1, 0.001, 0.3]
sampling = 200

parameter_analysis(pedestrian_bridge, params, 8, 0.1, 0.7, sampling, "MAX_X_mu_0.1_w_0.7_lambda_1_a_1_k_1_h_0.001_p_var0.1_0.7", param_dicitonary)
parameter_analysis(pedestrian_bridge, params, 6, 0.1, 2, sampling, "MAX_X_mu_0.1_w_0.7_lambda_1_a_1_k_1_h_0.001_p_var0.1_0.7", param_dicitonary)
parameter_analysis(pedestrian_bridge, params, 5, 0.1, 2, sampling, "MAX_X_mu_0.1_w_0.7_lambda_1_a_1_k_1_h_0.001_p_var0.1_0.7", param_dicitonary)





