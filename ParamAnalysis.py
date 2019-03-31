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

def parameter_analysis(func, params, index, start, end, figname): 
    varied_param = np.linspace(start, end, 60)
    max_x = []
    phi_x = [0.1, 0.1, 0.75, 0]
    """
    M = 5650
    mu = np.sqrt(varied_param/(M + varied_param))    
    print(mu)
    """
    t = np.linspace(0, 5000, 10000)

    for param in varied_param:
        print(param)
        params[index] = param
        params_tuple = tuple(params)
        x = odeint(func, phi_x, t, args = params_tuple)[8000:, 2]
        max_x += [np.amax(x)]
        
    plt.rcParams.update({'font.size':20})
    plt.figure(figsize=(12,8), dpi=80)
    plt.xlabel(r"$k$")
    plt.ylabel(r"max(x)")
    
    plt.plot(varied_param, max_x)
    
    plt.savefig("./NewFigs/{}.pdf".format(figname))
    plt.show()
    
    np.save("./Data/{}_60".format(figname), max_x)
    
    return

#params: (mu, w, g, l, lambda, a, k, h, p)

params = [0.1, 0.7, 10, 1, 1, 1, 1, 0.05, 0.3]

parameter_analysis(pedestrian_bridge, params, 4, 0.5, 1.5, "MAX_X2_mu_01_w_07_lambda_varied_a_1_k_1_h_005_p_03")
parameter_analysis(pedestrian_bridge, params, 5, 0.5, 1.5, "MAX_X2_mu_01_w_07_lambda_1_a_varied_k_1_h_005_p_03")
parameter_analysis(pedestrian_bridge, params, 6, 0.5, 1.5, "MAX_X2_mu_01_w_07_lambda_1_a_1_k_varied_h_005_p_03")
parameter_analysis(pedestrian_bridge, params, 7, 0, 0.2, "MAX_X2_mu_01_w_07_lambda_1_a_1_k_1_h_varied_p_03")
parameter_analysis(pedestrian_bridge, params, 8, 0.1, 0.7, "MAX_X2_mu_01_w_07_lambda_1_a_1_k_1_h_005_p_varied")
parameter_analysis(pedestrian_bridge, params, 1, 0.4, 1, "MAX_X2_mu_01_w_varied_lambda_1_a_1_k_1_h_005_p_03")




