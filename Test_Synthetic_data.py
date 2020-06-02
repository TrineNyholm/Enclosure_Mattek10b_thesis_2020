# -*- coding: utf-8 -*-
"""
Created on Fri May  8 09:30:49 2020

@author: Mattek10b

This script perform the test on the synhetic data sets.

This script need the module:
    - Main_Algorithm
    - Data_Simulation
"""
from Main_Algorithm import Main_Algorithm
import Data_Simulation
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(12345) 
# =============================================================================
# Simulated Data Set
# =============================================================================
data = 'Mix'   # choose between 'Mix' or 'Ran'

if data == 'Mix':
    branch = 'Cov-DL1'  # choose between 'Cov-DL1', 'Cov-DL2' and 'Fix'
    if branch == 'Cov-DL1':
        M = 3        # Number of sensors
        L = 1000     # Number of samples
        k = 4        # Number of active source
        N = 8        # Number of source
#        n_seg = 1    # Number of segment - n_seg = 1 --> whole data set
        
        " Construct Data Set "
        Y, A_real, X_real = Data_Simulation.mix_signals(L, M, version=1, duration=4)
        Y = np.reshape(Y, (1,Y.shape[0], Y.shape[1]))
#        X_real = np.reshape(X_real,(1, X_real.shape[0],X_real.shape[1]))
#        X_real = X_real.T[0:L-2].T        
        " Recover A and X "
        A_result, X_result = Main_Algorithm(Y, M, k, L, data = 'simulated', fix = False)
    
    if branch == 'Cov-DL2':
        M = 3        # Number of sensors
        L = 1000     # Number of samples
        k = 4        # Number of active source
        N = 5        # Number of source
#        n_seg = 1    # Number of segment - n_seg = 1 --> whole data set
            
        " Construct Data Set "
        Y, A_real, X_real = Data_Simulation.mix_signals(L, M, version=None, duration=4)
        Y = np.reshape(Y, (1,Y.shape[0], Y.shape[1]))
        X_real = np.reshape(X_real,(1, X_real.shape[0],X_real.shape[1]))
        X_real = X_real.T[0:L-2].T
        " Recover A and X "
        A_result, X_result = Main_Algorithm(Y, M, k, L, data = 'simulated', fix = False)

    if branch == 'Fix':
        M = 3        # Number of sensors
        L = 1000     # Number of samples
        k = 4        # Number of active source
        N = 5        # Number of source
#        n_seg = 1    # Number of segment - n_seg = 1 --> whole data set
            
        " Construct Data Set "
        Y, A_real, X_real = Data_Simulation.mix_signals(L, M, version=None, duration=4)
        Y = np.reshape(Y, (1,Y.shape[0], Y.shape[1]))
        X_real = np.reshape(X_real,(1, X_real.shape[0],X_real.shape[1]))
        X_real = X_real.T[0:L-2].T
        " Recover A and X "
        A_result, X_result = Main_Algorithm(Y, M, k, L, data = 'simulated', fix = True)






#
#Y, A_real, X_real = generate_AR(N, M, L, non_zero)
#
#Y = np.reshape(Y, (1,Y.shape[0], Y.shape[1]))
X_real = np.reshape(X_real,(1, X_real.shape[0],X_real.shape[1]))
X_real = X_real.T[0:L-2].T

A_mse = Data_Simulation.MSE_one_error(A_real,A_result[0])
A_mse0 = Data_Simulation.MSE_one_error(A_real,np.zeros(A_real.shape))
mse_array, mse_avg = Data_Simulation.MSE_segments(X_result[0], X_real[0])

print('\nMSE_A = {}'.format(np.round_(A_mse,decimals=3)))
print('MSE_A_0 = {}'.format(np.round_(A_mse0,decimals=3)))
print('MSE_X = {}'.format(np.round_(mse_avg,decimals=3)))





