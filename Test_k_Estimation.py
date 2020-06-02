# -*- coding: utf-8 -*-
"""
Created on Sun May  3 19:57:52 2020

@author: MATTEK10

This is a script to estimation of (k) active sources from real EEG data.

This script need the following libraries to run:
    - Numpy
    - main 
    - data
    - simulated_data
    - plot_functions
    
The script use the:
    - data library to import the data which has been segmented into time 
    segments of one second
    - main library to perform the recovering process of X given Y and a 
    random mixing matrix A
    - simulated_data library to calculated the MSE of the recovered X
    - plot_functions to plot the sources of X from a given time segment
"""
# =============================================================================
# Import libraries
# =============================================================================
from Main_Algorithm import Main_Algorithm
import Data_EEG
import Data_Simulation
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1234)
# =============================================================================
# Import EEG data file
# =============================================================================
data_name = 'S1_CClean.mat'
#data_name = 'S1_OClean.mat'
data_file = 'data/' + data_name            # file path

segment_time = 1                           # length of segments i seconds

# =============================================================================
# Main Algorithm with fixed A
# =============================================================================
" Import Segmented Dataset "
request='remove 1/2' # remove sensors and the same sources from dataset
#request='remove 1/3' # remove sensors and the same sources from dataset
#request = 'none'
Y, M, L, n_seg = Data_EEG._import(data_file, segment_time, request=request)

k = np.ones([len(Y)]) * 27
A_result, X_result = Main_Algorithm(Y, M, k, L, data = 'EEG', fix = True)

" Searching for replicates  "
list_ = np.zeros([len(Y), len(X_result[0])])
for seg in range(len(Y)):
    for i in range(len(X_result[0])):
        rep = 0
        for j in range(len(X_result[0])):
            mse = Data_Simulation.MSE_one_error(X_result[seg][i], X_result[seg][j])
            if  mse < 1.0:
                rep += 1
        list_[seg][i] = rep

" Plots of segment seg = 9 -- the 10 th segment"
seg = 9
figsave = "figures/EEG_second_removed_est_k" + str(data_name) + '_' + str(seg) + ".png"

def plot_seperate_sources(X_reconstruction,M,N,k,L,figsave,nr):
    plt.figure(nr)
    plt.title('M = {}, N = {}, k = {}, L = {}'.format(M,N,k,L))
    nr_plot=0
    for i in range(N):
        if np.any(X_reconstruction[i]!=0):
            nr_plot += 1
            plt.subplot(8, 1, nr_plot)
           
            plt.plot(X_reconstruction[i],'g', label='Recovered X')
        if nr_plot == 8:
            break
    plt.legend()
    plt.xlabel('sample') 
    plt.show()
    plt.savefig(figsave)


plot_seperate_sources(X_result[seg], M, int(k[seg]), int(k[seg]), L, figsave,1)
