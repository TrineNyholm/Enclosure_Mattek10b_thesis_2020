# -*- coding: utf-8 -*-
"""
Created on Wed Mar 25 10:43:00 2020

@author: Mattek10b

This script contain the functions needed to construct the simulated data sets
and the measure of performance.
It consist of:
    - MSE_all_errors
    - MSE_one_error
    - MSE_segments
    - mix_signals
    - generate_AR
    
The script need the sklearn.metrics, SciPy and NumPy libraries.
"""
# =============================================================================
# Libraries and Packages
# =============================================================================
import numpy as np
from sklearn.metrics import mean_squared_error
from scipy import signal
 
# =============================================================================
# Performance Measure
# =============================================================================
def MSE_all_errors(real, estimate):
    """
    Mean Squared Error (MSE) - M or N errors

    A small value -- close to zero -- is a good estimation.
    The inputs must be transponet to perform the action rowwise
    ----------------------------------------------------------------------
    Input:
        real: observation matrix
        estimate: calculated matrix
    
    Output: 
        error: The calculated MSE
    """
    error = mean_squared_error(real.T, estimate.T, multioutput='raw_values')
    return error

def MSE_one_error(real,estimate):
    """
    Mean Squared Error (MSE) - One error

    A small value -- close to zero -- is a good estimation.
    The inputs must be transponet to perform the action rowwise
    ----------------------------------------------------------------------
    Input:
        real: observation matrix
        estimate: calculated matrix
    
    Output: 
        error: The calculated MSE
    """
    error = mean_squared_error(real.T, estimate.T)
    return error

def MSE_segments(estimate, real):
    """
    Calculate one MSE error or M/N MSE error for each segment.
    ----------------------------------------------------------
    Input:
        real: observation matrix
        estimate: calculated matrix
    
    Output: 
        mse_rows: The calculated MSE for N/M errors for each segment
        mse_segment: The calculated MSE for one error for each segment
    """
    mse_segment = MSE_one_error(real, estimate)
    mse_rows = MSE_all_errors(real, estimate)
    return mse_rows, mse_segment

# =============================================================================
# Data Simulation
# =============================================================================
def mix_signals(n_samples, M, version=None, duration=4):
    """ 
    Generation of 4 independent signals, united in X with manuel zero rows in 
    between for sparsity. 
    Generation of random mixing matrix A and corresponding Y, such that Y = AX
    ---------------------------------------------------------------------------
    Input:
        n_samples: Number of samples
        M: Number of sensors
        version:
            M=3
            version test --> N=3, k=3
            version none --> N=5, k=4
            version 0    --> N=4, k=4
            version 1    --> N=8, k=4    --> Cov-DL1 
                    
        duration: The lenght of signal in seconds. Fixed to 4 seconds
    Output:
        Y: Measurement matrix of size M X L
        A: Mixing matrix of size M x N
        X: Source matrix of size N x L
    """
    time = np.linspace(0, duration, n_samples)  # list of time index 
    
    s1 = np.sin(2 * time)                       # sinusoidal
    s2 = np.sign(np.sin(3 * time))              # square signal
    s3 = signal.sawtooth(2 * np.pi * time)      # saw tooth signal
    s4 = np.sin(4 * time)                       # different sinusoidal
    
    zero_row = np.zeros(n_samples)
    X = np.c_[s1, zero_row, s3, s4, s2].T       # version 'none'
    
    if version == 'test':
        X = np.c_[s1, s2, s3].T
    if version == 0:
        X = np.c_[s1, s3, s4, s2].T
    if version == 1:
        X = np.c_[zero_row, s1, zero_row, s3, zero_row, zero_row, s4, s2].T
    
    " Finding A and Y "    
    N = len(X)
    A = np.random.randn(M,N)   # Random mixing matrix
    Y = np.dot(X.T, A.T)       # Measurement matrix
    return Y.T, A, X

def generate_AR(N, M, L, non_zero):
    """ 
    Generation of 4 independent autoregressive signals, united in X with 
    manuel zero rows in between for sparsity. 
    Generation of random mixing matrix A and corresponding Y, such that Y = AX
    ---------------------------------------------------------------------------
    Input:
        N: Number of sources
        L: Number of samples       
    Output:
        Y: Measurement matrix of size M x L
        A: Mixing matrix of size M x N
        X: Source matrix of size N x L 
    """
    X = np.zeros([N, L+2])
    
    for i in range(N):
        ind = np.random.randint(1,4)  # Choose randomly between the 4 AR processes
        for j in range(2,L):
            if ind == 1:
                sig = np.random.uniform(-1,1,(2))
                X[i][j] = sig[0] * X[i][j-1] + sig[1] * X[i][j-2] + np.random.randn(1)
            
            elif ind == 2: 
                sig = np.random.uniform(-1,1,(3))
                X[i][j] = sig[0] * X[i][j-1] + sig[1] * X[i][j-2] + sig[2] * X[i][j-3] + np.random.randn(1)
                
            elif ind == 3:
                sig = np.random.uniform(-1,1,(2))
                X[i][j] = sig[0] * X[i][j-1] + sig[1] * X[i][j-2] + np.random.randn(1)
            
            elif ind == 4:
                sig = np.random.uniform(-1,1,(4))
                X[i][j] = sig[0] * X[i][j-1] + sig[1] * X[i][j-2] + sig[2] * X[i][j-3] + sig[3] * X[i][j-4]+ np.random.randn(1)
                
    " Making zero and non-zero rows "
    Real_X = np.zeros([N, L+2])
    ind = np.random.random(non_zero)
    for i in range(len(ind)):
        temp = np.random.randint(0,N)
        while temp in ind:
            temp = np.random.randint(0,N)
        ind[i] = temp
    
    for j in ind:
        Real_X[int(j)] = X[int(j)]
    
    Real_X = Real_X.T[2:].T  

    " Finding A and Y "
    A_Real = np.random.randn(M,N)   # Random mixing matrix
    Y_Real = np.dot(A_Real, Real_X) # Measurement matrix   
    return Y_Real, A_Real, Real_X

def segmentation_split(Y, X, Ls, n_sampels):
    """
    Segmentation of simulated data by spliting the data into segments of 
    length Ls. The last segment is removed if too small. 
    --------------------------------------------------------------------
    Input:
        Y: Measurement matrix of size M x L
        X: Source matrix of size N x L
        Ls: Number of samples in one segment
        n_samples: Number of samples
    Output:
        Ys: Segmented measurement matrix of size n_seg x M x Ls) 
        Xs: Segmented source matrix of size n_seg x N x Ls) 
        n_seg: Number of segments
    """ 
    n_seg = int(n_sampels/Ls)               # Number of segments
    X = X.T[:n_seg*Ls]                      # remove last segement if too small
    Y = Y.T[:n_seg*Ls]
    
    Ys = np.split(Y.T, n_seg, axis=1)        # Matrices with segments in axis=0
    Xs = np.split(X.T, n_seg, axis=1)        # Matrices with segments in axis=0
    
    return Ys, Xs, n_seg