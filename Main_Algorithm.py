# -*- coding: utf-8 -*-
"""
Created on Fri May  8 09:41:05 2020

@author: Mattek10b

This script consists the complete main algorithm which calls the necessary
functions from the modules imported in this script. By setting fix = True
the module Cov-DL is overwritten by a fixed mixing matrix A. The script 
consists of:
    - Main_Algorithm
"""
import M_SBL
import Cov_DL
import numpy as np
np.random.seed(1234)

def Main_Algorithm(Y, M, k, L, data = 'EEG', fix = True): 
    """
    Perform the Main Algorithm of a given data set. Either simulated or the 
    real EEG data and for either fixed or Cov-DL mixing matrix
    -----------------------------------------------------------------------
    Input:
        Y: Measurement matrix of size M x L
        A: Mixing matrix of size M x N
        M: Number of sensors
        k: Number of active sources
        L: Number of samples
        data: Either 'EEG' or 'simulated'
        fix: Either True or False
    Output:
        X_result: The recovered source matrix
    """    
    if data == 'EEG':
        """
        Recover of the source matrix with a fixed mixing matrix A on a
        segmented data set.
        """
        if fix == True:
            " Perform Main Algorithm on Segmented Dataset "
            X_result = []    # Original recovered source matrix X
    
            for i in range(k.shape[0]):
                " Making the right size of X for all segments "
                X_result.append(np.zeros([len(Y), int(k[i])]))
            
            " Original Recovered Source Matrix X, MSE and Average MSE with X_ica "
            for seg in range(len(Y)): 
                # Looking at one time segment
                A = np.random.normal(0, 2, (M, int(k[seg])))
                X_result[seg] = M_SBL.M_SBL(A, Y[seg], M, int(k[seg]), int(k[seg]), iterations=1000, noise=False)
    
        return A, X_result
    
    if data == 'simulated':
        """
        Recover of the source matrix with a either a fixed mixing matrix A 
        or with mixing matrix from Cov-DL. This ask for the true number of
        k and N.
        """
        print('Data information:\n number of sensors \t \t M = {} \n number of samples pr segment \t L = {}'.format(M, Y[0].shape[1]))
    
        N = int(input("Please enter N: "))               # number of sources
        k = int(input("Please enter k: "))               # active sources to be found

        if fix == True:
            A = np.random.normal(0, 2, (M, N))
            X_result = M_SBL.M_SBL(A, Y, M, N, k, iterations=1000, noise=False)
            return A, X_result
        
        if fix == False:
            Ls = 10
            n_seg = int(L/Ls)
            A_result = np.zeros((n_seg, M, N))
            X_result = np.zeros((n_seg, N, L-2))
            for i in range(len(Y)):           
                Y_big = Cov_DL._covdomain(Y[i], L, Ls, M) # transformation to covariance-domain
                
                if N <= (M*(M+1))/2.:
                    A_rec, A_init = Cov_DL.Cov_DL2(Y_big, M, N, k)
                    A_result[i] = A_rec
        
                elif k <= (M*(M+1))/2.:
                    A_rec = Cov_DL.Cov_DL1(Y_big, M, N, k)
                    A_result[i] = A_rec
        
                elif k > (M*(M+1))/2.:
                    raise SystemExit('X is not sparse enogh (k > (m*(m+1))/2)')
            
                X_rec = M_SBL.M_SBL(A_rec, Y[i], M, N, k, iterations=1000, noise=False)
                X_result[i] = X_rec
            return A_result, X_result
