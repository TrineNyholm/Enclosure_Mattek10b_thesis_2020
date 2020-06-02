# -*- coding: utf-8 -*-
"""
Created on Fri Mar 20 09:55:56 2020

@author: Mattek10b

This script contain the functions needed to perform the preprocessing of 
real EEG data sets. The data sets are:
    - S1_CClean.mat
    - S1_OClean.mat
    - S3_CClean.mat
    - S3_OClean.mat
    - S4_CClean.mat
    - S4_OClean.mat
The functions are:
    - fit_Y
    - _import
    - _reduction
    
The script need the scipy.io and NumPy libraries.
"""
# =============================================================================
# Libraries and Packages
# =============================================================================
import numpy as np
import scipy.io

# =============================================================================
# Functions
# =============================================================================
def fit_Y(Y, data_name):
    """
    Fit the data sets such that the rows have same lenght.
    ------------------------------------------------------
    Input: 
        Y: A measurement matrix from a EEG data set
        data_name: The name of the chosen EEG data set
    Output:
        Y: The fitted measurement matrix of size M x L
    """
    
    if data_name == 'S1_CClean.mat':
        # For S1_CClean.mat remove last sample of first segment 
        print(Y[0].shape)
        Y[0] = Y[0].T[:-1]
        Y[0] = Y[0].T
        print(Y[0].shape)

    if data_name == 'S1_OClean.mat':
        # For S1_OClean.mat removed the last sample of the first 22 segments
        for i in range(len(Y)):
            if i <= 22:
                Y[i] = Y[i].T[:-1]
                Y[i] = Y[i].T
            else:
                continue

    if data_name == 'S3_CClean.mat':
        # For S3_CClean.mat removed the last sample of the first 12 segments
        for i in range(len(Y)):
            if i <= 12:
                Y[i] = Y[i].T[:-1]
                Y[i] = Y[i].T
            else:
                continue 
    
    if data_name == 'S3_OClean.mat':
        # For S3_OClean.mat removed the last sample of the first 139 segments
        for i in range(len(Y)):
            if i <= 139:
                Y[i] = Y[i].T[:-1]
                Y[i] = Y[i].T
            else:
                continue 

    if data_name == 'S4_CClean.mat':
        # For S4_CClean.mat removed the last sample of the first 63 segments
        for i in range(len(Y)):
            if i <= 63:
                Y[i] = Y[i].T[:-1]
                Y[i] = Y[i].T
            else:
                continue

    if data_name == 'S4_OClean.mat':
        # For S4_OClean.mat removed the last sample of the first 178 segments
        for i in range(len(Y)):
            if i <= 178:
                Y[i] = Y[i].T[:-1]
                Y[i] = Y[i].T
            else:
                continue
    return Y

def _import(data_file, segment_time, request='none', fs=512):
    """
    Import datafile and perform segmentation.
    ----------------------------------------------------------
    Input:      
        data_file: A string with file path
        segment_time: The length of segment in seconds

    Output:     
        Y: The measurement matrix of size M x L
        Ys: The measurement matrix of size n_seg x M x Ls
        M: Number of sensors
        Ls: Number of samples in one segment
    """
    mat = scipy.io.loadmat(data_file) # import mat file
    Y = np.array([mat['EEG']['data'][0][0][i] for i in
                  range(len(mat['EEG']['data'][0][0]))])
    if request != 'none':
        Y = _reduction(Y, request)
    M, L = Y.shape
    
    " No segmentation if segment_time = 0 "
    if segment_time == 0:
        n_seg = 1
        Y = np.reshape(Y,(1,Y.shape[0],Y.shape[1]))
        return Y, M, L, n_seg
 
    " Segmentation if segment_time > 0 "
    Ls = int(fs * segment_time)             # number of samples in one segment
    if Ls > L:
        raise SystemExit('segment_time is to high')
    n_seg = int(L/Ls)                       # total number of segments
    Y = Y[:n_seg*Ls]                        # remove last segement if too small
    Ys = np.array_split(Y, n_seg, axis=1)   # Matrices with segments in axis=0

    M, Ls = Ys[0].shape

    return Ys, M, Ls, n_seg

def _reduction(Y, request='remove 1/2'):
    """
    Remove a number of sensors (reducing M) corresponding to pre defined
    request. Function used inside data_import().
    ---------------------------------------------------------------------
    Input:      
        Y: Measurement matrix of size M x L
        request:  'remove 1/2' -> remove every second sensor
                  'remove 1/3' -> remove every third sensor
                  'remove 2'   -> remove sensor of index 4 and 8
                

    Output:     
        Y_new: Reduced measurement matrix of size M_new x L
    """
    if request == 'remove 1/2':
        Y_new = np.delete(Y, np.arange(0, Y.shape[0], 2), axis=0)
        return Y_new

    if request == 'remove 1/3':
        Y_new = np.delete(Y, np.arange(0, Y.shape[0], 3), axis=0)
        return Y_new

    if request == 'remove 2':
        Y_new = np.delete(Y, [0, 1], axis=0)
        return Y_new

    else:
        raise SystemExit('Data removeal request is not possible')

    