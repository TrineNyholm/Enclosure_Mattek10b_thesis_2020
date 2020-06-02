# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 10:36:46 2020

@author: Mattek10b

This script contain the functions needed to perform Cov-DL. It consist of:
    
    - _dictionarylearning
    - _inversevectorization
    - _vectorization
    - _A
    - _covdomain
    - Cov_DL1
    - Cov_DL2
    
The script need the sklearn.decomposition, scipy.optimize and NumPy libraries.
"""
# =============================================================================
# Libraries and Packages
# =============================================================================
from sklearn.decomposition import DictionaryLearning
from sklearn.decomposition import PCA
import numpy as np

# =============================================================================
# Functions
# =============================================================================
def _dictionarylearning(Y, N, k, iter_=100000):
    """
    Perform dictionary learning on the given data set.
    --------------------------------------------------
    Input:
        Y: Measurement matrix of size M x L
        N: Number of sources
        k: Number of active sources
    Output:
        D: A dictionary matrix
    """
    dct = DictionaryLearning(n_components=N, transform_algorithm='lars',
                             transform_n_nonzero_coefs=k, max_iter=iter_)
    dct.fit(Y.T)
    D = dct.components_
    return D.T

def _vectorization(Y, M):
    """
    Perform vectorization of a matrix.
    ----------------------------------
    Input:
        Y: A matrix of size M x L
        M: Number of sensors 
    Output:
        vec_Y: A vector of size M(M+1)/2
    """
    vec_Y = Y[np.tril_indices(M)]
    return vec_Y

def _inversevectorization(d):
    """
    Perform the devectorization of covariances.
    -------------------------------------------
    Input:  
        d: A vector of size M(M+1)/2
    Output: 
        D: A matrix of size M x M
    """
    M = int(np.sqrt(d.size*2))
    if (M*(M+1))//2 != d.size:
        print("Inverse vectorization fail")
        return None
    else:
        R, C = np.tril_indices(M)
        D = np.zeros((M, M), dtype=d.dtype)
        D[R, C] = d
        D[C, R] = d
    return D

def _A(D, M, N):
    """
    Determine the mixing matrix A from the dictionary D.
    ----------------------------------------------------
    Input:
        D: A dictionary matrix of size M x N
        M: Number of sensors
        N: Number of sources
    Output:
        A_rec: A recovered mixing matrix of size M x N
    """
    A_rec = np.zeros(([M, N]))
    for j in range(N):
        d = D.T[j]
        matrix_d = _inversevectorization(d)
        E = np.linalg.eig(matrix_d)[0]
        V = np.linalg.eig(matrix_d)[1]
        max_eig = np.max(np.abs(E))     
        index = np.where(np.abs(E) == max_eig)
        max_vec = V[:, index[0]]        # using the column as eigenvector here
        temp = np.sqrt(max_eig)*max_vec
        A_rec.T[j] = temp.T
    return A_rec

def _covdomain(Y, L, L_covseg, M):
    """
    Perform the transformation to the covariance-domain.
    ----------------------------------------------------
    Input:
        Y: Measurement matrix of size M x L
        L: Number of samples
        L_covseg: Number of samples in one segment
        M: Number of sensors
    """
    n_seg = int(L/L_covseg)               # number of segments
    Y = Y.T[:n_seg*L_covseg].T            # remove last segment if to small
    Ys = np.split(Y, n_seg, axis=1)       # list of all segments in axis 0
    Y_big = np.zeros([int(M*(M+1)/2.), n_seg])
    for j in range(n_seg):                # loop over all segments
        Y_cov = np.cov(Ys[j])             
        Y_big.T[j] = _vectorization(Y_cov, M)
    return Y_big

def Cov_DL1(Y_big, M, N, k):
    """
    Perform Cov-DL1 on the data set.
    --------------------------------
    Input:
        Y_big: Measurement matrix Y in covariance domain of size M(M+1)/2 
        M: Number of sensors
        N: Number of sources
        k: Number of active sources
    Output:
        A_rec: Recovered mixing matrix in time domain of size M x N
    """
    print('Using Cov-DL1')
    D = _dictionarylearning(Y_big, N, k) 
    A_rec = _A(D, M, N)   # Determined A from dictionary matrix D
    print('Estimation of A is done')
    return A_rec

def Cov_DL2(Y_big, M, N, k):
    """
    Perform Cov-DL2 on the data set.
    --------------------------------
    Input:
        Y_big: Measurement matrix Y in covariance domain of size M(M+1)/2 
        M: Number of sensors
        N: Number of sources
        k: Number of active sources
        A_real: True mixing matrix for known data sets
    Output:
        A_rec: Recovered mixing matrix in time domain of size M x N
        A_ini: Initial mixing matrix for optimization
    """
    print('Using Cov_DL2 \n')
    
    " Performing PCA of measurement matrix Y in covariance domain "
    pca = PCA(n_components=N, svd_solver='randomized', whiten=True)
    pca.fit(Y_big.T)
    U = pca.components_.T
    
    " Initial mixing matrix A for optimization "
    A_ini = np.random.randn(M, N)               # Gaussian initial A
    a = np.reshape(A_ini, (A_ini.size),order='F')   # Normal vectorization of initial A
    
    " Minimization of optimization problem "
    def D_(a):
        D = np.zeros((int(M*(M+1)/2), N))
        for i in range(N):
            A_tilde = np.outer(a[M*i:M*i+M], a[M*i:M*i+M].T)
            D.T[i] = A_tilde[np.tril_indices(M)]
        return D

    def D_term(a):
        return np.dot(np.dot(D_(a), (np.linalg.inv(np.dot(D_(a).T, D_(a))))),
                      D_(a).T)

    def U_term():
        return np.dot(np.dot(U, (np.linalg.inv(np.dot(U.T, U)))), U.T)

    def cost1(a):
        return np.linalg.norm(D_term(a)-U_term())**2
    
    # predefined optimization method.
    from scipy.optimize import minimize
    res = minimize(cost1, a, method='BFGS',# BFGS, Nelder-Mead
                  options={'maxiter': 10000, 'disp': True})
    a_new = res.x
    A_rec = np.reshape(a_new, (M, N), order='F')
    
    print('\nCost(A_init) = {}'.format(np.round_(cost1(a), decimals=4)))
    print('Cost(A_estimte) = {}'.format(np.round_(cost1(a_new), decimals=4)))
#    print('Cost(A_true) = {}'.format(np.round_(cost1(np.reshape(A_real, (A_real.size),order='F')),decimals=4)))
    
    return A_rec, A_ini
