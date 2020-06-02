# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 10:36:46 2020

@author: Mattek10b

This script contain the functions needed to perform M-SBL. It consist of:
    - M_SBL
The script need the NumPy library.
"""
# =============================================================================
# Libraries and Packages
# =============================================================================
import numpy as np

# =============================================================================
# Functions
# =============================================================================
def M_SBL(A, Y, M, N, k, iterations, noise):
    """
    Perform M-SBL of a given non-segmented data set.
    ------------------------------------------------
    Input:
        A: Mixing matrix of size M x N
        Y: Measurement matrix of size M x L
        M: Number of sensors
        N: Number of sources
        k: Number of active sources
        iterations: Number of iterations
        noise: Either True or False
    Output:
        X_rec: recovered sources signals of size N x L-2
        
    """
    L = Y.shape[1]
    tol = 0.0001
    if noise is False:
        Y = Y.T[:-2]
        Y = Y.T
        gamma = np.ones([iterations+2, N, 1])   
        Gamma = np.ones([iterations+1, N, N])
        mean = np.ones([iterations+1, N, L-2])
        Sigma = np.zeros([iterations+1, N, N])
        _iter = 1
        while _iter < 3 or any((gamma[_iter]-gamma[_iter-1]) > tol):
            Gamma[_iter] = np.diag(np.reshape(gamma[_iter], (N)))        # size 1 x 1
            
            " Making Sigma and Mu "
            Sigma[_iter] = np.dot((np.identity(N) - np.linalg.multi_dot(
                    [np.sqrt(Gamma[_iter]), np.linalg.pinv(
                            np.dot(A, np.sqrt(Gamma[_iter]))), A])), Gamma[_iter])
            mean[_iter] = np.linalg.multi_dot(
                    [np.sqrt(Gamma[_iter]), np.linalg.pinv(np.dot(A,
                                                    np.sqrt(Gamma[_iter]))), Y])
            for i in range(N):
                " Update gamma with EM and with M being Fixed-Point"
                gam_num = 1/L * np.linalg.norm(mean[_iter][i])
                gam_den = 1 - ((1/(gamma[_iter][i])) * Sigma[_iter][i][i])
                gamma[_iter + 1][i] = gam_num/gam_den
            if _iter == iterations:
                break
            _iter += 1

    elif noise is True:
        Y = Y.T[:-2]
        Y = Y.T
        gamma = np.ones([iterations+2, N, 1])
        Gamma = np.ones([iterations+1, N, N])
        mean = np.ones([iterations+1, N, L-2])
        Sigma = np.zeros([iterations+1, N, N])
        lam = np.ones([iterations+2, 1])     # size N x 1
        _iter = 1
        while _iter < 3 or any((gamma[_iter] - gamma[_iter - 1]) > tol):
            Gamma[_iter] = np.diag(np.reshape(gamma[_iter], (N)))
            
            " Making Sigma and Mu "
            sig = lam[_iter] * np.identity(M) +  np.linalg.multi_dot([A, Gamma[_iter], A.T])
            inv = np.linalg.pinv(sig)
            Sigma[_iter] = (Gamma[_iter]) - (np.linalg.multi_dot([Gamma[_iter], A.T, inv, A, Gamma[_iter]]))
            mean[_iter] = np.linalg.multi_dot([Gamma[_iter], A.T, inv, Y])

            " Making the noise variance/trade-off parameter lambda of p(Y|X)"
            lam_num = 1/L * np.linalg.norm(Y - A.dot(mean[_iter]), ord='fro')**2  # numerator
            lam_for = 0                         
            for j in range(N):
                lam_for += Sigma[_iter][j][j] / gamma[_iter][j]
                
            lam_den = M - N + lam_for                                             # denominator
            lam[_iter + 1] = lam_num / lam_den
                
            for i in range(N):
                " Update gamma with EM and with M being Fixed-Point"
                gam_num = 1/L * np.linalg.norm(mean[_iter][i])
                gam_den = 1 - ((1/(gamma[_iter][i])) * Sigma[_iter][i][i])
                gamma[_iter + 1][i] = gam_num/gam_den

                if _iter == iterations:
                    break
                _iter += 1

    " Finding the support set "
    support = np.zeros(k)
    H = gamma[_iter - 1]
    for l in range(k):
        if H[np.argmax(H)] != 0:
            support[l] = np.argmax(H)
            H[np.argmax(H)] = 0

    " Create new mean with support set "
    X_rec = np.zeros([N, L - 2])
    for i in support:
        X_rec[int(i)] = mean[_iter - 1][int(i)]
    return X_rec
