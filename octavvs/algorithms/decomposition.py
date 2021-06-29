#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 18 14:45:53 2021

@author: carl
"""

import numpy as np
import scipy
import sklearn

def simplisma(d, nr, f):
    """
    The SIMPLISMA algorithm for finding a set of 'pure' spectra to serve
    as starting point for MCR-ALS etc.
    Reference Matlab Code:
        J. Jaumot, R. Gargallo, A. de Juan, R. Tauler,
        Chemometrics and Intelligent Laboratoty Systems, 76 (2005) 101-110

    Parameters
    ----------
    d : array(nspectra, nwavenums)
        input spectra.
    nr : int
        number of output components.
    f : float
        noise threshold.

    Returns
    -------
    spout: array(nr, nspectra)
        concentration profiles of 'purest' spectra.
    imp : array(nr, dtype=int)
        indexes of the 'purest' spectra.
    """

    nrow = d.shape[0]
    ncol = d.shape[1]
    s = d.std(axis=0)
    m = d.mean(axis=0)
    mf = m + m.max() * f
    p = s / mf

    # First Pure Spectral/Concentration profile
    imp = np.empty(nr, dtype=np.int)
    imp[0] = p.argmax()

    #Calculation of correlation matrix
    l2 = s**2 + mf**2
    dl = d / np.sqrt(l2)
    c = (dl.T @ dl) / nrow

    #calculation of the first weight
    w = (s**2 + m**2) / l2
    p *= w
    #calculation of following weights
    dm = np.zeros((nr+1, nr+1))
    for i in range(1, nr):
        dm[1:i+1,1:i+1] = c[imp[:i],:][:,imp[:i]]
        for j in range(ncol):
            dm[0,0] = c[j,j]
            dm[0,1:i+1]=c[j,imp[:i]]
            dm[1:i+1,0]=c[imp[:i],j]
            w[j] = np.linalg.det(dm[0:i+1, 0:i+1])

        imp[i] = (p * w).argmax()

    ss = d[:,imp]
    spout = ss / np.sqrt(np.sum(ss**2, axis=0))
    return spout.T, imp

# import pymcr


# def pymcr_als(sp, initial_components, maxiters, reltol,
#             callback_iter):

#     mcr = pymcr.mcr.McrAR(max_iter=maxiters,
#                           tol_err_change=reltol,
#                           tol_increase=1., tol_n_increase=10,
#                           tol_n_above_min=30)
#     mcr.fit(sp, ST=initial_components, post_iter_fcn=callback_iter)

#     return mcr.n_iter, mcr.C_opt_.T, mcr.ST_opt_, np.asarray(mcr.err)



def mcr_als(sp, initial_components, maxiters, tol_rel_error,
            tol_ups_after_best=None, callback=None):
    """
    Perform MCR-ALS nonnegative matrix decomposition on the matrix sp

    Parameters
    ----------
    sp : array(nsamples, nfeatures)
        Spectra to be decomposed.
    initial_components : array(ncomponents, nfeatures)
        Initial concentrations.
    maxiters : int
        Maximum number of iterations.
    tol_rel_error : float
        Error target (relative to first iteration).
    tol_ups_after_best : float, optional
        Stop after error going net up this many times since best error.
    callback : func(it : int, err : array(float), A : array, B : array)
        Callback for every half-iteration.

    Returns
    -------
    iterations : int
    Number of iterations performed
    A : array(ncomponents, nfeatures)
    Spectra (at lowest error)
    B : array(ncomponents, nsamples)
    Concentrations at lowest error
    error : list(float)
    Errors at all half-iterations
    """
    nrow, ncol = np.shape(sp)
    nr = initial_components.shape[0]
    # u, s, v = np.linalg.svd(sp)
    # s = scipy.linalg.diagsvd(s, nrow, ncol)
    # u = u[:, :nr]
    # s = s[:nr, :nr]
    # v = v[:nr, :]
    # dauxt = sklearn.preprocessing.normalize((u @ s @ v).T)
    # dauxt = sp.T
    A = initial_components.T.copy()
    B = np.empty((nr, nrow))
    errors = []
    errorbest = None # Avoid spurious warning

    for it in range(maxiters * 2):
        if it & 1:
            for i in range(ncol):
                A[i, :] = scipy.optimize.nnls(B.T, sp[:, i])[0]
        else:
            for i in range(nrow):
                B[:, i] = scipy.optimize.nnls(A, sp[i, :])[0]

        error = sklearn.metrics.mean_squared_error(sp.T, np.dot(A, B))
        errors.append(error)
        if not it or error < errorbest:
            errorbest = error
            Abest = A.copy()
            Bbest = B.copy()
            netups = 0
        elif it:
            if tol_ups_after_best is not None:
                if error < errors[-2]:
                    netups = max(0, netups - 1)
                else:
                    netups = netups + 1
                    if netups > tol_ups_after_best:
                        break
            if np.abs(error - errors[0]) / error < tol_rel_error:
                break
        if callback is not None:
            callback(it, errors, A.T, B)

    return it // 2, Abest.T, Bbest, errors

