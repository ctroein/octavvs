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




# def mcr_als(sp, initial_components, maxiters=1000, reltol=1e-2,
#             callback_a=None, callback_b=None):
#     """
#     Perform MCR-ALS nonnegative matrix decomposition on the matrix sp

#     Parameters
#     ----------
#     sp : array(nrow, ncol)
#         Spectra to be decomposed.
#     initial_components : None or array(ncomponents, ncol)
#         Initial concentrations.
#     maxiters : int
#         Maximum number of iterations.
#     reltol : float
#         Relative error target.
#     callback_a : func(int iter, float err, A)
#         Callback for when concentrations(?) are updated
#     callback_concentrations : func(int iter, float err, B)
#         Callback for when spectra(?) are updated

#     Returns
#     -------
#     foo : array(...)
#     Something.

#     """

#     u, s, v = np.linalg.svd(sp)
#     nrow, ncol = np.shape(sp)
#     nr = initial_components.shape[0]
#     assert initial_components.shape[0] == ncol
#     s = scipy.linalg.diagsvd(s, nrow, ncol)
#     u = u[:, :nr]
#     s = s[:nr, :nr]
#     v = v[:nr, :]
#     dn = u @ s @ v
#     A = initial_components
#     dauxt = sklearn.preprocessing.normalize(dn.T)


#     for it in range(maxiters):

#         Btemp = scipy.optimize.nnls(A.T, dauxt.T)
#         # self.solve_lineq(ST_.T,dauxt.T)

#         Dcal = np.dot(Btemp.T, A)
#         error = sklearn.metrics.mean_squared_error(dauxt, Dcal)

#         if iter == 0:
#             error0 = error.copy()
#             B = Btemp

#         err = abs(error-error0)/error
#         print(it,'err',err)
#         if err == 0:
#             Bf = B.copy()
#             Af = A.copy()
#             A_temp = A.copy()
#             status = 'Iterating'

#         elif  per < reltol:
#             Cf = C_.copy()
#             Sf = ST_.copy()
#             status = 'converged'
#             break
#         else:
#             error0 = error.copy()
#             C_ = Ctemp.copy()
#             Sf = ST_.copy()
#             # status = 'Iterating'
#             # self.purest.emit(iter,per,status, C_.T,Sf.T)


