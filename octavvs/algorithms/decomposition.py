#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 18 14:45:53 2021

@author: carl
"""

import numpy as np


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
