#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 18 14:45:53 2021

@author: carl
"""

import numpy as np
import scipy
import math
import time
from threadpoolctl import threadpool_limits

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
        dm[1:i+1, 1:i+1] = c[imp[:i], :][:, imp[:i]]
        for j in range(ncol):
            dm[0, 0] = c[j, j]
            dm[0, 1:i+1] = c[j, imp[:i]]
            dm[1:i+1, 0] = c[imp[:i], j]
            w[j] = np.linalg.det(dm[0:i+1, 0:i+1])
        imp[i] = (p * w).argmax()

    ss = d[:,imp]
    spout = ss / np.sqrt(np.sum(ss**2, axis=0))
    return spout.T, imp

def clustersubtract(data, components, skewness=100, power=2, verbose=False):
    """
    Create initial spectra for MCR-ALS based on successively removing
    what appears to be the strongest remaining component.

    Parameters
    ----------
    data : array (nspectra, nfeatures)
        Spectral data.
    components : int
        Number of components to return.
    skewness : float, optional
        Asymmetry between positive and negative residuals when computing
        how much of the previous component to remove from the data.
        The default is 100.
    power : float, optional
        The sum of residuals is raised to this power before summation to
        determine the leading remaining component.

    Returns
    -------
    initial_spectra : array (components, nfeatures)
    """
    def typical_cluster(data, first):
        # draw sqrt(n) random data points, r
        # find closest in r for each s in data
        # for r with most s, return mean of those s
        r = np.random.choice(len(data), min(200, math.floor(math.sqrt(len(data)))))
        rd = data[r]
        nearest = scipy.spatial.distance.cdist(
            rd, data, 'cosine').argmin(axis=0)
        # Mean of those who are nearest the biggest cluster
        if first:
            selected = np.bincount(nearest).argmax()
        else:
            sums = data.sum(1) ** power
            selected = np.bincount(nearest, weights=sums).argmax()
        return data[nearest == selected].mean(0)

    comps = []
    sgn = np.ones_like(data, dtype=bool)
    for c in range(components):
        tc = typical_cluster(data, c == 0)
        tc = np.maximum(tc, 0)
        tc = tc / (tc * tc).sum() ** .5

        comps.append(tc)
        for i in range(10):
            ww = 1 * sgn + skewness * ~sgn
            a = (data * ww * tc).sum(1) / (ww * tc * tc).sum(1)
            oldsgn = sgn
            sgn = data > a[:, None] @ tc[None, :]
            if verbose:
                chg = (sgn != oldsgn).sum()
                print(f'clsub iter {c:3d} {i:2d} changed {chg}')
            if np.array_equal(sgn, oldsgn):
                break
        data = data - a[:, None] @ tc[None, :]
    return np.array(comps)

def numpy_scipy_threading_fix_(func):
    """
    This decorator for mcr_als prevents threading in BLAS if scipy's NNLS
    is used, because for some reason NNLS won't be parallelized if called
    shortly after lstsq or @. This makes a *massive* difference to the
    time needed for Anderson acceleration, where the BLAS calls themselves
    take negligible time. For mixed NNLS/lstsq solving (of MCR-ALS on
    derivatives) it's less obvious whether NNSL or lstsq should be allowed
    to be parallelized.
    Note: This issue is seen on my Linux setup and might be highly version-
    dependent. More investigation is needed.
    """
    def check(*args, **kwargs):
        if 'nonnegative' not in kwargs or np.any(kwargs['nonnegative']):
            with threadpool_limits(1, 'blas'):
                return func(*args, **kwargs)
        else:
            return func(*args, **kwargs)
    return check

@numpy_scipy_threading_fix_
def mcr_als(sp, initial_A, *, maxiters, nonnegative=(True, True),
            tol_abs_error=0, tol_rel_improv=None, tol_iters_after_best=None,
            maxtime=None, callback=None, acceleration=None, normalize=None,
            contrast_weight=None, return_time=False, **kwargs):
    """
    Perform MCR-ALS nonnegative matrix decomposition on the matrix sp

    Parameters
    ----------
    sp : array(nsamples, nfeatures)
        Spectra to be decomposed.
    initial_A : array(ncomponents, nfeatures)
        Initial spectra or concentrations.
    maxiters : int
        Maximum number of iterations.
    nonnegative : pair of bool, default (True, True)
        True if (initial, other) components must be non-negative
    tol_abs_error : float, optional
        Error target (mean square error).
    tol_rel_improv : float, optional
        Stop when relative improvement is less than this over 10 iterations.
    tol_iters_after_best : int, optional
        Stop after this many iterations since last best error.
    maxtime : float, optional
        Stop after this many seconds of process time have elapsed
    callback : func(it : int, err : float, A : array, B : array)
        Callback for every iteration.
    acceleration : str, optional
        None or 'Anderson'.
        Anderson acceleration operates on whole iterations (A or B updates),
        mixing earlier directions to step towards the fixed point. This
        implementation restarts from basic updates when those would be
        better.
    normalize : str, optional
        Which matrix to l2 normalize: None, 'A' or 'B'
    contrast_weight : (str, float), optional
        Increase contrast in one matrix by mixing the other, named matrix
        ('A' or 'B') with the mean of its vectors. If A is spectra,
        try contrast_weight=('B', 0.05) to increase spectral contrast.
        See Windig and Keenan, Applied Spectroscopy 65: 349 (2011).
    return_time : bool, default False
        Measure and return process_time at each iteration.

    Anderson acceleration parameters in kwargs
    -------
    m : int, >1, default 2
        The maximum number of earlier steps to consider.
    alternate : bool, default True
        Alternate between accelerating A and B, switching when restarting.
    beta : float, default 1.
        Scaling factor for accelerated step length.
    betascale : float, default 1.
        Reduction factor for beta after each restart.
    bmode : bool, default False
        Start with accelerating B instead of A.

    Returns
    -------
    A : array(ncomponents, nfeatures)
        Spectra (at lowest error)
    B : array(ncomponents, nsamples)
        Concentrations at lowest error
    error : list(float)
        Mean square error at every iteration
    process_time : list(float)
        Time relative start at each iteration, only if return_time is True.
    """
    if normalize not in [None, 'A', 'B']:
        raise ValueError('Normalization must be None, A or B')
    unknown_args = kwargs.keys() - {
        'm', 'alternate', 'beta', 'betascale', 'bmode'}
    if unknown_args:
        raise TypeError('Unknown arguments: {}'.format(unknown_args))

    nrow, ncol = sp.shape
    nr = initial_A.shape[0]
    if normalize == 'A':
        norm = np.linalg.norm(initial_A, axis=1)
        A = np.divide(initial_A.T, norm, where=norm!=0,
                      out=np.zeros(initial_A.shape[::-1]))
    else:
        A = initial_A.T.copy()
    B = np.empty((nr, nrow))
    errors = []
    errorbest = None # Avoid spurious warning
    # prevA, prevB = (None, None)
    newA = newB = None
    error = preverror = None

    cw = 0
    if contrast_weight is not None:
        if contrast_weight[0] == 'A':
            cw = contrast_weight[1]
        elif contrast_weight[0] == 'B':
            cw = -contrast_weight[1]
        else:
            raise ValueError("contrast_weight must be ('A'|'B', [0-1])")


    if acceleration == 'Anderson':
        ason_Bmode = kwargs.get('bmode', False)
        ason_alternate = kwargs.get('alternate', True)
        ason_m = kwargs.get('m', 2)
        ason_beta = kwargs.get('beta', 1.)
        ason_betascale = kwargs.get('betascale', 1.)
        ason_g = None
        ason_G = []
        ason_X = []
    elif acceleration:
        raise ValueError("acceleration must be None or 'Anderson'")

    starttime = time.process_time()
    if return_time:
        times = []
    tol_rel_iters = 10

    for it in range(maxiters):
        ba = 0
        retry = False
        while ba < 2:
            if not retry:
                preverror = error
            if ba == 0:
                if newA is None:
                    newA = A
                prevA = newA
                if cw > 0:
                    newA = (1 - cw) * newA + cw * newA.mean(1)[:,None]
                if nonnegative[1]:
                    error = 0
                    if not retry:
                        B = np.empty_like(B)
                    for i in range(nrow):
                        B[:, i], res = scipy.optimize.nnls(newA, sp[i, :])
                        error = error + res * res
                else:
                    B, res, _, _ = np.linalg.lstsq(newA, sp.T, rcond=-1)
                    error = res.sum()
                if normalize == 'B':
                    norm = np.linalg.norm(B, axis=1)
                    B = np.divide(B.T, norm, where=norm!=0, out=B.T).T
                newA = None
            else:
                if newB is None:
                    newB = B
                prevB = newB
                if cw < 0:
                    newB = (1 + cw) * newB - cw * newB.mean(1)[:,None]
                if nonnegative[0]:
                    error = 0
                    if not retry:
                        A = np.empty_like(A)
                    for i in range(ncol):
                        A[i, :], res = scipy.optimize.nnls(newB.T, sp[:, i])
                        error = error + res * res
                else:
                    A, res, _, _ = np.linalg.lstsq(newB.T, sp, rcond=-1)
                    A = A.T
                    error = res.sum()
                if normalize == 'A':
                    norm = np.linalg.norm(A, axis=0)
                    np.divide(A, norm, where=norm!=0, out=A)
                newB = None

            if acceleration is None:
                pass
            elif ba == ason_Bmode:
                if retry:
                    retry = False
                    if ason_alternate:
                        ason_Bmode = not ason_Bmode
                    ason_beta = ason_beta * ason_betascale
                elif len(ason_X) > 1 and error > preverror:
                    ason_X = []
                    ason_G = []
                    retry = True
                    ba = ba - 1
                else:
                    pass
            elif ason_Bmode == 1 and it == 0:
                pass
            else:
                prevg = ason_g
                ason_g = ((A - prevA) if ba else (B - prevB)).flatten()
                if len(ason_X) < 1:
                    ason_X.append(ason_g)
                else:
                    ason_G.append(ason_g - prevg)
                    while(len(ason_G) > ason_m):
                        ason_G.pop(0)
                        ason_X.pop(0)
                    Garr = np.asarray(ason_G)
                    try:
                        gamma = scipy.linalg.lstsq(Garr.T, ason_g)[0]
                    except scipy.linalg.LinAlgError:
                        print('lstsq failed to converge; '
                              'restart at iter %d' % it)
                        # print('nans', np.isnan(Garr).sum(),
                        #       np.isnan(ason_g).sum())
                        ason_X = []
                        ason_G = []
                    else:
                        gamma = ason_beta * gamma
                        dx = ason_g - gamma @ (np.asarray(ason_X) + Garr)
                        ason_X.append(dx)
                        if ba:
                            newA = prevA + dx.reshape(A.shape)
                            if nonnegative[0]:
                                np.maximum(0, newA, out=newA)
                        else:
                            newB = prevB + dx.reshape(B.shape)
                            if nonnegative[1]:
                                np.maximum(0, newB, out=newB)
            ba = ba + 1
        # error = error / sp.size

        curtime = time.process_time() - starttime
        if return_time:
            times.append(curtime)
        errors.append(error)
        if not it or error < errorbest:
            errorbest = error
            Abest = A
            Bbest = B
            iterbest = it
        if it:
            if error < tol_abs_error:
                break
            if tol_rel_improv and it > tol_rel_iters:
                emax = max(errors[-tol_rel_iters-1:-2])
                if (emax - errors[-1]) <= \
                    tol_rel_improv * tol_rel_iters * emax:
                        break
            if tol_iters_after_best is not None:
                if iterbest + tol_iters_after_best < it:
                    break
        if it and maxtime and curtime >= maxtime:
            break
        if callback is not None:
            callback(it, errors, A.T, B)

    if return_time:
        return Abest.T, Bbest, errors, times
    return Abest.T, Bbest, errors

