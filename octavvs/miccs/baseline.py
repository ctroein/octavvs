#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  4 11:48:28 2019

A set of baseline correction algorithms

@author: Carl Troein
"""

import numpy as np
from scipy import sparse
from scipy.spatial import ConvexHull
from scipy.interpolate import interp1d
from scipy.sparse.linalg import spsolve
from multiprocessing.pool import Pool, ThreadPool
import os
import dill


def straight(x, y):
    """
    Return a straight line baseline correction.
    x: wavenumbers, sorted either way
    y: spectrum or spectra at those wavenumbers; shape (..., wavenumber)
    progressCallback(int a, int b): callback function called to indicated that the processing
        is complete to a fraction a/b.
    Returns: baseline of the spectrum, measured at the same points
    """
    # Create baseline using linear interpolation between vertices
    if x[0] < x[-1]:
        return interp1d(x[[0,-1]], y[...,[0,-1]], assume_sorted=True)(x)
    return interp1d(x[[-1,0]], y[...,[-1,0]], assume_sorted=True)(x)


def apply_packed_function_for_map(dumped):
    "Unpack dumped function as target function and call it with arguments."
    return dill.loads(dumped[0])(dumped[1])

def pack_function_for_map(target_function, items):
    dumped_function = dill.dumps(target_function)
    dumped_items = [(dumped_function, item) for item in items]
    return apply_packed_function_for_map, dumped_items


def mp_bgcorrection(func, y, lim_single=8, lim_tp=40, progressCallback=None):
    if len(y) < 1:
        return y.copy()
    if y.ndim < 2:
        return func(y)
    cpus = min(len(os.sched_getaffinity(0)), len(y))
    if cpus == 1 or len(y) <= lim_single:
        cpus = 1
        it = map(func, y)
    elif len(y) <= lim_tp:
        cpus = min(cpus, 3)
        pool = ThreadPool(cpus)
        it = pool.imap(func, y, chunksize=5)
    else:
        pool = Pool(cpus)
        it = pool.imap(*pack_function_for_map(func, y), chunksize=10)

    ret = np.empty_like(y)
    for i in range(len(y)):
        ret[i] = next(it)
        if progressCallback:
            progressCallback(i+1, len(y))
    return ret

def asls(y, lam, p, niter=20, progressCallback=None):
    """
    Return the baseline computed by Asymmetric least squares background correction, AsLS.
    Ref: Baseline correction with asymmetric least squares smoothing. PHC Eilers & HFM Boelens.
        Leiden University Medical Centre Report, 2005
    Parameters:
    y: one spectrum to correct, or multiple as an array of shape (spectrum, wavenumber)
    lam: lambda, the smoothness parameter
    p: p, the asymmetry parameter, typically .001 to 0.1
    niter: maximum number of iterations
    progressCallback(int a, int b): callback function called to indicated that the processing
        is complete to a fraction a/b.
    Returns: baseline of the spectrum, measured at the same points
    """
    L = y.shape[-1]
    D = sparse.csc_matrix(np.diff(np.eye(L), 2))
    D = lam * D.dot(D.T)

    def asls_one(yy):
        w = np.ones(L)
        for i in range(niter):
            z = spsolve(sparse.spdiags(w, 0, L, L) + D, w * yy)
            wnew = (p - .5) * np.sign(yy - z) + .5
#            wnew = p * (yy > z) + (1-p) * (yy < z)
            if np.array_equal(wnew, w):
                break
            w = wnew
        return z

    return mp_bgcorrection(asls_one, y, progressCallback=progressCallback)

def iasls(y, lam, lam1, p, niter=30, progressCallback=None):
    """
    Return the baseline computed by Improved asymmetric least squares background correction, IAsLS.
    Ref: Baseline correction for Raman spectra using an improved asymmetric least squares method.
        Shixuan He, Wei Zhang, Lijuan Liu, Yu Huang, Jiming He, Wanyi Xie, Peng Wu and Chunlei Du.
        Anal. Methods, 2014, 6, 4402-4407. DOI: 10.1039/C4AY00068D
    In this implementation, W is not squared so p carries the same meaning as in AsLS.
    Parameters:
    y: one spectrum to correct, or multiple as an array of shape (spectrum, wavenumber)
    lam: lambda, the smoothness parameter
    lam1: lambda1, the 1st derivatives smoothness parameter
    p: p, the asymmetry parameter
    niter: maximum number of iterations
    progressCallback(int a, int b): callback function called to indicated that the processing
        is complete to a fraction a/b.
    Returns: baseline of the spectrum, measured at the same points
    """
    L = y.shape[-1]
    D = sparse.csc_matrix(np.diff(np.eye(L), 2))
    D = lam * D.dot(D.T)
    D1 = sparse.csc_matrix(np.diff(np.eye(L), 1))
    D1 = lam1 * D1.dot(D1.T)

    def iasls_one(yy):
        w = np.ones(L)
        W = sparse.spdiags(w, 0, L, L)
#        W = W @ W.T
        z = spsolve(W + D, w * yy)
        w = p * (yy > z) + (1-p) * (yy < z)
        for i in range(niter):
            W = sparse.spdiags(w, 0, L, L)
#            W = W @ W.T
            z = spsolve(W + D + D1, (W + D1) * yy)
            wnew = p * (yy > z) + (1-p) * (yy < z)
            if np.array_equal(wnew, w):
                break
            w = wnew
        return z
    return mp_bgcorrection(iasls_one, y, progressCallback=progressCallback)


def arpls(y, lam, ratio=1e-6, niter=1000, progressCallback=None):
    """
    Return the baseline computed by asymmetric reweighted penalized least squares smoothing, arPLS.
    Ref: Baseline correction using asymmetrically reweighted penalized least squares smoothing
        Sung-June Baek, Aaron Park, Young-Jin Ahn and Jaebum Choo
        Analyst, 2015, 140, 250-257. DOI: 10.1039/C4AN01061B
    In this implementation, W is not squared so p carries the same meaning as in AsLS.
    Parameters:
    y: one spectrum to correct, or multiple as an array of shape (spectrum, wavenumber)
    lam: lambda, the smoothness parameter
    ratio: convergence criterion; target relative change in weights between iterations
    niter: maximum number of iterations
    progressCallback(int a, int b): callback function called to indicated that the processing
        is complete to a fraction a/b.
    Returns: baseline of the spectrum, measured at the same points
    """
    L = y.shape[-1]
    D = sparse.csc_matrix(np.diff(np.eye(L), 2))
    D = lam * D.dot(D.T)
    
    def arpls_one(yy):
        w = np.ones(L)
        for i in range(niter):
            W = sparse.spdiags(w, 0, L, L)
            z = sparse.linalg.spsolve(W + D, w * yy)
            d = yy - z
            dn = d[d < 0]
            s = dn.std()
            wt = 1. / (1 + np.exp(2 / s * (d - (2*s-dn.mean()))))
            if np.linalg.norm(w - wt) / np.linalg.norm(w) < ratio:
                break
            w = wt
        return z
    return mp_bgcorrection(arpls_one, y, progressCallback=progressCallback)



def rubberband(x, y, progressCallback=None):
    """
    Rubberband baseline correction of one or more spectra.
    Parameters:
    x: wavenumbers, sorted in either direction
    y: spectrum at those wavenumbers, or multiple spectra as array of shape (spectrum, wavenumber)
    progressCallback(int a, int b): callback function called to indicated that the processing
        is complete to a fraction a/b.
    Returns: baseline of the spectrum, measured at the same points
    """
    if x[0] > x[-1]:
        return rubberband(x[::-1], y[...,::-1],
                          progressCallback=progressCallback)[...,::-1]
    def rubberband_one(yy):
        # Find the convex hull
        v = ConvexHull(np.column_stack((x, yy))).vertices
        # Rotate convex hull vertices until they start from the lowest one
        v = np.roll(v, -v.argmin())
        # Leave only the ascending part
        v = v[:v.argmax()+1]
        # Create baseline using linear interpolation between vertices
        b = np.interp(x, x[v], yy[v])
        return b
    return mp_bgcorrection(rubberband_one, y, lim_single=100, lim_tp=10000,
                           progressCallback=progressCallback)

def concaverubberband(x, y, iters, progressCallback=None):
    """
    Concave rubberband baseline correction. This algorithm removes more than a
    straight line, alternating with normal rubberband to bring negative points
    up again. It does not converge nicely and will eat up all the data if run
    with many iterations.
    Parameters:
    x: wavenumbers, sorted from low to high (todo: implement high-to-low)
    y: spectrum at those wavenumbers
    iters: iterations to run; note that this algorithm doesn't converge nicely
    progressCallback(int a, int b): callback function called to indicated that the processing
        is complete to a fraction a/b.
    Returns: baseline of the spectrum, measured at the same points
    """
    def concaverubberband_one(yy):
        origyy = yy
        yy = yy - rubberband(x, yy);
        for i in range(iters):
            F = .1 * (yy.max() - yy.min())
            xmid = .5 * (x[-1] + x[0])
            d2 = .25 * (x[-1] - x[0]) ** 2
            yy += F * (x - xmid)**2 / d2
            yy -= rubberband(x, yy);
        return origyy - yy
    return mp_bgcorrection(concaverubberband_one, y, lim_single=30, lim_tp=500,
                           progressCallback=progressCallback)


