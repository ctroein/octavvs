#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Carl Troein

Atmospheric and scattering correction

"""

import gc
import os.path
from time import monotonic
import numpy as np
import sklearn.linear_model
import sklearn.cluster
#import statsmodels.multivariate.pca
from scipy.interpolate import PchipInterpolator
from scipy.signal import hilbert, savgol_filter, tukey
from scipy.io import loadmat, savemat
import matplotlib.pyplot as plt

from . import baseline

def load_reference(wn, what=None, matfilename=None):
    """
    Loads and normalizes a spectrum from a Matlab file, interpolating at the given points.
        The reference is assumed to cover the entire range of wavenumbers.
    Parameters:
        wn: array of wavenumbers at which to get the spectrum
        what: A string defining what type of reference to get, corresponding to a file in the
        'reference' directory
        matfilename: the name of an arbitrary Matlab file to load data from; the data must be
        in a matrix called AB, with wavenumbers in the first column.
        Returns: spectrum at the points given by wn
    """
    if (what is None) == (matfilename is None):
        raise ValueError("Either 'what' or 'matfilename' must be specified")
    if what is not None:
        matfilename = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__),
                                       'reference', what + '.mat'))
    ref = loadmat(matfilename)['AB']
    # Handle the case of high-to-low since the interpolator requires low-to-high
    d = 1 if ref[0,0] < ref[-1,0] else -1
    ref = PchipInterpolator(ref[::d,0], ref[::d,1])(wn)
    return ref #/ ref.max()

def nonnegative(y, fracspectra=.02, fracvalues=.02):
    """
    Make a matrix of spectral data nonnegative by shifting all the spectra up by the same computed
    amount, followed by setting negative values to 0. The shift is chosen such that at most
    fracspectra of the spectra get more than fracvalues of their intensities set to zero.
    Parameters:
    y: array of intensities for (pixel, wavenumber)
    fracspectra: unheeded fraction of the spectra
    fracvalues: maximal fraction of points to clip at 0
    Returns: shifted spectra in the same format as y
    """
    s = int(fracspectra * y.shape[0])
    v = int(fracvalues * y.shape[1])
    if s == 0 or v == 0:
        return y - np.min(y.min(), 0)
    if s >= y.shape[0] or v >= y.shape[1]:
        return np.maximum(y, 0)
    yp = np.partition(y, v, axis=1)[:,v]
    a = np.partition(yp, s)[s]
    return np.maximum(y - a if a < 0 else y, 0)

def find_wn_ranges(wn, ranges):
    """
    Find indexes corresponding to the beginning and end of a list of ranges of wavenumbers. The
    wavenumbers have to be sorted in either direction.
    Parameters:
    wn: array of wavenumbers
    ranges: numpy array of shape (n, 2) with desired wavenumber ranges in order [low,high]
    Returns: numpy array of shape (n, 2) with indexes of the wavenumbers delimiting those ranges
    """
    if isinstance(ranges, list):
        ranges = np.array(ranges)
    if(wn[0] < wn[-1]):
        return np.stack((np.searchsorted(wn, ranges[:,0]),
                         np.searchsorted(wn, ranges[:,1], 'right')), 1)
    return len(wn) - np.stack((np.searchsorted(wn[::-1], ranges[:,1], 'right'),
                              np.searchsorted(wn[::-1], ranges[:,0])), 1)

def cut_wn(wn, y, ranges):
    """
    Cut a set of spectra, leaving only the given wavenumber range(s).
    Parameters:
    wn: array of wavenumbers, sorted in either direction
    y: array of spectra, shape (..., wavenumber)
    ranges: list or numpy array of shape (..., 2) with desired wavenumber ranges in pairs (low, high)
    Returns: (wavenumbers, spectra) with data in the given wavenumber ranges
    """
    if isinstance(ranges, list):
        ranges = np.array(ranges)
    inrange = lambda w: ((w >= ranges[...,0]) & (w <= ranges[...,1])).any()
    ix = np.array([inrange(w) for w in wn])
    return wn[ix], y[...,ix]

def atmospheric(wn, y, atm=None, cut_co2 = True, extra_iters=5, extra_factor=0.25,
                       smooth_win=9, progressCallback = None):
    """
    Apply atmospheric correction to multiple spectra, subtracting as much of the atompsheric
    spectrum as needed to minimize the sum of squares of differences between consecutive points
    in the corrected spectra. Each supplied range of wavenumbers is corrected separately.

    Parameters:
        wn: array of wavenumbers, sorted in either direction
        y: array of spectra in the order (pixel, wavenumber), or just one spectrum
        atm: atmospheric spectrum; if None, load the default
        cut_co2: replace the CO2 region with a neatly fitted spline
        extra_iters: number of iterations of subtraction of a locally reshaped atmospheric spectrum
        (needed if the relative peak intensities are not always as in the atmospheric reference)
        extra_factor: how much of the reshaped atmospheric spectrum to remove per iteration
        smooth_win: window size (in cm-1) for smoothing of the spectrum in the atm regions
        progressCallback(int a, int b): callback function called to indicated that the processing
        is complete to a fraction a/b.
    Returns:
        tuple of (spectra after correction, array of correction factors; shape (spectra,ranges))
    """
    squeeze = False
    yorig = y
    if y.ndim == 1:
        y = y[None,:]
        squeeze = True
    else:
        y = y.copy()

    if atm is None or (isinstance(atm, str) and atm == ''):
        atm = load_reference(wn, what='water')
    elif isinstance(atm, str):
        atm = load_reference(wn, matfilename=atm)
    else:
        atm = atm.copy()

    # ranges: numpy array (n, 2) of n non-overlapping wavenumber ranges (typically for H2O only), or None
    # extra_winwidth: width of the window (in cm-1) used to locally reshape the atm spectrum
    ranges = [[1300, 2100], [3410, 3850], [2190, 2480]]
    extra_winwidth = [30, 150, 40]
    corr_ranges = 2 if cut_co2 else 3
#        ranges = ranges[:2]
#        extra_winwidth = extra_winwidth[:2]

    if ranges is None:
        ranges = np.array([0, len(wn)])
    else:
        ranges = find_wn_ranges(wn, ranges)

    for i in range(corr_ranges):
        p, q = ranges[i]
        if q - p < 2: continue
        atm[p:q] -= baseline.straight(wn[p:q], atm[p:q]);

    savgolwin = 1 + 2 * int(smooth_win * (len(wn) - 1) / np.abs(wn[0] - wn[-1]))

    if progressCallback:
        progressA = 0
        progressB = 1 + corr_ranges * (extra_iters + (1 if savgolwin > 1 else 0))
        progressCallback(progressA, progressB)

    dh = atm[:-1] - atm[1:]
    dy = y[:,:-1] - y[:,1:]
    dh2 = np.cumsum(dh * dh)
    dhdy = np.cumsum(dy * dh, 1)
    az = np.zeros((len(y), corr_ranges))
    for i in range(corr_ranges):
        p, q = ranges[i]
        if q - p < 2: continue
        r = q-2 if q <= len(wn) else q-1
        az[:, i] = ((dhdy[:,r] - dhdy[:,p-1]) / (dh2[r] - dh2[p-1])) if p > 0 else (dhdy[:,r] / dh2[r])
        y[:, p:q] -= az[:, i, None] @ atm[None, p:q]

    if progressCallback:
        progressA += 1
        progressCallback(progressA, progressB)

    for pss in range(extra_iters):
        for i in range(corr_ranges):
            p, q = ranges[i]
            if q - p < 2: continue
            window = 2 * int(extra_winwidth[i] * (len(wn) - 1) / np.abs(wn[0] - wn[-1]))
            winh = (window+1)//2
            dy = y[:,:-1] - y[:,1:]
            dhdy = np.cumsum(dy * dh, 1)
            aa = np.zeros_like(y)
            aa[:,1:winh+1] = dhdy[:,1:window:2] / np.maximum(dh2[1:window:2], 1e-8)
            aa[:,1+winh:-winh-1] = (dhdy[:,window:-1] - dhdy[:,:-1-window]) / np.maximum(dh2[window:-1] - dh2[:-1-window], 1e-8)
            aa[:,-winh-1:-1] = (dhdy[:,-1:] - dhdy[:,-1-window:-1:2]) / np.maximum(dh2[-1] - dh2[-1-window:-1:2], 1e-8)
            aa[:, 0] = aa[:, 1]
            aa[:, -1] = aa[:, -2]
            aa = savgol_filter(aa, window + 1, 3, axis=1)
            y[:, p:q] -= extra_factor * aa[:, p:q] * atm[p:q]
            if progressCallback:
                progressA += 1
                progressCallback(progressA, progressB)

    if savgolwin > 1:
        for i in range(corr_ranges):
            p, q = ranges[i]
            if q - p < savgolwin: continue
            y[:, p:q] = savgol_filter(y[:, p:q], savgolwin, 3, axis=1)
            if progressCallback:
                progressA += 1
                progressCallback(progressA, progressB)

    if cut_co2:
        rng = np.array([[2190, 2260], [2410, 2480]])
        rngm = rng.mean(1)
        rngd = rngm[1] - rngm[0]
        cr = find_wn_ranges(wn, rng).flatten()
        if cr[1] - cr[0] > 2 and cr[3] - cr[2] > 2:
            a = np.empty((4, len(y)))
            a[0:2,:] = np.polyfit((wn[cr[0]:cr[1]]-rngm[0])/rngd, y[:,cr[0]:cr[1]].T, deg=1)
            a[2:4,:] = np.polyfit((wn[cr[2]:cr[3]]-rngm[1])/rngd, y[:,cr[2]:cr[3]].T, deg=1)
            P,Q = find_wn_ranges(wn, rngm[None,:])[0]
            t = np.interp(wn[P:Q], wn[[Q,P] if wn[0] > wn[-1] else [P,Q]], [1, 0])
            tt = np.array([-t**3+t**2, -2*t**3+3*t**2, -t**3+2*t**2-t, 2*t**3-3*t**2+1])
            pt = a.T @ tt
            y[:, P:Q] += (pt - y[:, P:Q]) * tukey(len(t), .3)

    corrs = np.zeros(2)
    ncorrs = np.zeros_like(corrs)
    for i in range(len(ranges)):
        p, q = ranges[i]
        if q - p < 2: continue
        corr = np.abs(yorig[:, p:q] - y[:, p:q]).sum(1) / np.maximum(np.abs(yorig[:, p:q]), np.abs(y[:, p:q])).sum(1)
        gas = int(i > 1)
        corrs[gas] += corr.mean()
        ncorrs[gas] += 1
    if ncorrs[0] > 1:
        corrs[0] = corrs[0] / ncorrs[0]

    return (y.squeeze() if squeeze else y), corrs

def kkre(wn, ref):
    wn2 = wn ** 2.
    wa = wn * ref
    kk = np.empty_like(wn)
    for i in range(len(wn)):
        with np.errstate(divide='ignore', invalid='ignore'):
            fg = wa / (wn2 - wn[i] ** 2.)
        if i == 0 or i == len(wn) - 1:
            fg[i] = 0
        else:
            fg[i] = (fg[i-1] + fg[i+1]) / 2
        kk[i] = 2/np.pi * np.trapz(x=wn, y=fg)
    if wn[0] < wn[-1]:
        return kk
    return -kk

def hilbert_n(wn, ref, zeropad=500):
    """
    Compute the Kramers-Kronig relations by Hilbert transform, extending the absorption spectrum
    with 0 to either size and resampling the spectrum at evenly spaced intervals if it is found to
    be unevenly sampled.
    """
    # Cache some data structures to avoid having to reinitialize them on every call.
    if not hasattr(hilbert_n, "wn") or hilbert_n.wn is not wn or hilbert_n.zp != zeropad:
        hilbert_n.wn = wn
        even = (wn[-1] - wn[0]) / (len(wn) - 1)
        diff = np.abs((np.diff(wn) - even) / even).mean()
        hilbert_n.zp = zeropad
        hilbert_n.evenspaced = diff < 1e-3
        hilbert_n.increasing = wn[0] < wn[-1]
#        print('hilbert',hilbert_n.evenspaced,hilbert_n.increasing)
        if not hilbert_n.evenspaced:
            hilbert_n.lin = np.linspace(min(wn[0], wn[-1]), max(wn[0], wn[-1]), len(wn))
        hilbert_n.npad = int(len(wn) / abs(wn[-1] - wn[0]) * zeropad)
        hilbert_n.nim = np.zeros((len(wn) + 2 * hilbert_n.npad))

    if hilbert_n.evenspaced:
        if hilbert_n.npad == 0:
            if hilbert_n.increasing:
                hilbert_n.nim = ref
            else:
                hilbert_n.nim = ref[::-1]
        elif hilbert_n.increasing:
            hilbert_n.nim[hilbert_n.npad:hilbert_n.npad+len(wn)] = ref
        else:
            hilbert_n.nim[hilbert_n.npad:hilbert_n.npad+len(wn)] = ref[::-1]
    else:
        if hilbert_n.increasing:
            hilbert_n.nim[hilbert_n.npad:hilbert_n.npad+len(wn)] = PchipInterpolator(wn, ref)(hilbert_n.lin)
        else:
            hilbert_n.nim[hilbert_n.npad:hilbert_n.npad+len(wn)] = PchipInterpolator(wn[::-1], ref[::-1])(hilbert_n.lin)
    nreal = -np.imag(hilbert(hilbert_n.nim))
    if hilbert_n.npad:
        nreal = nreal[hilbert_n.npad:-hilbert_n.npad]
    if hilbert_n.evenspaced:
        return nreal if hilbert_n.increasing else nreal[::-1]
    return PchipInterpolator(hilbert_n.lin, nreal)(wn)

def pca_nipals(x, ncomp, tol=1e-5, max_iter=1000, copy=True, explainedvariance=None):
    """
    NIPALS algorithm for PCA, based on the code in statmodels.multivariate
    but with optimizations as in Bassan's Matlab implementation to
    sacrifice some accuracy for speed.
    x: ndarray of data, will be altered
    ncomp: number of PCA components to return
    tol: tolerance
    copy: If false, destroy the input matrix x
    explainedvariance: If >0, stop after this fraction of the total variance is explained
    returns: PCA loadings as rows
    """
    if copy:
        x = x.copy()
    if explainedvariance is not None and explainedvariance > 0:
        varlim = (x * x).sum() * (1. - explainedvariance)
    else:
        varlim = 0
    npts, nvar = x.shape
    vecs = np.empty((ncomp, nvar))
    for i in range(ncomp):
        factor = np.ones(npts)
        for j in range(max_iter):
            vec = x.T @ factor #/ (factor @ factor)
            vec = vec / np.sqrt(vec @ vec)
            f_old = factor
            factor = x @ vec #/ (vec @ vec)
            f_old = factor - f_old
            if tol > np.sqrt(f_old @ f_old) / np.sqrt(factor @ factor):
                break
        vecs[i, :] = vec
        if i < ncomp - 1:
            x -= factor[:,None] @ vec[None,:]
            if varlim > 0 and (x * x).sum() < varlim:
                return vecs[:i+1, :]
    return vecs

def compute_model(wn, ref, n_components, a, d, bvals, konevskikh=True, linearcomponent=False,
                  variancelimit = None):
    """
    Support function for rmiesc_miccs. Compute the extinction matrix for Bassan's algorithm,
    then PCA transform it.
    Parameters:
    wn: array of wavenumbers
    ref: reference spectrum
    n_components: number of PCA components to use
    a: array of values for the parameter a (index of refraction)
    d: array of values for the parameter d*4pi (sphere size)
    bvals: number of values for parameter b (mixing of a and real part of n from absorption ref)
    konevskikh: if True, use the faster method by Konevskikh et al.
    variancelimit: if a number (around 0.9996), use as many PCA components as needed to
       explain this fraction of the variance of the extinction matrix
    """

    # Compute the scaled real part of the refractive index by Kramers-Kronig transform:
    # We skip a factor 2pi because it's normalized away anyway.
#    n_im = ref / wn
#    nkk = -np.imag(hilbert(n_im)) if wn[0] < wn[-1] else -np.imag(hilbert(n_im[::-1]))[::-1]

    # I'm a bit confused about the division/multiplication by wn.
    # Bassan's matlab code uses wn*ref in kkre
        # bassan2010 says k (=n_im) is proportional to ref
        # but various sources say k is ref/wn * const
    # My kkre reproduces's Bassan with the same wn*ref
    # My hilbert_n gives the same output with Hilbert transform of ref
    # Causin's python code Hilbert transforms ref for Bassan's algorith but ref/wn for Konevskikh's!
    # Solheim's matlab code Hilbert transforms ref/wn

    if konevskikh:
#        nim = ref / ref.max() / (wn * 100)
        nim = ref / (wn * 100)
        nre = hilbert_n(wn, nim, 300)
        nmin = nre.min()
#        print('refmax', ref.max(), 'nrange', nmin, nre.max())
        if nmin < -1:
            nre = nre / -nmin
            nim = nim / -nmin
        # My revised distribution of alpha_0 and gamma
        alpha_0 = 1e-2 * np.linspace(d[0] * (a[0]-1), d[-1] * (a[-1]-1), len(a))
        gamma = .25 * 2 * np.log(10) / np.pi * np.linspace(1 / alpha_0[0], 1 / alpha_0[-1], len(alpha_0))
        # Solheim's distributions of alpha_0 and gamma
#        alpha_0 = 1e-2 * d * (a - 1)
#        gamma = .25  * 2 * np.log(10) / np.pi / alpha_0
        Q = np.empty((len(alpha_0) * len(gamma), len(wn)))  # Initialize the extinction matrix

#        print('alpha_0', alpha_0)
#        print('gamma', gamma)
        # Build the extinction matrix
        n_row = 0
        for a0 in alpha_0:
            for g in gamma:
                rho = a0 * (1. + g * nre) * wn * 100
                denom = 1. / g + nre
                tanbeta = nim / denom
                beta = np.arctan2(nim, denom)
                cosb = np.cos(beta)
                cosbrho = cosb / rho

                # Following Konevskikh et al 2016
                Q[n_row] = 2. - 4. * cosbrho * (np.exp(-rho * tanbeta) *
                     (np.sin(rho - beta) + cosbrho * np.cos(rho - 2 * beta)) -
                     cosbrho * np.cos(2 * beta))

                n_row += 1
#        savemat('everything-p.mat', locals())
    else:
        nkk = kkre(wn, ref/wn) # should divide by wn here (or not multiply by wn in the function)
#        nkk = hilbert_n(wn, ref / wn, 300) # should divide by wn
        nkk = nkk / abs(nkk.min())

        # Build the extinction matrix
        Q = np.empty((len(a) * bvals * len(d), len(wn)))  # Initialize the extinction matrix
        n_row = 0
        for i in range(len(a)):
            b = np.linspace(0.0, a[i] - 1.01, bvals)  # Range of amplification factors of nkk
            for j in range(len(b)):
                n = a[i] + b[j] * nkk  # Compute the real refractive index
                for k in range(len(d)):
                    rho = d[k] * (n - 1.) * wn
                    #  Compute the extinction coefficients for each combination of a, b and d:
                    Q[n_row] = 2. - 4. / rho * np.sin(rho) + \
                               4. / (rho * rho) * (1. - np.cos(rho))
                    n_row += 1

    n_nonpca = 3 if linearcomponent else 2

    # Orthogonalization of the model to improve numeric stability
    refn = ref / np.sqrt(ref@ref)
    Q = Q - (Q @ refn)[:,None] @ refn[None,:]

    # Perform PCA of the extinction matrix
    pca = pca_nipals(Q, ncomp=n_components, tol=1e-5, copy=False, explainedvariance=variancelimit)
    model = np.empty((n_nonpca + pca.shape[0], pca.shape[1]))
    model[n_nonpca:, :] = pca

#    # This method is up to 50% slower but gives additional info
#    pca = statsmodels.multivariate.pca.PCA(Q, ncomp=n_components,
#           method='nipals', tol=1e-5, demean=False, standardize=False)
#    esum = pca.eigenvals.cumsum()
#    n_components = np.min((np.searchsorted(esum, esum[-1] * variancelimit) + 1, len(esum)))
#    model = np.zeros((n_nonpca + n_components, len(wn)))
#    model[n_nonpca:, :] = pca.loadings[:,:n_components].T

    model[0,:] = ref
    model[1,:] = 1
    if linearcomponent:
        model[2,:] = np.linspace(0., 1., len(wn))
    if linearcomponent:
        model[2,:] = np.linspace(0., 1., len(wn))
#    for i in range(2, n_nonpca):
#        w = model[i, :] - np.sum(np.dot(model[i, :], b) * b for b in model[0:i, :])
#        model[i,:] = w / np.sqrt(w @ w)

#    savemat('everything_after-p.mat', locals())
#    killmenow

    # Orthogonalization of the model to improve numeric stability (doing it after PCA is only
    # marginally slower)
#    for i in range(len(model)):
#        v = model[i, :]
#        w = v - np.sum(np.dot(v, b) * b for b in model[0:i, :])
#        model[i,:] = w / np.linalg.norm(w)

    return model

def stable_rmiesc_clusters(iters, clusters):
    """
    Make a cluster size scheme for reliable convergence in rmiesc_miccs.
    Parameters:
    iters: The number of basic iterations to be used, preferably at least about 12-20
    Returns:
    array of cluster sizes (or zeros) for each iteration; this will be bigger than
    the input iterations
    """
    iters = max(2, iters) * 2
    cc = np.zeros(iters, dtype=np.int)
    cc[:iters//3] = 1
    cc[iters//2:iters*3//4] = clusters
    return cc

def rmiesc(wn, app, ref, n_components=7, iterations=10, clusters=None,
           pcavariancelimit=None,
           verbose=False, a=np.linspace(1.1, 1.5, 10), d=np.linspace(2.0, 8.0, 10),
           bvals=10, plot=False, progressCallback = None, progressPlotCallback=None,
           konevskikh=False, linearcomponent=True, weighted=False, renormalize=False,
           autoiterations=False, targetrelresiduals=0.95):
    """
    Correct scattered spectra using Bassan's algorithm. This implementation does no orthogonalization
    of the extinction matrix or PCA components relative to the reference, nor is the reference smoothed
    or filtered through a sum of gaussians as in the original Matlab implementation.
    Parameters:
    wn: sorted array of wavenumbers (high-to-low or low-to-high)
    app: apparent spectrum, shape (pixels, wavenumbers)
    ref: reference spectrum; array (wavenumbers)
    n_components: number of principal components to be calculated for the extinction matrix
    iterations: number of iterations of the algorithm
    clusters: if not None, cluster pixels into this many clusters in each iteration and use
        a common reference spectrum for each cluster. May be given as a list with one value per
        iteration, in which case 0 means to reuse clusters from the previous iteration and mix
        new/old references for stable convergence.
        If clusters is negative, use stable_rmiesc_clusters to generate the list.
    verbose: print progress information
    a: indexes of refraction to use in model
    d: sphere sizes to use in model, in micrometers
    bvals: number of values for the model parameter b
    plot: produce plots of the cluster references, if in cluster mode
    progressCallback(int a, int b): callback function called to indicated that the processing
        is complete to a fraction a/b.
    konevskikh: if True, use the faster method by Konevskikh et al.
    linearcomponent: if True, include a linear term in the model (used in Bassan's paper only).
    weighted: if true, downweight the 1800-2800 region when fitting the model.
    renormalize: if True, renormalize spectra against reference in each generation.
    autoiterations; if True, iterate until residuals stop improving
    targetrelresiduals: if autoiterations, stop when this relative change in residuals is seen
    Return: corrected apparent spectra (the best encountered if autoiterations, else the final ones)
    """

    # Make a rescaled copy of d and include the factor 4*pi
    d = d * 4e-4  * np.pi;

    # The input can be a single spectrum or a matrix of spectra. If the former, squeeze at the end.
    squeeze = False
    if app.ndim == 1:
        app = app[None,:]
        squeeze = True

    if weighted:
        weights = np.ones_like(wn)
        weights[range(*find_wn_ranges(wn, [[1800, 2800]])[0])] = .001 ** .5
        weights = weights[:, None]
    else:
        weights = None

    if plot:
        plt.figure()
        color=plt.cm.jet(np.linspace(0, 1, iterations))
        plt.plot(wn, app.mean(0), 'k', linewidth=.5)

    if np.isscalar(clusters):
        if clusters == 0:
            clusters = None
        elif clusters < 0:
            clusters = stable_rmiesc_clusters(iterations, -clusters)
            iterations = len(clusters)
        else:
            clusters = np.repeat(clusters, iterations)
    elif clusters is not None:
        if len(clusters) != iterations:
            raise ValueError('len(clusters) must match iterations')
        clusters = clusters.copy()

    if progressCallback:
        # Compute the number of progress steps
        progressA = 0
        if clusters is None:
            progressB = 1 + (iterations > 1) * len(app)
        else:
            progressB = 0
            prev = 1
            for cl in clusters:
                if cl > 0:
                    prev = cl
                progressB += prev
    startt = monotonic()

    corrected = None # Just to get rid of warnings in the editor; will be set on iteration 0

    # Set parameters for automatic iteration control
    if renormalize:
        autoupadd = 3  # Residual going up counts as residual going down too little this many times
        automax = 3    # Stop when residual has gone down too little this many times
    else:
        autoupadd = 1
        automax = 5

    if clusters is not None:
        # Cluster mode: In each iteration, after correcting all the spectra, cluster them. Then take the
        # mean of the corrected spectra in each cluster as the new reference for that cluster in the next
        # iteration.
        ref = ref.copy()[None, :]    # One reference per cluster
        ref = ref / (np.abs(ref).mean() / np.abs(app).mean())
        labels = np.zeros(len(app))  # Cluster labels; initially all in cluster 0
#        clusters[-1] = 0
        progstep = 1 # Current progress bar step size

        for iteration in range(iterations):
            gc.collect()    # Because my old laptop was unhappy with RAM usage otherwise

            curc = clusters[iteration] # Current cluster size setting
            if curc > 0:
                progstep = curc

            # Skip this iteration if every spectrum has stopped improving and the cluster settings
            # are unchanged
            if autoiterations:
                if not iteration or curc != clusters[iteration-1]:
                    unimproved = np.zeros(len(app), dtype=int)
                elif (unimproved <= automax).sum() == 0:
                    progressA += progstep
                    if progressCallback:
                        progressCallback(progressA, progressB)
#                    print('progX',progressA,progressB)
                    continue

            # Possibly recluster the spectra and compute reference spectra
            if iteration == 0:
                pass
            elif curc > 0:
                if autoiterations:
                    notdone = unimproved <= automax
                    nds = notdone.sum()
                    curc = min(curc, int(nds))
                    labels = np.zeros(len(app)) - 1
                    if curc == nds:
                        labels[notdone] = range(0, nds)
                    elif curc > 1:
                        kmeans = sklearn.cluster.MiniBatchKMeans(curc)
                        labels[notdone] = kmeans.fit_predict(corrected[notdone,:])
                    else:
                        labels[notdone] = 0
                else:
                    if curc > 1:
                        kmeans = sklearn.cluster.MiniBatchKMeans(curc)
                        labels = kmeans.fit_predict(corrected)
                    else:
                        labels = np.zeros(len(app), dtype=int)

                if(len(ref) != curc):
                    ref = np.zeros((curc, len(wn)))
                for cl in range(curc):
                    sel = labels == cl
                    if sel.sum() == 0:
                        print('Info: empty cluster at %d, %d' % (iteration, cl))
                    else:
                        ref[cl,:] = corrected[sel].mean(0)
            else:
                # Mix old reference and corrected spectrum. This requires the clusters
                # to remain unchanged.
                if autoiterations:
                    labels[unimproved > automax] = -1 # Exclude all that are done already
                for cl in range(len(ref)):
                    sel = labels == cl
                    if sel.sum() > 0:
                        ref[cl,:] = .5 * corrected[sel].mean(0) + .5 * ref[cl,:]
            if plot:
                plt.plot(wn, ref.T, c=color[iteration], linewidth=.5)
            if progressPlotCallback:
                progressPlotCallback(ref, (iteration, iterations))
            ref[ref < 0] = 0

            for cl in range(len(ref)):
                ix = np.where(labels == cl)[0] # Indexes of spectra in this cluster
                if autoiterations:
                    ix = ix[unimproved[ix] <= automax]
                if ix.size:
                    model = compute_model(wn, ref[cl], n_components, a, d, bvals,
                                          konevskikh=konevskikh, linearcomponent=linearcomponent,
                                          variancelimit=pcavariancelimit)
                    if weights is None:
                        cons = np.linalg.lstsq(model.T, app[ix, :].T, rcond=None)[0]
                    else:
                        cons = np.linalg.lstsq(model.T * weights, app[ix, :].T * weights, rcond=None)[0]
                    corrs = app[ix] - cons[1:, :].T @ model[1:, :]
                    if renormalize:
                        corrs = corrs / cons[0, :, None]
                    resids = ((corrs - model[0, :])**2).sum(1)

                    if iteration == 0:
                        corrected = corrs
                        residuals = resids
                        nimprov = len(resids)
                    else:
                        improved = resids < residuals[ix]
                        iximp = ix[improved]  # Indexes of improved spectra
                        if autoiterations:
                            impmore = resids[improved] < residuals[iximp] * targetrelresiduals
                            unimproved[iximp[impmore]] = 0
                            unimproved[iximp[np.logical_not(impmore)]] += 1
                            unimproved[ix[np.logical_not(improved)]] += autoupadd
                        corrected[iximp, :] = corrs[improved, :]
                        residuals[iximp] = resids[improved]
                        nimprov = improved.sum()

                    if verbose:
                        print("iter %3d, cluster %3d (%5d px): avgres %7.3g  imprvd %4d  time %f" %
                              (iteration, cl, len(ix), resids.mean(), nimprov, monotonic()-startt))
                if progressCallback:
                    progressCallback(progressA + cl + 1, progressB)
            progressA += progstep
            if progressCallback and len(ref) < progstep:
                progressCallback(progressA, progressB)
#            print('progY',progressA,progressB)

    else:
        # For efficiency, compute the model from the input reference spectrum only once
        model = compute_model(wn, ref, n_components, a, d, bvals, konevskikh=konevskikh,
                              linearcomponent=linearcomponent, variancelimit=pcavariancelimit)

        if weights is None:
            cons = np.linalg.lstsq(model.T, app.T, rcond=None)[0]
        else:
            cons = np.linalg.lstsq(model.T * weights, app.T * weights, rcond=None)[0]
        corrected = app - cons[1:, :].T @ model[1:, :]
        if renormalize:
            corrected = corrected / cons[0, :, None]
        if autoiterations:
            residuals = ((corrected - model[0, :])**2).sum(1)

        if progressPlotCallback:
            progressPlotCallback(ref, (0, len(app) + 1))

        if verbose:
            print("all pixels, iter %2d  time %f" % (0, monotonic()-startt))
        if progressCallback:
            progressA += 1
            progressCallback(progressA, progressB)
        if iterations > 1:
            for s in range(len(app)):
                gc.collect()
                unimproved = 0
                ref = corrected[s, :]  # Corrected spectrum as new reference
                for iteration in range(1, iterations):
                    ref[ref < 0] = 0. # No negative values in reference spectrum
                    model = compute_model(wn, ref, n_components, a, d, bvals,
                                          konevskikh=konevskikh, linearcomponent=linearcomponent,
                                          variancelimit=pcavariancelimit)
                    if weights is None:
                        cons = np.linalg.lstsq(model.T, app[s], rcond=None)[0]
                    else:
                        cons = np.linalg.lstsq(model.T * weights, app[s] * weights[:, 0], rcond=None)[0]
                    corr = app[s] - cons[1:] @ model[1:, :]
                    if renormalize:
                        corr = corr / cons[0]
                    print("pixel %5d: iter %3d  residual %7.3g  " %
                          (s, iteration+1, ((corr - model[0, :])**2).sum()))
                    if autoiterations:
                        residual = ((corr - model[0, :])**2).sum()
                        if residual < residuals[s]:
                            corrected[s, :] = corr
                            unimproved =  unimproved + 1 if residual > residuals[s] * targetrelresiduals else 0
                            residuals[s] = residual
                        else:
                            unimproved += autoupadd
                        if unimproved > automax:
                            break
                    ref = corr
                if not autoiterations:
                    corrected[s, :] = corr
                    residual = ((corr / cons[0] - model[0, :])**2).sum()

                if verbose:
                    print("pixel %5d: iter %3d  residual %7.3g  time %f" %
                          (s, iteration+1, residual, monotonic()-startt))
                if progressCallback:
                    progressA += 1
                    progressCallback(progressA, progressB)
                if progressPlotCallback and len(app) < 50:
                    progressPlotCallback(ref, (s + 1, len(app) + 1))

    return corrected.squeeze() if squeeze else corrected


