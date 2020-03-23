#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Carl Troein

Atmospheric and scattering correction

"""

import gc
from time import monotonic
import numpy as np
import sklearn.linear_model
import sklearn.cluster
#import statsmodels.multivariate.pca
from scipy.interpolate import PchipInterpolator
from scipy.signal import hilbert
#from scipy.io import loadmat, savemat

from .util import pca_nipals, find_wn_ranges

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

def compute_model(wn, ref, n_components, a, d, bvals, konevskikh=True, linearcomponent=False,
                  variancelimit = None):
    """
    Compute the extinction matrix for resonant Mie scattering, then PCA transform it.
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
#        alpha_0 = 1e-2 * np.linspace(d[0] * (a[0]-1), d[-1] * (a[-1]-1), len(a))
#        gamma = .25 * 2 * np.log(10) / np.pi * np.linspace(1 / alpha_0[0], 1 / alpha_0[-1], len(alpha_0))
        # Solheim's distributions of alpha_0 and gamma
        alpha_0 = 1e-2 * d * (a - 1)
        gamma = .25  * 2 * np.log(10) / np.pi / alpha_0
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

#    savemat('everything_python.mat', {'Q':Q})

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
#        for i in range(1, n_nonpca):
#            model[i,:] = model[i,:] - refn * (model[i,:] @ refn)
#        for i in range(1, n_nonpca):
#            w = model[i, :] - np.sum(np.dot(model[i, :], b) * b for b in model[0:i, :])
#            model[i,:] = w / np.sqrt(w @ w)

#    savemat('everything_after-p.mat', locals())

    # Orthogonalization of the model to improve numeric stability (doing it after PCA is only
    # marginally slower)
#    for i in range(len(model)):
#        v = model[i, :]
#        w = v - np.sum(np.dot(v, b) * b for b in model[0:i, :])
#        model[i,:] = w / np.linalg.norm(w)

#    print('model',ref.sum(), model.sum())
    return model

def stable_rmiesc_clusters(iters, clusters):
    """
    Make a cluster size scheme for reliable convergence in rmiesc.
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
#    cc = np.zeros(max(2, iters) * 3 // 2, dtype=np.int)
#    cc[:iters] = clusters
    return cc

def rmiesc(wn, app, ref, n_components=7, iterations=10, clusters=None,
           pcavariancelimit=None,
           verbose=False, a=np.linspace(1.1, 1.5, 10), d=np.linspace(2.0, 8.0, 10),
           bvals=10, progressCallback = None, progressPlotCallback=None,
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
    progressCallback(int a, int b): callback function called to indicated that the processing
        is complete to a fraction a/b.
    progressPlotCallback(array ref, (a, b)): like the above with a (cluster) reference spectrum
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

    if progressCallback:
        # Compute the number of progress steps
        progressA = 0
        if clusters is None:
            progressB = 1 + (iterations > 1) * len(app)
        else:
            progressB = 1
            prev = 1
            for cl in clusters[1:]:
                if cl > 0:
                    prev = cl
                progressB += prev
    startt = monotonic()

    corrected = None # Just to get rid of warnings in the editor; will be set on iteration 0

    # Set parameters for automatic iteration control
    autoupadd = 1 # Residual going up counts as residual going down too little this many times
    automax = 5   # Stop when residual has gone down too little this many times

    if clusters is not None:
        # Cluster mode: In each iteration, after correcting all the spectra, cluster them.
        # Then take the mean of the corrected spectra in each cluster as the new reference
        # for that cluster in the next iteration. Spectra should still have their own reference
        # in the EMSC step. To avoid having to loop over all the spectra in a cluster when
        # solving that equation system, we remove the projection of the reference from the
        # apparent spectra.

        if progressPlotCallback:
            progressPlotCallback(ref, (0, iterations))
        projs = ((app @ ref) / (ref @ ref))[:,None] * ref
        app_deref = app - projs
        model = compute_model(wn, ref, n_components, a, d, bvals,
                              konevskikh=konevskikh, linearcomponent=linearcomponent,
                              variancelimit=pcavariancelimit)
        if weights is None:
            cons = np.linalg.lstsq(model.T, app_deref.T, rcond=None)[0]
        else:
            cons = np.linalg.lstsq(model.T * weights, app_deref.T * weights, rcond=None)[0]
        corrected = app - cons[1:].T @ model[1:]
        residuals = ((corrected - projs)**2).sum(1) #We compare to the previous correction, not the reference
        ref = []    # Force init
        if autoiterations:
            unimproved = np.zeros(len(app), dtype=int)

        if progressCallback:
            progstep = 1 # Current progress bar step size
            progressA += progstep
            progressCallback(progressA, progressB)

        for iteration in range(1, iterations):
            corrected[corrected < 0] = 0

            curc = clusters[iteration] # Current cluster size setting
            if curc > 0:
                progstep = curc

            # Possibly recluster the spectra and compute reference spectra
            if curc > 0:
                # Find the remaining spectra to process
                if autoiterations:
                    if curc != clusters[iteration-1]:
                        unimproved = np.zeros(len(app), dtype=int)
                        nds = len(app)
                        notdone = range(nds)
                    else:
                        notdone = np.where(unimproved <= automax)[0]
                        nds = len(notdone)
                    # Skip this iteration if every spectrum has stopped improving
                    if nds == 0:
                        progressA += progstep
                        if progressCallback:
                            progressCallback(progressA, progressB)
                        continue
                else:
                    nds = len(app)
                    notdone = range(nds)

                # Find the clusters of spectra
                curc = min(curc, nds)
                if curc == nds:
                    cindices = [np.array([i]) for i in notdone] # Each index in its own little array
                elif curc > 1:
                    kmeans = sklearn.cluster.MiniBatchKMeans(curc)
                    labels = np.zeros(len(app)) - 1
                    labels[notdone] = kmeans.fit_predict(corrected[notdone,:])
                    cindices = [ np.where(labels == i)[0] for i in range(curc) ] # Quadratic but simple
                else:
                    cindices = [ np.array(notdone) ] # One big cluster

                # Compute cluster references
                if(len(ref) != curc):
                    ref = np.zeros((curc, len(wn)))
                empties = 0
                for cl in range(curc):
                    if len(cindices[cl]):
                        ref[cl,:] = corrected[cindices[cl]].mean(0)
                    else:
                        empties += 1
                if empties:
                    print('Info: %d empty cluster(s) at iteration %d' % (empties, iteration))
            else:
                # Mix old reference and corrected spectrum. This requires the clusters
                # to remain unchanged.
                for cl in range(len(ref)):
                    if len(cindices[cl]):
                        # Stop processing clusters only when all members are done
                        if autoiterations and (unimproved[cindices[cl]] > automax).all():
                            cindices[cl] = []
                        else:
                            ref[cl,:] = .5 * corrected[cindices[cl]].mean(0) + .5 * ref[cl,:]

            w = (app * corrected).sum(1) / (corrected * corrected).sum(1)
            projs = w[:,None] * corrected
            app_deref = app - projs

            if progressPlotCallback:
                progressPlotCallback(ref, (iteration, iterations))

            for cl in range(len(ref)):
                ix = cindices[cl] # Indexes of spectra in this cluster
                if len(ix) > 0 and autoiterations:
                    ix = ix[unimproved[ix] <= automax]
                if len(ix) == 0:
                    continue

                model = compute_model(wn, ref[cl], n_components, a, d, bvals,
                                      konevskikh=konevskikh, linearcomponent=linearcomponent,
                                      variancelimit=pcavariancelimit)
                # Including the reference in the model gives results in agreement with the
                # unclustered approach.
                if weights is None:
                    cons = np.linalg.lstsq(model.T, app_deref[ix].T, rcond=None)[0]
                else:
                    cons = np.linalg.lstsq(model.T * weights, app_deref[ix].T * weights, rcond=None)[0]
                corrs = app[ix] - cons[1:].T @ model[1:]

                # We compare to the previous correction, not the reference
                resids = ((corrs - projs[ix])**2).sum(1)
                improved = resids < residuals[ix]

                if renormalize:
                    corrs = corrs / (1 + cons[0,:,None])

                if autoiterations:
                    iximp = ix[improved]  # Indexes of improved spectra
                    impmore = resids[improved] < residuals[iximp] * targetrelresiduals
                    unimproved[iximp[impmore]] = 0
                    unimproved[iximp[np.logical_not(impmore)]] += 1
                    unimproved[ix[np.logical_not(improved)]] += autoupadd

                    corrected[iximp, :] = corrs[improved, :]
                    residuals[iximp] = resids[improved]
#                    corrected[ix,:] = corrs
#                    residuals[ix] = resids
                else:
                    corrected[ix,:] = corrs
                    residuals[ix] = resids

                nimprov = improved.sum()

                if verbose:
                    print("iter %3d, cluster %3d (%5d px): avgres %7.3g  imprvd %4d  time %f" %
                          (iteration, cl, len(ix), resids.mean(), nimprov, monotonic()-startt))
                if progressCallback:
                    progressCallback(progressA + cl + 1, progressB)
            if progressCallback:
                progressA += progstep
                progressCallback(progressA, progressB)
            gc.collect()    # Because my old laptop was unhappy with RAM usage otherwise

    else:
        if renormalize:
            # More aggressive stopping if residual rises
            autoupadd = 3  # Residual going up counts as residual going down too little this many times
            automax = 3    # Stop when residual has gone down too little this many times
        ref = ref.copy()
        ref[ref < 0] = 0
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
                          (s, iteration, residual, monotonic()-startt))
                if progressCallback:
                    progressA += 1
                    progressCallback(progressA, progressB)
                if progressPlotCallback and len(app) < 50:
                    progressPlotCallback(ref, (s + 1, len(app) + 1))

    return corrected.squeeze() if squeeze else corrected


