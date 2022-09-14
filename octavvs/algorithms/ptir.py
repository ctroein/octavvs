#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 02:56:27 2021

@author: carl
"""

import numpy as np
import itertools
from statsmodels.multivariate import pca as smpca
from scipy.optimize import minimize

def correct_mirage(wn, data, breaks=[(973,977), (1201.5,1205.5), (1449,1453)],
                   endpoints=[[4,4],[4,4],[4,4]], slopefactor=[1, 1, 1],
                   pca=False, pca_ncomp=[2, 2, 2],
                   soft_limits=False, sl_level=[1, 1, 1],
                   chipweight=[0, .3, .35, .35]):
    """
    Correct for drift in quantum cascade laser intensity between the four
    regions of mIRage PTIR spectra.

    Near the end of each of the regions, spectra are be assumed to have the
    same level except for an overall per-spectrum intensity level. The
    normalization factors at the region ends are linearly interpolated
    within each of the regions, then applied. Finally, values in the
    break regions are interpolated anew from the surrounding points.

    'A first chip provides IR light from 770 cm-1 to 975.8 cm-1; a
    second chip from 975.9 cm-1 to 1204 cm-1; a third chip from
    1204.1 cm-1 to 1451.2 cm-1 and a fourth chip from 1451.3
    cm-1 to 1802 cm-1.'

    Parameters
    ----------
    wn : array
        wavenumbers, sorted
    data : array(pixel, wn)
        raw mIRage data
    breaks : list of pairs, optional
        regions where the spectra are interpolated between the QCL chips
    endpoints: int or array-like, optional
        number of wavenumbers to consider when normalizing levels at
        the breaks between regions; can be given as a single int, a triplet
        or a triplet of pairs (for the left and right side of the break).
    slopefactor : float or array-like, optional
        the extent to which the slope of points near a break are used when
        inferring the level at the break point; useful range is 0 to 1.
    pca:
    pca_ncomp : int or array-like, optional
        number of components for PCA filter of intensity levels around
        each break; useful for reducing noise in the estimation.
    soft_limits : bool, optional
        limit the range of correction factors based on their estimated
        distribution and the reliability of individual spectra.
    sl_level : float or array-like, optional
        base level of the soft limits (in stddevs).
    chipweight : list of floats, optional
        weights of the chip regions when normalizing correction levels.

    Returns
    -------
    data : array
        corrected data
    shifts : array(npixels, nbreaks)
        log rescaling factor for the spectra at each of the breaks.

    """
    data = data.copy()
    if not len(data):
        return data, None
    flipwn = wn[0] > wn[-1]
    if flipwn:
        data = data[:, ::-1]
        wn = wn[::-1]

    def linreg(x, y):
        xm = x.mean()
        ym = y.mean(1)
        rx = x - xm
        s = (rx * (y.T - ym).T).sum(1) / (rx * rx).sum()
        return xm, ym, s

    cuts = [0]
    logshifts = []
    weights = []
    # endp = (np.ones((len(breaks), 2), dtype=int).T * np.transpose(endpoints)).T
    endp = np.broadcast_to(np.transpose(endpoints), (2, len(breaks))).T
    slopef = np.broadcast_to(np.transpose(slopefactor), (2, len(breaks))).T
    if pca:
        ncomp = np.broadcast_to(pca_ncomp, len(breaks))
    for j, bw in enumerate(breaks):
        # First index (in, after) the break region
        b = np.searchsorted(wn, bw[0]), np.searchsorted(wn, bw[1])
        cuts.extend(b)
        c = [max(0, b[0] - endp[j, 0]), b[0],
             b[1], min(b[1] + endp[j, 1], len(wn))]
        # Possibly do PCA noise reduction before computing slope
        dat = np.maximum(
            0, data[:, list(range(c[0], c[1])) + list(range(c[2], c[3]))])
        if pca and ncomp[j] and ncomp[j] < len(dat):
            dat = smpca.PCA(dat, ncomp=ncomp[j]).projection
            dat = np.maximum(0, dat)

        dleft = dat[:, :(c[1] - c[0])]
        dright = dat[:, (c[1] - c[0]):]
        wb = linreg(wn[c[0]:c[1]], dleft)
        we = linreg(wn[c[2]:c[3]], dright)
        wid = (we[0] - wb[0]) / 2
        # Compute desired step with some provision for near-zero values
        epsilon = 1e-3
        lb = wb[1] + slopef[j, 0] * wid * wb[2]
        le = we[1] - slopef[j, 1] * wid * we[2]
        logshifts.append(np.log(np.maximum(epsilon * lb.mean(), lb)) -
                         np.log(np.maximum(epsilon * le.mean(), le)))

        weps = .1
        weights.append(np.minimum(
            dleft.mean(1) / (np.std(dleft, 1) + weps * dleft.mean()),
            dright.mean(1) / (np.std(dright, 1) + weps * dright.mean())))

    cuts.append(len(wn))
    cuts = np.array(cuts).reshape((-1, 2))
    logshifts = np.array(logshifts)

    means = None
    if soft_limits:
        sl_level = np.broadcast_to(sl_level, len(breaks))
        sl_level = np.maximum(sl_level, 0)
        # Weight/importance: values above a fraction of the mean
        # weights = np.array(weights)
        weights = np.array(weights)
        # Compute weighted mean and stddev
        wsum = weights.sum(1)
        means = ((logshifts * weights).sum(1) / wsum)[:, None]
        sd = (((logshifts - means) ** 2 * weights).sum(1) / wsum) ** .5
        # or
        # sd = (((logshifts - means) ** 2 * weights**2).sum(1) / wsum**2) ** .5

        lim = (sd * sl_level)[:, None]
        logshifts = logshifts - means
        logshifts = means + logshifts / (1 + (logshifts / lim)**2)**.5

    # Turn steps into scale factors for the regions
    chipscale = np.zeros((len(breaks) + 1, len(data)))
    for j, ls in enumerate(logshifts):
        chipscale[j+1, :] = chipscale[j, :] + logshifts[j]
    chipscale = np.exp(chipscale)
    # Normalize levels to preserve weighted mean level
    chipscale = chipscale / (chipweight @ chipscale)

    # Rescale data and interpolate in gaps
    pc = None
    for j, c in enumerate(cuts):
        data[:, c[0]:c[1]] *= chipscale[j, :, None]
        if j:
            data[:, pc:c[0]] = (np.linspace(data[:, pc-1], data[:, c[0]],
                                           num=c[0]-pc+2)[1:-1]).T
        pc = c[1]

    if flipwn:
        data = data[:,::-1]
    return np.maximum(0, data), (logshifts, weights)


def optimize_mirage_pca(wn, data, callback=None, **cm_args):

    def mirage_variance(x=None):
        corr = correct_mirage(wn, data, **cm_args)[0]
        normsd = np.std(corr / corr.mean(1)[:, None], 0).mean()
        return normsd

    nb = len(cm_args['breaks'])
    pca_ncomps = itertools.product(*([range(0, 4)] * nb))
    bestsol = None
    cm_args['pca'] = True
    # cm_args['soft_limits'] = False
    for i, pca_ncomp in enumerate(pca_ncomps):
        cm_args['pca_ncomp'] = pca_ncomp
        v = mirage_variance()
        if bestsol is None or v < bestsol[0]:
            bestsol = (v, pca_ncomp)
        if callback is not None:
            callback(i)
    return bestsol

def optimize_mirage_sl(wn, data, callback=None, **cm_args):
    nb = len(cm_args['breaks'])

    lims = (.001, 10)

    def mirage_variance(x):
        x = np.maximum(lims[0], np.minimum(lims[1], x))
        cm_args['sl_level'] = x
        corr = correct_mirage(wn, data, **cm_args)[0]
        normsd = np.std(corr / corr.mean(1)[:, None], 0).mean()
        return normsd

    init = np.ones((nb + 1, nb)) * 2.
    init[1:, :] -= np.eye(nb)
    cm_args['soft_limits'] = True
    optr = minimize(mirage_variance, method='Nelder-Mead', x0=init[0],
                    tol=1e-2, callback=callback,
                    options={'xatol': .05, 'initial_simplex': init})
    if not optr.success:
        raise RuntimeError('Optimization failed: ' + optr.message)
    return optr.fun, np.maximum(lims[0], np.minimum(lims[1], optr.x))

