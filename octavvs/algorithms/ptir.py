#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  4 02:56:27 2021

@author: carl
"""

import numpy as np

from .util import find_wn_ranges



def normalize_mirage(wn, data, breaks=[(922,926), (1202,1206), (1448,1452)],
                     endpoints=6, slopefactor=.5):
    """
    Correct for drift in quantum cascade laser intensity between the four
    regions of mIRage PTIR spectra.

    Near the end of each of the regions, spectra are be assumed to have the
    same level except for an overall per-spectrum intensity level. The
    normalization factors at the region ends are linearly interpolated
    within each of the regions, then applied. Finally, values in the
    break regions are interpolated anew from the surrounding points.

    Parameters
    ----------
    wn : array
        wavenumbers, sorted
    data : array(pixel, wn)
        raw mIRage data
    breaks : list of 2-ples, optional
        regions where the spectra are interpolated between the QCL chips
    endpoints: int
        number of wavenumbers to consider when normalizing levels at
        the edge of each region.

    Returns
    -------
    data : array
        corrected data
    scale : array(pixels, 8)
        rescaling factor for the spectra at start and end of each of the
        four regions defined by the breaks.

    """
    data = data.copy()
    if not len(data):
        return data, None
    flipwn = wn[0] > wn[-1]
    if flipwn:
        data = data[:, ::-1]
        wn = wn[::-1]
    breaks = find_wn_ranges(wn, np.array(breaks))
    cuts = np.concatenate(([0], breaks.flatten(), [len(wn)])).reshape((-1, 2))

    # Straight baseline of each segment. shape=(ncuts, 2, nspectra)
    scale = np.zeros(cuts.shape + (len(data),))
    # Overall level in each segment
    dsum = np.zeros(len(cuts))
    slopes = np.zeros(cuts.shape)
    cutpoints = np.zeros(cuts.shape)

    def linreg(x, y):
        xm = x.mean()
        ym = y.mean(1)
        rx = x - xm
        s = (rx * (y.T - ym).T).sum(1) / (rx * rx).sum()
        return xm, ym, s

    for i in range(len(cuts)):
        cb = cuts[i][0]
        ce = cuts[i][1]
        cwidth = min(endpoints, (ce - cb) // 2)

        wb = linreg(wn[cb:cb+cwidth], np.abs(data[:, cb:cb+cwidth]))
        we = linreg(wn[ce-cwidth:ce], np.abs(data[:, ce-cwidth:ce]))

        cutpoints[i, :] = [wb[0], we[0]]
        # sc = np.maximum([wb[1], we[1]], 1)
        sc = [wb[1], we[1]]
        scale[i,:,:] = sc
        slopes[i, :] = np.array([wb[2], we[2]]).mean(1)
        # need to handle negative values here!
        # dsum[i] = np.abs(data[:, cb:ce]).sum()
        dsum[i] = np.maximum(data[:, cb:ce], 0).sum()
    # Mean level of all spectra in each of the cut points
    means = scale.mean(2)
    # Make the mean levels identical on both sides of the cuts
    for i in range(len(means)-1):
        mm = min(means[i][1], means[i+1][0])
        ds = (slopes[i+1, 0] + slopes[i, 1]) / mm * (
            wn[cuts[i+1, 0]] - wn[cuts[i, 1]]) * slopefactor
        ds = max(-.5, min(.5, ds))
        means[i][1] = mm * (1 - ds)
        means[i+1][0] = mm * (1 + ds)
    # print('means', means)
    scale = (scale.T / means.T).T
    weights = dsum / dsum.mean()
    scale = scale / ((scale.min(1).T * weights).mean(1))
    # scale = scale / ((scale.mean(1).T * weights).mean(1))

    for i in range(len(cuts)):
        cb = cuts[i][0]
        ce = cuts[i][1]
        data[:, cb:ce] /= np.linspace(scale[i][0], scale[i][1], ce-cb, axis=-1)
        if i:
            pce = cuts[i-1][1]
            data[:, pce:cb] = np.linspace(data[:, pce-1], data[:, cb], cb-pce+1,
                                          endpoint=False, axis=-1)[:,1:]
    scale = scale.reshape((-1, len(data))).T
    if flipwn:
        data = data[:,::-1]
    return data, scale
