#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 16:21:46 2020

@author: carl
"""

import numpy as np
from scipy.signal import savgol_filter, tukey

from . import baseline
from .util import load_reference, find_wn_ranges


def cut_wn(wn, y, ranges):
    """
    Cut a set of spectra, leaving only the given wavenumber range(s).

    Parameters
    ----------
    wn: array of wavenumbers, sorted in either direction
    y: array of spectra, shape (..., wavenumber)
    ranges: list or numpy array of shape (..., 2) with desired wavenumber
    ranges in pairs (low, high)
    Returns: (wavenumbers, spectra) with data in the given wavenumber ranges
    """
    if isinstance(ranges, list):
        ranges = np.array(ranges)
    inrange = lambda w: ((w >= ranges[...,0]) & (w <= ranges[...,1])).any()
    ix = np.array([inrange(w) for w in wn])
    return wn[ix], y[...,ix]

def atmospheric(wn, y, atm=None, cut_co2 = True, extra_iters=5,
                extra_factor=0.25, smooth_win=9, progressCallback = None):
    """
    Correct for atmospheric gases.

    Apply atmospheric correction to multiple spectra, subtracting as much
    of the atompsheric spectrum as needed to minimize the sum of squares of
    differences between consecutive points in the corrected spectra.
    Each supplied range of wavenumbers is corrected separately.

    Parameters
    ----------
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

    Returns
    -------
        Tuple of (spectra after correction, array of correction factors;
                  shape (spectra,ranges))
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

    # ranges: numpy array (n, 2) of n non-overlapping wavenumber ranges
    # (typically for H2O only), or None
    # extra_winwidth: width of the window (in cm-1) used to locally
    # reshape the atm spectrum
    ranges = [[399, 776], [1300, 2100], [3410, 3960], [2190, 2480]]
    do_ranges = np.ones(len(ranges), dtype=bool)
    n_gas = np.array([0, 0, 0, 1])
    extra_winwidth = [100, 30, 150, 40]
    if cut_co2:
        do_ranges[n_gas == 1] = False

    if ranges is None:
        ranges = np.array([0, len(wn)])
    else:
        ranges = find_wn_ranges(wn, ranges)

    for i, (p, q) in enumerate(ranges):
        if not do_ranges[i]:
            continue
        if q - p < 2:
            do_ranges[i] = False
            continue
        # atm[p:q] -= baseline.straight(wn[p:q], atm[p:q])
        if np.abs(atm[p:q]).sum() == 0:
            do_ranges[i] = False

    savgolwin = 1 + 2 * int(smooth_win * (len(wn) - 1) / np.abs(wn[0] - wn[-1]))

    corr_ranges = do_ranges.sum()
    if progressCallback:
        progressA = 0
        progressB = 1 + corr_ranges * (extra_iters + (1 if savgolwin > 1 else 0))
        progressCallback(progressA, progressB)

    dh = atm[:-1] - atm[1:]
    dy = y[:,:-1] - y[:,1:]
    dh2 = np.cumsum(dh * dh)
    dhdy = np.cumsum(dy * dh, 1)
    for i, (p, q) in enumerate(ranges[do_ranges]):
        r = q-2 if q <= len(wn) else q-1
        az = ((dhdy[:,r] - dhdy[:,p-1]) / (dh2[r] - dh2[p-1])
                    ) if p > 0 else (dhdy[:,r] / dh2[r])
        y[:, p:q] -= az[..., None] @ atm[None, p:q]

    if progressCallback:
        progressA += 1
        progressCallback(progressA, progressB)

    for pss in range(extra_iters):
        for i, (p, q) in enumerate(ranges[do_ranges]):
            window = 2 * int(extra_winwidth[i] * (len(wn) - 1) / np.abs(wn[0] - wn[-1]))
            winh = (window+1)//2
            dy = y[:,:-1] - y[:,1:]
            dhdy = np.cumsum(dy * dh, 1)
            aa = np.zeros_like(y)
            aa[:,1:winh+1] = dhdy[:,1:window:2] / np.maximum(dh2[1:window:2], 1e-8)
            aa[:,1+winh:-winh-1] = (dhdy[:,window:-1] -
              dhdy[:,:-1-window]) / np.maximum(dh2[window:-1] - dh2[:-1-window], 1e-8)
            aa[:,-winh-1:-1] = (dhdy[:,-1:] -
              dhdy[:,-1-window:-1:2]) / np.maximum(dh2[-1] - dh2[-1-window:-1:2], 1e-8)
            aa[:, 0] = aa[:, 1]
            aa[:, -1] = aa[:, -2]
            aa = savgol_filter(aa, window + 1, 3, axis=1)
            y[:, p:q] -= extra_factor * aa[:, p:q] * atm[p:q]
            if progressCallback:
                progressA += 1
                progressCallback(progressA, progressB)

    if savgolwin > 1:
        for i, (p, q) in enumerate(ranges[do_ranges]):
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
    if cut_co2:
        do_ranges[n_gas == 1] = True
    for i, (p, q) in enumerate(ranges[do_ranges]):
        if q > p:
            corr = np.abs(yorig[:, p:q] - y[:, p:q]).sum(1) / np.maximum(
                np.abs(yorig[:, p:q]), np.abs(y[:, p:q])).sum(1)
            g = n_gas[do_ranges][i]
            corrs[g] += corr.mean()
            ncorrs[g] += 1
    corrs[ncorrs > 1] = corrs[ncorrs > 1] / ncorrs[ncorrs > 1]

    return (y.squeeze() if squeeze else y), corrs
