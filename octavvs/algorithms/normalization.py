#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 23:44:05 2020

@author: carl
"""

import numpy as np

def normalize_spectra(method, y, wn=None, **kwargs):
    """
    Normalize the spectra in the matrix y (pixel,wavenum) with the given method

    Parameters
    ----------
    method : str
        Normalization method; one of: 'mean', 'area', 'wn', 'max', 'n2'
    y : array
        spectra in order (spectrum, wavenum)
    wn : array, optional
        wavenumber array; needed for the 'area' and 'wn' methods.
        The default is None.
    wavenum : float
        wavenumber to normalize at; used by 'wn'

    Raises
    ------
    ValueError
        if give an undefined method.

    Returns
    -------
    array
        Array of normalized spectra in the same format as y.
    """

    if method == 'mean':
        return (y.T / y.mean(axis=1)).T
    elif method == 'area':
         return (y.T / -np.trapz(y, wn, axis=1)).T
    elif method == 'wn':
        idx = (np.abs(wn - kwargs['wavenum'])).argmin()
        return (y.T / y[:, idx]).T
    elif method == 'max':
        return (y.T / y.max(axis=1)).T
    elif method == 'n2':
        return (y.T / np.sqrt((y * y).mean(axis=1))).T
    raise ValueError('undefined normalization method ' + method)

def normalize_features(y, method):
    """
    Normalize intensities or other features.

    Parameters
    ----------
    y : array(nsamples, nfeatures)
        Data matrix, e.g. raw spectra or decomposition components.
    method : str
        One of 'none', 'mean1', 'mean0', 'mean0var1'

    Returns
    -------
    Normalized data.
    """
    if method == 'none':
        return y
    elif method == 'mean1':
        means = y.mean(0)
        means[means == 0] = 1
        return y / means
    elif method == 'mean0':
        return y - y.mean(0)
    elif method == 'mean0var1':
        y = y - y.mean(0)
        sds = y.std(0)
        sds[sds == 0] = 1
        return y / sds
    raise ValueError('undefined normalization method ' + method)

