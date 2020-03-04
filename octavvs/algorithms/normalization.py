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
