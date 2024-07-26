#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 16:24:50 2020

@author: carl
"""

import os.path
from pkg_resources import resource_filename
import numpy as np
from scipy.interpolate import PchipInterpolator
#from scipy.io import loadmat
from pymatreader import read_mat
import pandas as pd


def load_reference(wn, what=None, filename=None):
    """
    Load and normalize a spectrum from a Matlab file, interpolating at
    the given points. The reference is assumed to cover the entire range
    of wavenumbers.

    Parameters
    ----------
    wn : ndarray
        wavenumbers at which to get the spectrum.
    what : string, optional
        Type of reference to get, corresponding to a file in the
        'reference' directory.
    filename : string, optional
        The name of an arbitrary Matlab/CSV file to load data from.
        For .mat, the data must be in a matrix called AB, with wavenumbers
        in the first column.
        For CSV, the wavenumbers must be in the first row and the spectrum
        in the second row.

    Returns
    -------
    Spectrum at the points given by wn.

    """
    if (what is None) == (filename is None):
        raise ValueError("Either 'what' or 'filename' must be specified")
    if what is not None:
        filename = resource_filename(
            'octavvs.reference_spectra', what + ".mat")
    if os.path.splitext(filename)[1].lower() == ".mat":
        ref = read_mat(filename)['AB']
    else:
        ref = pd.read_csv(filename, sep=None, comment='#', engine="python",
                          header=None)
        ref = ref.to_numpy()
        if ref.shape[0] < ref.shape[1]:
            ref = ref.T
        if ref.shape[1] != 2:
            raise RuntimeError(
                "CSV reference error: wavenumbers must be in the"
                " first row/col and the spectrum in the second row/col."
                f" (got: {ref.shape})")
    # Handle the case of high-to-low since the interpolator requires low-to-high
    d = 1 if ref[0,0] < ref[-1,0] else -1
    ref = PchipInterpolator(ref[::d,0], ref[::d,1])(wn)
    return ref #/ ref.max()

def nonnegative(y, fracspectra=0, fracvalues=0):
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

def pixels_fit(npixels, wh):
    "Check that n pixels fit snugly within w * h"
    return wh[0] * (wh[1] - 1) < npixels <= wh[0] * wh[1]


def subtract_background(y, background):
    """
    Subtract such an amount of a background spectrum that the resulting
    spectra become maximally smooth.

    Parameters
    ----------
    y : ndarray
        1D or 2D array of spectra.
    background : ndarray
        1D reference spectrum of the same length as y.

    Returns
    -------
    ret : ndarray
        Input spectra minus background.
    """

    dh = background[:-1] - background[1:]
    dy = y[:,:-1] - y[:,1:]
    dh2 = np.cumsum(dh * dh)
    dhdy = np.cumsum(dy * dh, 1)

    az = dhdy[:,-1] / dh2[-1]
    ret = y - az[:, None] @ background[None, ...]
    return ret
