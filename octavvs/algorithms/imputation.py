#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 27 00:23:22 2021

@author: carl
"""

import numpy as np
import math

def impute_from_neighbors(data, spatial=True, spectral=False):
    """
    Impute missing intensities as mean of surrounding values in the image.

    The mean of nearest neighbor pixels replaces all NaNs that have at least
    one non-NaN neighbor, and this is iterated until all NaNs are replaced.

    Parameters
    ----------
    data : numpy.ndarray
        Intensities stored as data[x, y, ..., wavenumber].
    spatial : bool
        Use spatial nearest neighbors (n-1 first dimensions).
    spectral : bool
        Use spectral nearest neighbors (last dimension).

    Returns
    -------
    data : numpy.ndarray
        Transformed data.

    """
    todo = np.argwhere(~np.isfinite(data))
    if not len(todo):
        return data
    if len(todo) == data.size:
        raise ValueError('Imputation requires at least one non-NaN in input')
    wh = data.shape
    data = data.copy()
    if spatial and spectral:
        sumdims = range(len(wh))
    elif spatial:
        sumdims = range(len(wh) - 1)
    elif spectral:
        sumdims = [-1]
    else:
        raise ValueError('Imputation must use spatial and/or spectral neighbors')

    while len(todo):
        # print('todo:', len(todo))
        done = []
        donev = []
        stilltodo = []
        for xy in todo:
            nsum = 0.
            nn = 0
            for d in sumdims:
                c = xy[d]
                if c > 0:
                    xy[d] = c - 1
                    v = data[tuple(xy)]
                    if math.isfinite(v):
                        nsum = nsum + v
                        nn = nn + 1
                if c + 1 < wh[d]:
                    xy[d] = c + 1
                    v = data[tuple(xy)]
                    if math.isfinite(v):
                        nsum = nsum + v
                        nn = nn + 1
                xy[d] = c
            if nn:
                done.append(tuple(xy))
                donev.append(nsum / nn)
            else:
                stilltodo.append(xy)
        if not len(done):
            # Should perhaps raise an exception instead
            break
        data[tuple(np.asarray(done).T)] = donev
        todo = stilltodo
    return data
