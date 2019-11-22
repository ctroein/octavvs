#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 14:29:38 2019

@author: carl
"""

import numpy as np
import scipy.signal, scipy.io

class ImageData:
    """
    The list of current files and the raw data loaded from a Matlab file in the preprocessing GUI.
    """
    def __init__(self):
        self.curFile = ''    # The currently loaded file
        self.wavenumber = None # np.array
        self.wmin = 800
        self.wmax = 4000
        self.raw = np.empty((0,0)) # data in order (pixel, wavenumber)
        self.wh = (0, 0)  # Width and height in pixels

    def readmat(self, filename):
        """
        Read data from a file, with some error checking. The object is modified
        only if the file is successfully loaded.
        """
        s = scipy.io.loadmat(filename)
        info = scipy.io.whosmat(filename)
        ss = s[info[0][0]]
        if ss.ndim != 2 or ss.shape[0] < 10 or ss.shape[1] < 1:
            raise RuntimeError('file does not appear to describe an FTIR image matrix')
        d = -1 if ss[0,0] < ss[-1,0] else 1
        wn = ss[::d,0]
        if (np.diff(wn) >= 0).any():
            raise RuntimeError('wavenumbers must be sorted')
        npix = ss.shape[1] - 1
        res = int(np.sqrt(npix))
#        if npix != res*res:
#            raise RuntimeError('non-square images not implemented')
        self.wavenumber = wn
        self.raw = ss[::d,1:].T
        self.wh = (res, npix//res)
        self.wmin = self.wavenumber.min()
        self.wmax = self.wavenumber.max()
        self.curFile = filename
