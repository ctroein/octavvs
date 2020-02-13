#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 14:01:08 2020

@author: carl
"""

import os.path
import numpy as np
import scipy.signal, scipy.io

from .opusreader import OpusReader

class SpectralData:
    """
    The list of current files and the raw data loaded from a hyperspectral image
    file in the preprocessing GUI.
    """
    def __init__(self):
        self.foldername = '' # Root dir
        self.filenames = []  # With full paths
        self.curFile = ''    # The currently loaded file
        self.wavenumber = None # array in order high->low
        self.wmin = 800
        self.wmax = 4000
        self.raw = np.empty((0,0)) # data in order (pixel, wavenumber)
        self.wh = (0, 0)  # Width and height in pixels
        self.image = None # Image loaded from raw data file; tuple (bytes, format_str)

    def setWidth(self, w):
        try:
            w = int(w)
        except ValueError:
            w = 0
        if w <= 0:
            return False
        h = int(self.raw.shape[0] / w)
        if w * h != self.raw.shape[0]:
            return False
        self.wh = (w, h)
        return True

    def setHeight(self, h):
        if self.setWidth(h):
            self.wh = (self.wh[1], self.wh[0])
            return True
        return False

    def readMatrix(self, filename):
        """
        Read data from a file, with some error checking. The object is modified
        only if the file is successfully loaded.
        """
        wh = None
        image = None
        fext = os.path.splitext(filename)[1].lower()
#        opusformat = False
        if fext in ['.txt', '.csv', '.mat']:
            if fext == '.mat':
                s = scipy.io.loadmat(filename)
                info = scipy.io.whosmat(filename)
                ss = s[info[0][0]]
                if 'wh' in s:
                    wh = s['wh'].flatten()
            else:
                ss = np.loadtxt(filename)

            if ss.ndim != 2 or ss.shape[0] < 10 or ss.shape[1] < 2:
                raise RuntimeError('file does not appear to describe an FTIR image matrix')
            d = -1 if ss[0,0] < ss[-1,0] else 1
            raw = ss[::d,1:].T
            wn = ss[::d,0]
        else:
            reader = OpusReader(filename)
            raw = reader.AB
            wn = reader.wavenum
            wh = reader.wh
            image = reader.image

        if (np.diff(wn) >= 0).any():
            raise RuntimeError('wavenumbers must be sorted')
        npix = raw.shape[0]
        if wh is not None:
            if len(wh) != 2:
                raise RuntimeError('Image size in "wh" must have length 2')
            wh = (int(wh[0]), int(wh[1]))
            if wh[0] * wh[1] != npix:
                raise RuntimeError('Image size in "wh" does not match data size')
            self.wh = wh
        elif npix != self.wh[0] * self.wh[1]:
            res = int(np.sqrt(npix))
            if npix == res * res:
                self.wh = (res, res)
            else:
                self.wh = (npix, 1)
        self.raw = raw
        self.wavenumber = wn
        self.wmin = self.wavenumber.min()
        self.wmax = self.wavenumber.max()
        self.image = image
        self.curFile = filename

