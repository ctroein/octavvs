#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 14:01:08 2020

@author: carl
"""

import os.path
import numpy as np
import scipy.signal, scipy.io
import pandas as pd
from pymatreader import read_mat

from .opusreader import OpusReader
from .ptirreader import PtirReader
from .omnicreader import OmnicReader
from ..algorithms.util import pixels_fit

class SpectralData:
    """
    The list of current files and the raw data loaded from a hyperspectral
    image file in the GUIs.
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
        self.pixelxy = None # Positions of pixels in image
        self.images = None # list of Image loaded from raw data file
        self.filetype = None # str from [txt, mat, opus, ptir]

    def set_width(self, w):
        try:
            w = int(w)
        except ValueError:
            w = 0
        if w <= 0:
            return False
        h = (self.raw.shape[0] + w - 1) // w
        # if w * h != self.raw.shape[0]:
        #     return False
        self.wh = (w, h)
        return True

    def set_height(self, h):
        if self.set_width(h):
            self.wh = (self.wh[1], self.wh[0])
            return True
        return False

    def read_matrix(self, filename):
        """
        Read data from a file, with some error checking. The object is modified
        only if the file is successfully loaded.
        """
        wh = None
        xy = None
        images = None
        fext = os.path.splitext(filename)[1].lower()
        if fext in ['.txt', '.csv', '.mat']:
            if fext == '.mat':
                filetype = 'mat'
                try:
                    s = read_mat(filename)
                except TypeError:
                    # Workaround for uint16_codec bug (pymatreader
                    # assumes mio5, not mio4)
                    s = scipy.io.loadmat(filename)
                # Assume data are in the biggest matrix in the file
                ss = max(s.items(), key=lambda k: np.size(k[1]) )[1]
            else:
                filetype = 'txt'
                s = {}
                # ss = np.loadtxt(filename)
                ss = pd.read_csv(filename, sep=None, engine='python').to_numpy()
            if 'ImR' in s and 'lambda' in s:
                ss = s['ImR']
                if ss.ndim == 2:
                    raw = ss
                else:
                    assert ss.ndim == 3
                    wh = [ss.shape[1], ss.shape[0]]
                    raw = ss.reshape((-1, ss.shape[2]))
                wn = s['lambda'].flatten()
                if wn[0] < wn[-1]:
                    wn = wn[::-1]
                    raw = raw[:, ::-1]
                assert len(wn) == raw.shape[1]
            elif ss.ndim == 2 and ss.shape[0] > 1 and ss.shape[1] > 1:
                if 'wh' in s:
                    wh = s['wh'].flatten()
                # Assume wavenumbers are a) first column or b) first row
                deltac = np.diff(ss[:, 0])
                deltar = np.diff(ss[0, :])
                if np.all(deltac > 0) or np.all(deltac < 0):
                    d = 1
                    raw = ss[::d, 1:].T
                    wn = ss[::d, 0]
                elif np.all(deltar > 0) or np.all(deltar < 0):
                    raw = ss[1:, :]
                    wn = ss[0, :]
                else:
                    raise RuntimeError(
                        'first column or row must contain sorted wavenumbers')
            else:
                raise RuntimeError(
                    'file does not appear to describe an FTIR image matrix')
        else:
            if fext == '.ptir' or fext == '.hdf':
                filetype = 'ptir'
                reader = PtirReader(filename)
                xy = reader.xy
            elif fext in ['.spa', '.spg']:
                filetype = 'omnic'
                reader = OmnicReader(filename)
            else:
                filetype = 'opus'
                reader = OpusReader(filename)
            raw = reader.AB
            wn = reader.wavenum
            wh = reader.wh
            images = reader.images

        diffsign = np.sign(np.diff(wn))
        if (diffsign >= 0).any() and (diffsign <= 0).any():
            raise RuntimeError('wavenumbers must be sorted')
        npix = raw.shape[0]
        if wh is not None:
            if len(wh) != 2:
                raise RuntimeError('Image size in "wh" must have length 2')
            wh = (int(wh[0]), int(wh[1]))
            if not pixels_fit(npix, wh):
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
        self.images = images
        self.pixelxy = xy
        self.curFile = filename
        self.filetype = filetype
