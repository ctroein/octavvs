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
import h5py

from .opusreader import OpusReader
from .ptirreader import PtirReader
from .omnicreader import OmnicReader
from ..algorithms.util import pixels_fit
from .. import octavvs_version

class SpectralData:
    """
    The list of current files and the raw data loaded from a hyperspectral
    image file in the GUIs.
    """
    def __init__(self):
        self.foldername = '' # Root dir
        self.filenames = []  # With full paths
        self.curFile = ''    # The currently loaded file
        self.wavenumber = None # array in order low->high
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
        raw = None
        fext = os.path.splitext(filename)[1].lower()
        if fext in ['.txt', '.csv', '.mat', '.xls', '.xlsx']:
            s = {} # possible
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
            elif fext in ['.xls', '.xlsx']:
                filetype = 'xls'
                ss = pd.read_excel(filename, header=None).to_numpy()
            else:
                filetype = 'txt'
                ss = pd.read_csv(filename, sep=None, engine='python',
                                 header=None).to_numpy()
            # Check for some particular format(s)
            if 'ImR' in s and 'lambda' in s:
                # Matlab file with data in ImR, wavelengths in lambda
                # (encountered with hyperspectral data in Lund)
                ss = s['ImR']
                if ss.ndim == 2:
                    raw = ss
                else:
                    assert ss.ndim == 3
                    wh = [ss.shape[1], ss.shape[0]]
                    raw = ss.reshape((-1, ss.shape[2]))
                wn = s['lambda'].flatten()
                assert len(wn) == raw.shape[1]
            if 'wavenumber' in s and 'y' in s:
                # Quasar.mat - and we ought to load map_x/y to a wh rectangle
                # or fall back to pixelxy if necessary.
                wn = s['wavenumber'].flatten()
                raw = s['y']
            elif ss.ndim == 2 and ss.shape[0] > 1 and ss.shape[1] > 1:
                if 'wh' in s:
                    wh = s['wh'].flatten()
                # Assume wavenumbers are a) first row or b) first column
                for transp in range(2):
                    if transp:
                        ss = ss.T
                    wn = ss[0]
                    cols = None
                    # if mixed data, remove strings (these should be saved
                    # as annotations!)
                    if wn.dtype == object:
                        cols = [not isinstance(s, str) for s in wn]
                        wn = wn[cols].astype(float, casting='unsafe')
                        if len(wn) < 2:
                            continue
                    deltas = np.diff(wn)
                    if np.all(deltas > 0) or np.all(deltas < 0):
                        if cols is not None:
                            raw = ss[1:, cols].astype(float)
                        else:
                            raw = ss[1:]
                        break
                if raw is None:
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

        # Ensure ascending wavenumbers
        assert len(wn) == raw.shape[1]
        if wn[0] > wn[1]:
            wn = wn[::-1]
            raw = raw[:, ::-1]

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

    def save_matrix_matlab_ab(self, filename, wn, ydata):
        ab = np.hstack((wn[:, None], ydata.T))
        abdata = {'AB': ab, 'wh': self.wh }
        if self.pixelxy is not None:
            abdata['xy'] = self.pixelxy
        scipy.io.savemat(filename, abdata)

    def save_matrix_matlab_quasar(self, filename, wn, ydata):
        out = {'y': ydata, 'wavenumber': wn}
        if self.pixelxy is not None:
            map_x, map_y = np.array(self.pixelxy).T
        else:
            map_x = np.tile(range(self.wh[1]), self.wh[0])
            map_y = np.repeat(range(self.wh[0]), self.wh[1])
        out['map_x'] = map_x[:, None]
        out['map_y'] = map_y[:, None]
        scipy.io.savemat(filename, out)

    def save_matrix_ptir(self, filename, wn, ydata):
        f = h5py.File(filename, mode='w')

        f.attrs['DocType'] = b'IR'
        f.attrs['SoftwareVersion'] = octavvs_version.encode('utf8')

        digits = int(np.log10(len(ydata))) + 1
        for i, d in enumerate(ydata):
            mm = f.create_group(f'Measurement{i+1:0{digits}}')
            if self.pixelxy is not None:
                x, y = self.pixelxy[i]
            else:
                x = i % self.wh[0]
                y = i // self.wh[0]
            mm.attrs['LocationX'] = [x]
            mm.attrs['LocationY'] = [y]

            if not i:
                specval = mm.create_dataset('Spectroscopic_Values',
                                            data=wn[None, :])

            ch = mm.create_group('Channel_000')
            rd = ch.create_dataset('Raw_Data', data=d[None, :])
            rd.attrs['Spectroscopic_Values'] = specval.ref

        if self.images is not None and len(self.images):
            imgs = f.create_group('Images')
            digits = max(3, int(np.log10(len(self.images))) + 1)
            for i, im in enumerate(self.images):
                imds = imgs.create_dataset(f'Image_{i:0{digits}}',
                                           data=im.data[::-1])
                imds.attrs['Label'] = np.bytes_(im.name)
                imds.attrs['PositionX'] = [im.xy[0]]
                imds.attrs['PositionY'] = [im.xy[1]]
                imds.attrs['SizeWidth'] = [im.wh[0]]
                imds.attrs['SizeHeight'] = [im.wh[1]]

        f.close()

    def save_matrix(self, filename, fmt='ab', wn=None, ydata=None):
        """
        Save processed data in some format.

        Parameters
        ----------
        filename : str
            Name of the output file.
        fmt : str, optional
            File format: 'ab', 'quasar' or 'ptir'. The default is 'ab', a
            MATLAB file with array 'AB' with wavenumbers in the first _column_,
            image size in 'wh' and optionally pixel coordinates in 'xy'.
            The 'quasar' format has wavenumbers in a separate array.
            The 'ptir' format is a reduced variant of the HDF5 format used
            by PTIR Studio.
        wn : 1D array, optional
            Wavenumber vector. The default is self.wn.
        ydata : 2D array, optional
            Data in (pixel, wn) order. The default is self.raw.

        Returns
        -------
        None.

        """
        if wn is None:
            wn = self.wn
        if ydata is None:
            ydata = self.raw
        if fmt == 'ab':
            self.save_matrix_matlab_ab(filename, wn, ydata)
        elif fmt == 'quasar':
            self.save_matrix_matlab_quasar(filename, wn, ydata)
        elif fmt == 'ptir':
            self.save_matrix_ptir(filename, wn, ydata)
        else:
            raise ValueError('Unknown save file format')


