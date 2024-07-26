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

    def get_xy(self):
        """
        Get pixel coordinates, regardless of data format.

        Returns
        -------
        Coordinates : numpy.ndarray
            Array of size (npixels, 2) with x, y pairs.
        """
        if self.pixelxy is not None:
            return self.pixelxy
        elif self.wh is not None and self.wh[0] and self.wh[1]:
            return np.indices((self.wh[1], self.wh[0])).reshape(
                (2, -1)).T[:len(self.raw),::-1]
        else:
            return np.vstack(range(len(self.raw)), [0] * len(self.raw)).T

    def extract_data_from_matrix(self, mx, meta):
        """
        Extract wavenumbers, raw data and image dimensions from some kind
        of data matrix, with or without additional metadata

        Parameters
        ----------
        mx : numpy.ndarray or None
            raw matrix data, possibly with wavenumbers and annotations.
        meta : dict
            additional metadata, possibly including mx itself

        Returns
        -------
        raw : numpy.ndarray
            matrix of floats
        wn : numpy.ndarray
            sorted array of wavenumbers
        wh : tuple or None
            image dimensions (width and height)
        """
        if meta is None:
            meta = {}
        wh = None

        # Check for some particular format(s)
        if 'ImR' in meta and 'lambda' in meta:
            # Matlab file with data in ImR, wavelengths in lambda
            # (encountered with hyperspectral data in Lund)
            mx = meta['ImR']
            if mx.ndim == 2:
                raw = mx
            else:
                assert mx.ndim == 3
                wh = [mx.shape[1], mx.shape[0]]
                raw = mx.reshape((-1, mx.shape[2]))
            wn = meta['lambda'].flatten()
            assert len(wn) == raw.shape[1]
            return raw, wn, wh
        if 'wavenumber' in meta and 'y' in meta:
            # Quasar.mat - and we ought to load map_x/y and figure out a
            # wh rectangle or fall back to pixelxy if necessary.
            wn = meta['wavenumber'].flatten()
            raw = meta['y']
            ...
            return raw, wn, None

        if mx is None:
            # Assume data are in the biggest matrix in the file
            mx = max(meta.items(), key=lambda k: np.size(k[1]))[1]

        if mx.ndim == 3 and (np.array(mx.shape) > 1).all():
            wh = (mx.shape[0], mx.shape[1])
            mx = mx.reshape((-1, mx.shape[2]))
        if mx.ndim != 2 or (np.array(mx.shape) < 2).any():
            raise RuntimeError(
                'file does not appear to describe an FTIR image matrix')

        if 'wh' in meta:
            wh = meta['wh'].flatten()

        # Assume wavenumbers are a) first row or b) first column
        if mx.dtype == object:
            # Find columns with only numbers. Removed strings ought to
            # be saved as annotations.
            alltypes = np.vectorize(lambda x: np.issubdtype(
                type(x), np.number))(mx)
            numcols = [np.all(alltypes, axis=0), np.all(alltypes, axis=1)]
            wns = [wn.astype(float, casting='unsafe') for wn in
                   [mx[0, numcols[0]], mx[numcols[1], 0]]]
        else:
            wns = [mx[0, :], mx[:, 0]]
        # Find the direction with most sorted wavenumbers
        for i, wn in enumerate(wns):
            deltas = np.diff(wn)
            if np.any(deltas <= 0) and np.any(deltas >= 0):
                wns[i] = []
        if len(wns[0]) < 3 and len(wns[1]) < 3:
            raise RuntimeError(
                'First column or row must contain sorted wavenumbers')
        # Transpose if needed
        if len(wns[1]) > len(wns[0]):
            mx = mx.T
            mxt = 1
        else:
            mxt = 0
        wn = wns[mxt]
        # Remove the non-numeric stuff
        if mx.dtype == object:
            nonnum = mx[0, ~numcols[mxt]]
            if len(nonnum):
                print(f'Removing non-numeric fields: {nonnum}')
            raw = mx[1:, numcols[mxt]].astype(float)
        else:
            raw = mx[1:]

        if raw.dtype != np.float32 and raw.dtype != np.float64:
            print(f'Loaded as {raw.dtype}, converting to float64')
            wn = wn.astype(float)
            raw = raw.astype(float)
        return raw, wn, wh

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
        if fext == '.mat':
            filetype = 'mat'
            try:
                s = read_mat(filename)
            except TypeError:
                # Workaround for uint16_codec bug (pymatreader
                # assumes mio5, not mio4)
                s = scipy.io.loadmat(filename)
            raw, wn, wh = self.extract_data_from_matrix(None, s)
        elif fext in ['.txt', '.csv', '.xls', '.xlsx']:
            if fext in ['.xls', '.xlsx']:
                filetype = 'xls'
                ss = pd.read_excel(filename, header=None).to_numpy()
            else:
                filetype = 'txt'
                ss = pd.read_csv(filename, sep=None, engine='python',
                                 header=None, comment="#").to_numpy()
            raw, wn, wh = self.extract_data_from_matrix(ss, None)
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

        # Guess the image shape if square
        npix = raw.shape[0]
        if wh is not None:
            # print(f'image size {wh}')
            if len(wh) != 2:
                raise RuntimeError('Image size in "wh" must have length 2')
            wh = (int(wh[0]), int(wh[1]))
            if not pixels_fit(npix, wh):
                raise RuntimeError('Image size in "wh" does not match data size')
        elif xy is not None:
            # We should do some clever deduction of size and add code
            # to deduce the best rectangular grid from xy coordinates,
            # but for now we keep it simple.
            yy = np.diff(xy[:, 0])
            yy = yy - yy.mean()
            yy = yy / np.median(yy)
            guessh = (yy < .25 * yy.min()).sum() + 1
            if guessh > npix**.2 and guessh < npix **.8:
                wh = ((npix + guessh - 1) // guessh, guessh)
                print(f'assuming image size {wh} from xy data')
        if wh is None:
            if npix == self.wh[0] * self.wh[1]:
                wh = self.wh
            else:
                res = int(np.sqrt(npix))
                if npix == res * res:
                    wh = (res, res)
                    print(f'assuming square image {wh}')
                else:
                    wh = (npix, 1)
                    print(f'assuming non-grid data {wh}')
        self.wh = wh

        self.raw = raw
        self.wavenumber = wn
        self.wmin = self.wavenumber.min()
        self.wmax = self.wavenumber.max()
        self.images = images
        self.pixelxy = xy
        self.curFile = filename
        self.filetype = filetype

    def save_matrix_matlab_ab(self, filename, wn, ydata, metadata):
        ab = np.hstack((wn[:, None], ydata.T))
        abdata = {'AB': ab, 'wh': self.wh }
        if self.pixelxy is not None:
            abdata['xy'] = self.pixelxy
        for k, v in metadata.items():
            abdata[k] = v
        scipy.io.savemat(filename, abdata)

    def save_matrix_matlab_quasar(self, filename, wn, ydata, metadata):
        out = {'y': ydata, 'wavenumber': wn}
        if self.pixelxy is not None:
            map_x, map_y = np.array(self.pixelxy).T
        else:
            map_x = np.tile(range(self.wh[1]), self.wh[0])
            map_y = np.repeat(range(self.wh[0]), self.wh[1])
        out['map_x'] = map_x[:, None]
        out['map_y'] = map_y[:, None]
        for k, v in metadata.items():
            out[k] = v
        scipy.io.savemat(filename, out)

    def save_matrix_pandas(self, filename, wn, ydata, fmt, metadata):
        df = pd.DataFrame(ydata.T, index=wn)
        kpad = "#" if fmt == "csv" else ""
        mdf = pd.DataFrame(metadata.values(),
                           index=[kpad + k for k in metadata])
        df = pd.concat([mdf, df])
        if fmt == 'csv':
            df.to_csv(filename, header=False, float_format="%.16g")
        elif fmt == 'xlsx':
            df.to_excel(filename)

    def save_matrix_ptir(self, filename, wn, ydata, metadata=None):
        f = h5py.File(filename, mode='w')

        f.attrs['DocType'] = b'IR'
        f.attrs['SoftwareVersion'] = octavvs_version.encode('utf8')

        digits = int(np.log10(len(ydata))) + 1
        for i, d in enumerate(ydata):
            mm = f.create_group(f'Measurement{i+1:0{digits}}')
            # if self.pixelxy is not None:
            #     x, y = self.pixelxy[i]
            # else:
            #     x = i % self.wh[0]
            #     y = i // self.wh[0]
            x = metadata["X"][i]
            y = metadata["Y"][i]
            mm.attrs['LocationX'] = [x]
            mm.attrs['LocationY'] = [y]
            if "Filename" in metadata:
                mm.attrs["Filename"] = [metadata["Filename"][i]]

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

    def save_matrix(self, filename, fmt='ab', wn=None, ydata=None,
                    metadata=None):
        """
        Save processed data in some format.

        Parameters
        ----------
        filename : str
            Name of the output file.
        fmt : str, optional
            File format: 'ab', 'quasar', 'ptir', 'csv', 'xlsx'.
            The default is 'ab', a MATLAB file with array 'AB'
            with wavenumbers in the first _column_,
            image size in 'wh' and optionally pixel coordinates in 'xy'.
            The 'quasar' format has wavenumbers in a separate array.
            The 'ptir' format is a reduced variant of the HDF5 format used
            by PTIR Studio.
        wn : 1D array, optional
            Wavenumber vector. The default is self.wn.
        ydata : 2D array, optional
            Data in (pixel, wn) order. The default is self.raw.
        metadata : dict, optional
            Data to be included in the output, e.g. as comment lines before
            the data.

        Returns
        -------
        None.

        """
        if wn is None:
            wn = self.wn
        if ydata is None:
            ydata = self.raw
        if metadata is None:
            pxy = self.get_xy().T
            metadata = {"X": pxy[0], "Y": pxy[1]}
        if fmt == 'ab':
            self.save_matrix_matlab_ab(filename, wn, ydata, metadata=metadata)
        elif fmt == 'quasar':
            self.save_matrix_matlab_quasar(
                filename, wn, ydata, metadata=metadata)
        elif fmt == 'ptir':
            self.save_matrix_ptir(filename, wn, ydata, metadata=metadata)
        elif fmt == 'csv' or fmt == 'xlsx':
            self.save_matrix_pandas(
                filename, wn, ydata, fmt, metadata=metadata)
        else:
            raise ValueError('Unknown save file format')


