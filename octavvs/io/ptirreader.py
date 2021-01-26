#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 16:16:44 2020

@author: carl
"""

# from struct import unpack
import h5py
import numpy as np

class PtirReader:
    """
    A simple class to extract data from a PTIR Studio HDF5 file. This is meant
        as an alternative to .mat export and will only load the spectra from
        the scanned points, not the whole image in some specific wavenumber.

    Member variables:
        wh: image dimensions (w, h)
        wavenum: array of wavenumbers
        AB: data matrix, array(w*h, len(wn))
        image: tuple (img_data_bytes, img_type_str)
    """

    def __init__(self, filename=None):
        """
        filename:  HDF5 file to load data from
        """
        self.wh = (0, 0)
        self.xy = []
        self.wavenum = None
        self.AB = None
        self.image = None
        if filename is not None:
            self.load(filename)

    def load(self, filename):
        f = h5py.File(filename)

        wns = []
        raw = []
        xy = []
        for k, v in f.items():
            if 'MirageDC' in v.attrs:
                wn = v['Spectroscopic_Values'].value[0,:]
                wns.append(wn)
                for kk, vv in v.items():
                    try:
                        r = vv['Raw_Data']
                    except (AttributeError, ValueError):
                        continue
                    d = r.value[0,:]
                    assert d.shape == wn.shape
                    raw.append(d)
                    try:
                        xy.append([v.attrs['LocationX'].value[0,0],
                                  v.attrs['LocationY'].value[0,0]])
                    except AttributeError:
                        xy.append([0, 0])
        if not wns:
            raise RuntimeError('No spectra in input file')
        if not all([len(w) == len(wns[0]) for w in wns]):
            raise NotImplementedError('Unable to load spectra of different length')
        wns = np.array(wns)
        beg = wns[:,0]
        end = wns[:,-1]
        if beg.max() - beg.min() > 5 or end.max() - end.min() > 5:
            raise NotImplementedError('Unable to load spectra with different wavenumbers')
        self.wavenum = np.median(wns, axis=0)[::-1]
        self.AB = np.array(raw)[:, ::-1]
        self.xy = np.array(xy)
        self.wh = (len(wns), 1)

        try:
            im = f['Images']['Image_000']
        except (AttributeError, ValueError):
            self.image = None
        else:
            self.image = im.value

