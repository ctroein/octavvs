#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 16:16:44 2020

@author: carl
"""

# from struct import unpack
import h5py
import numpy as np
from .image import Image

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
        self.wh = None
        self.xy = None
        self.wavenum = None
        self.AB = None
        self.images = []
        if filename is not None:
            self.load(filename)

    def load(self, filename, clip_to_images=True):
        f = h5py.File(filename, mode='r')

        wns = []
        raw = []
        xy = []
        for k, v in f.items():
            if 'MirageDC' in v.attrs:
                wn = v['Spectroscopic_Values'][0,:]
                wns.append(wn)
                for kk, vv in v.items():
                    try:
                        r = vv['Raw_Data']
                    except (AttributeError, ValueError):
                        continue
                    d = r[0,:]
                    if d.shape != wn.shape:
                        # print('incompatible shapes', d.shape, wn.shape)
                        continue
                    raw.append(d)
                    try:
                        xy.append([v.attrs['LocationX'][0],
                                  v.attrs['LocationY'][0]])
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
        self.wh = (len(self.AB), 1)

        self.images = []
        for imtype in ['Image', 'Heightmap']:
            for imnum in range(1000):
                try:
                    im = f['%ss' % (imtype)]['%s_%03d' % (imtype, imnum)]
                except (KeyError):
                    break
                else:
                    if imtype == 'Image':
                        imname = im.name
                    else:
                        imwnum = im.attrs['IRWavenumber'].decode()
                        imname=imwnum + " " + im.attrs['Label'].decode()
                    img = Image(data=im[()][::-1,:], name=imname)
                    img.xy = ([im.attrs['PositionX'][0],
                              im.attrs['PositionY'][0]])
                    img.wh = ([im.attrs['SizeWidth'][0],
                              im.attrs['SizeHeight'][0]])
                    self.images.append(img)

        if clip_to_images:
            # Remove pixel outside imaging area (always the first one?)
            imgxy = np.array([img.xy for img in self.images])
            imgwh = np.array([img.wh for img in self.images])
            minxy = (imgxy - imgwh / 2).min(0)
            maxxy = (imgxy + imgwh / 2).max(0)
            inside = (minxy <= self.xy).all(1) & (self.xy <= maxxy).all(1)
            self.xy = self.xy[inside,:]
            self.AB = self.AB[inside,:]
            self.wh = (len(self.AB), 1)
