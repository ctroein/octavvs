#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 16:16:44 2020

@author: carl
"""

# from struct import unpack
import h5py
import numpy as np
import re
from .image import Image

class PtirReader:
    """
    A simple class to extract data from a PTIR Studio HDF5 file. This is meant
        as an alternative to .mat export and will load the spectra from
        the scanned points plus whole images in specific wavenumbers.

    Member variables:
        wh: image dimensions (w, h)
        xy: points coordinates [(x,y)]
        wavenum: array of wavenumbers
        AB: data matrix, array(points, len(wn))
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
        wh = None
        defwn = None
        if 'Measurement_000' in f:
            v = f['Measurement_000']
            defwn = v['Spectroscopic_Values'][0,:]

        for k, v in f.items():
            if 'MirageDC' in v.attrs:
                if re.match(r'^Measurement_0+$', k):
                    continue
                if 'Spectroscopic_Values' in v:
                    wn = v['Spectroscopic_Values'][0,:]
                else:
                    wn = defwn
                wns.append(wn)
                for kk, vv in v.items():
                    try:
                        r = vv['Raw_Data']
                    except (AttributeError, ValueError):
                        if kk not in ['Spectroscopic_Values',
                                      'Spectroscopic_Indices']:
                            print('skipping unknown', kk)
                        continue
                    d = r[:,:]
                    if d.shape[1] != len(wn):
                        print('incompatible shapes', d.shape, wn.shape)
                        continue
                    if len(d) == 1:
                        try:
                            xy.append([v.attrs['LocationX'][0],
                                      v.attrs['LocationY'][0]])
                        except AttributeError:
                            xy.append([0, 0])
                            print('no LocationX/Y')
                    else:
                        try:
                            rxy = v['Position_Values']
                        except AttributeError:
                            print('no position values')
                            continue
                        if rxy.shape != (len(d), 2):
                            print('position shape error', rxy.shape, d.shape)
                            continue
                        xy.extend(rxy)
                        if wh is None:
                            try:
                                wh = [v.attrs['RangeXPoints'][0],
                                      v.attrs['RangeYPoints'][0]]
                            except AttributeError:
                                wh = [0, 0]
                        else:
                            wh = [0, 0]
                    raw.extend(d)
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
            if self.images:
                # Remove pixel(s) outside imaging area (always the first one?)
                imgxy = np.array([img.xy for img in self.images])
                imgwh = np.array([img.wh for img in self.images])
                minxy = (imgxy - imgwh / 2).min(0)
                maxxy = (imgxy + imgwh / 2).max(0)
                inside = (minxy <= self.xy).all(1) & (self.xy <= maxxy).all(1)
                self.xy = self.xy[inside,:]
                self.AB = self.AB[inside,:]
            self.wh = (len(self.AB), 1)

        # Switch to rectangle mode if appropriate
        if not self.images and wh is not None and wh[0] * wh[1] == len(raw):
            self.wh = wh
            self.xy = None
