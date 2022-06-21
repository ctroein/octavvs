#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 16:16:44 2020

@author: carl
"""

from struct import unpack
import numpy as np
from .image import Image

class OpusReader:
    """
    A simple class to extract data from an OPUS file. This is meant as an alternative to
        .mat export and will only load the data matrix and wavenumbers plus the white light image.

    Member variables:
        wh: image dimensions (w, h)
        wavenum: array of wavenumbers
        AB: data matrix, array(w*h, len(wn))
        image: tuple (img_data_bytes, img_type_str)
    """

    def __init__(self, filename=None):
        """
        filename: OPUS file to load data from
        """
        self.wh = (0, 0)
        self.wavenum = None
        self.AB = None
        self.images = []
        if filename is not None:
            self.load(filename)

    def read_chunk(self, file, chunksize, chunkoffset):
        file.seek(chunkoffset)
        return file.read(4 * chunksize)

    def parse_params(self, chunk):
        cix = 0
        inttypes = {0: 'B', 1: '<H', 2: '<I'}
        params = {}
        while cix < len(chunk):
            pname = chunk[cix:cix+3].decode()
            ptype = unpack('<H', chunk[cix+4:cix+6])[0]
            psize = unpack('<H', chunk[cix+6:cix+8])[0]
            pend = cix+8+2*psize
            pdata = chunk[cix+8:pend]
            cix = pend
            if pname == 'END':
                return params, cix
            if ptype == 0:
                params[pname] = unpack(inttypes[psize], pdata)[0]
            elif ptype == 1:
                params[pname] = unpack('<f' if psize == 2 else '<d', pdata)[0]
#            print('pname',pname, ptype, psize)
        return params, cix

    def parse_image(self, chunk):
        p = self.parse_params(chunk[12:])[0]
        jump = p['F00']
        p, q = self.parse_params(chunk[jump:])
        imgtype = 'jpg' if p['V08'] == 258 else 'bmp'
        q = q + jump + (14 if imgtype == 'jpg' else 0)
#        print('pq',p,q)
#        print(chunk[q-16:q+128])
        imgdata = chunk[q:q+p['SRC']]
        return (imgdata, imgtype)

    def load(self, filename):
        f = open(filename, 'rb')
        self.images = []
        HLEN = 504
        header = f.read(HLEN)
        abparams = abmatrix = None
        for ix in range(24, HLEN, 12):
            dt = header[ix]
            db = header[ix+1]
            dtt = header[ix+2]
            dcsize = unpack('<I', header[ix+4:ix+8])[0]
            dcoffs = unpack('<I', header[ix+8:ix+12])[0]
#            print(dt, db, dtt, dcsize, dcoffs)
            if not dcoffs:
                break
            if dt == 31:
                abparams = self.parse_params(self.read_chunk(f, dcsize, dcoffs))[0]
            elif dt == 15:
                abmatrix = self.read_chunk(f, dcsize, dcoffs)
            elif dt == 0 and db == 0 and dtt == 160:
                im = self.parse_image(self.read_chunk(f, dcsize, dcoffs))
                self.images.append(Image(data=im[0], fmt=im[1]))
        # Assume the data were read ok.
        # Some files have only a single spectrum and no start/pad.
        # Example file 1 {'NPT': 1530 (points), 'FXV': 3847.55 (wn1),
        # 'LXV': 898.72 (wn2), 'CSF': 1.0, 'MXY': 0.7652,
        # 'MNY': -0.05269, 'DPF': 1, 'AOX': 34254.5 (origin-x?),
        # 'AOY': 6906.9 (origin-y?), 'NPX': 64 (w), 'DDX': 2.61079 (xres),
        # 'NPY': 64 (h), 'DDY': 2.61078 (yres) }
        # Example file 2: {'DPF': 1, 'NPT': 3734, 'FXV': 3998.983,
        # 'LXV': 399.223, 'CSF': 1.0, 'MXY': 0.030427, 'MNY': -0.1112276}
        raw = np.frombuffer(abmatrix, dtype=np.float32)
        wns, wn1, wn2 = [abparams[k] for k in ['NPT', 'FXV', 'LXV']]
        start, pad = 16006, 38
        try:
            w, h = abparams['NPX'], abparams['NPY']
        except KeyError:
            w, h = 1, 1
            pad = 0
            if len(raw) < start + wns:
                start = 0
        n = w * h * (wns + pad)
        self.AB = raw[start:start+n].reshape((w * h, wns + pad))[:, :wns]
        self.wavenum = np.linspace(wn1, wn2, wns)
        self.wh = (w, h)
