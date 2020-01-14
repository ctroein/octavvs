#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 16:16:44 2020

@author: carl
"""

from struct import unpack
import numpy as np
from io import BytesIO
import matplotlib.pyplot as plt

class OpusReader:
    """
    A simple class to extract data from an OPUS file. This is meant as an alternative
        to .mat export and will only load the data matrix and wavenumbers.

    Member variables:
        wh - image dimensions (w, h)
        wn - array of wavenumbers
        AB - data matrix, array(w*h, len(wn))
    """

    def __init__(self, filename=None):
        self.wh = (0, 0)
        self.wavenum = None
        self.AB = None
        self.img = None
        if filename is not None:
            self.load(filename)

    def readChunk(self, file, chunksize, chunkoffset):
        file.seek(chunkoffset)
        return file.read(4 * chunksize)

    def parseParams(self, chunk):
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

    def parseImage(self, chunk):
        p = self.parseParams(chunk[12:])[0]
        jump = p['F00']
        p, q = self.parseParams(chunk[jump:])
        imgtype = 'jpg' if p['V08'] == 258 else 'bmp'
        q = q + jump + (14 if imgtype == 'jpg' else 0)
        print('pq',p,q)
        print(chunk[q-16:q+128])
        imgdata = chunk[q:q+p['SRC']]
        return plt.imread(BytesIO(imgdata), imgtype)

    def load(self, filename):
        f = open(filename, 'rb')
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
                abparams = self.parseParams(self.readChunk(f, dcsize, dcoffs))[0]
            elif dt == 15:
                abmatrix = self.readChunk(f, dcsize, dcoffs)
            elif dt == 0 and db == 0 and dtt == 160:
                self.img = self.parseImage(self.readChunk(f, dcsize, dcoffs))
        # Assume the data were read ok
        start, pad = 16006, 38
        wns, w, h, wn1, wn2 = [abparams[k] for k in ['NPT', 'NPX', 'NPY', 'FXV', 'LXV']]
        n = w * h * (wns + pad)
        raw = np.frombuffer(abmatrix, dtype=np.float32)
        self.AB = raw[start:start+n].reshape((w * h, wns + pad))[:, :wns]
        self.wavenum = np.linspace(wn1, wn2, wns)
        self.wh = (w, h)

filename = '/home/carl/src/2dftir/FTIR_images/Oxana/Control1_Spot5.0'
#filename = '/home/carl/src/2dftir/FTIR_images/anders_engdahl/1_Remission deceased mice V_Cerebellum__res_4_AE-2015-04-21.2'
# 12 00 2000 0 1 0
foo = OpusReader(filename)

plt.imshow(foo.img)

#fout = open('/home/carl/src/2dftir/FTIR_images/Oxana/testout.bmp', 'wb')
#fout.write(foo.img)
#fout.close()

#foo.parseParams(foo.florp[12:])
#pms = foo.parseParams(foo.florp[2012:])
#img = foo.florp[2442:2442+pms['SRC']]
#
#foo.florp[2000:2452]
#foo.florp[2400:2600]
