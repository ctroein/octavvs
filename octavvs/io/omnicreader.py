#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  9 16:16:44 2020

@author: carl
"""

import numpy as np
# from .image import Image

class OmnicReader:
    """
    A simple wrapper around the spectrochempy loader for OMNIC files, with
    very limited ability to convert data into the shape assumed by OCTAVVS.

    Member variables:
        wh: image dimensions (w, h)
        xy: points coordinates [(x,y)]
        wavenum: array of wavenumbers
        AB: data matrix, array(points, len(wn))
        image: tuple (img_data_bytes, img_type_str) (unused)
    """

    def __init__(self, filename=None):
        """
        filename:  spa/spg file to load data from
        """
        self.wh = None
        self.xy = None
        self.wavenum = None
        self.AB = None
        self.images = []
        if filename is not None:
            self.load(filename)

    def load(self, filename):
        try:
            import spectrochempy as scp
        except ImportError:
            raise RuntimeError(
                'Importing an OMNIC file requires the spectrochempy package. '
                'E.g. "conda install -c conda-forge -c cantera -c spectrocat '
                'spectrochempy" or "pip install spectrochempy"')
        om = scp.read_omnic(filename)
        self.AB = om.data
        self.wavenum = om.x.values.m
        assert self.AB.ndim == 2 and (self.AB.shape[1],) == self.wavenum.shape


