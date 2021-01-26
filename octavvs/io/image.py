#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 15 17:14:10 2020

@author: carl
"""

class Image:
    """
    An image, possibly white light or from a particular wavenumber
    """
    def __init__(self, data=None, name=None, fmt=None):
        self.data = data
        self.fmt = fmt
        # filename=None,
        # self.filename = filename
        self.name = name  #filename if name is None else name
        self.xy = None
        self.wh = None
