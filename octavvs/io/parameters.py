#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 15 19:58:51 2021

@author: carl
"""

import json

class Parameters:
    """
    A class representing all the settings that can be made in some UI,
    saved/loaded and e.g. used to start a batch job.
    """
    def __init__(self, items=None):
        if items is not None:
            self.__dict__.update(items)

    def save(self, filename):
        with open(filename, 'w') as fp:
            json.dump(vars(self), fp, indent=4)

    def load(self, filename):
        with open(filename, 'r') as fp:
            data = json.load(fp)
            self.__dict__.update(data)

    def filtered(self, prefix):
        "Returns a dict of parameters that start with the given prexix"
        return { k: v for k, v in vars(self).items() if k.startswith(prefix)}
