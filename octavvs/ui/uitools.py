#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 17:25:10 2020

@author: carl
"""

import numpy as np

from . import constants

def ixfinder_nearest(wn, val):
    return len(wn)-1 - (np.abs(wn - val)).argmin()

def ixfinder_noless(wn, val):
    return wn[::-1].searchsorted(val, 'left')

def ixfinder_nomore(wn, val):
    return wn[::-1].searchsorted(val, 'right')-1

def slider_to_box(slider, box, wn, ixfinder):
    """Set a QLineEdit from the value on a slider, discretized by
    the wavenumber array wn"""
    if wn is not None:
        if (not box.hasAcceptableInput() or
            slider.value() != ixfinder(wn, box.value())):
            box.setValue(wn[-1-slider.value()])
    elif (not box.hasAcceptableInput() or
            slider.value() != int(round(slider.maximum() *
            (box.value() - constants.WMIN) / (constants.WMAX - constants.WMIN)))):
        box.setValue(constants.WMIN + (constants.WMAX - constants.WMIN) *
                     slider.value() / slider.maximum())

def box_to_slider(slider, box, wn, ixfinder):
    """Set a slider from the value in a lineedit, discretized by
    the wavenumber array wn"""
    oldb = slider.blockSignals(True)
    if wn is not None:
        # assert slider.maximum() == len(wn) - 1
        if slider.maximum() != len(wn) - 1:
            slider.setMaximum(len(wn) - 1)
        if box.hasAcceptableInput():
            slider.setValue(ixfinder(wn, box.value()))
        else:
            box.setValue(wn[-1-slider.value()])
    else:
        slider.setValue(int(round(slider.maximum() *
              (box.value() - constants.WMIN) / (constants.WMAX - constants.WMIN))))
    slider.blockSignals(oldb)

