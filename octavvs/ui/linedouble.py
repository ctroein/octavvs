#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 14 16:04:24 2019

@author: carl

A QLineEdit widget that holds a double and handles some validation but
doesn't require it to be valid at all points during editing

"""

from PyQt5.QtWidgets import QLineEdit
#from PyQt5.QtGui import QDoubleValidator
from PyQt5.QtCore import pyqtSlot #, QLocale

class LineDouble(QLineEdit):
    def __init__(self, parent=None, vmin=0, vmax=100, fmt='%.2f'):
        QLineEdit.__init__(self)
        self.default = (vmin + vmax) / 2
        self.min = vmin
        self.max = vmax
        self.setFormat(fmt)
        self.editingFinished.connect(self.validate)
#        self.setValidator(QDoubleValidator(min, max, 1000, parent=self))

#    @pyqtSlot(float, float)
#    def setRange(self, min, max):
#        self.min = min
#        self.max = max
#        self.validator().setRange(min, max, 1000)

    def setRange(self, vmin, vmax, default=None):
        assert(vmin <= vmax)
        self.min = vmin
        self.max = vmax
        if default is not None:
            self.default = default
        elif self.text() != '':
            self.default = float(self.text())
        else :
            self.default = (vmin+vmax)/2.
        assert(vmin <= self.default <= vmax)

    def setFormat(self, fmt):
        self.fmt=fmt

    @pyqtSlot()
    def validate(self):
        self.setValue(self.value())

    @pyqtSlot(float)
    def setValue(self, val=None):
#        if sci:
#            self.setText(QLocale().toString(val, 'g', 3))
#        else:
#            self.setText(QLocale().toString(val, 'f', 2))
        if val is None:
            val = self.default
        self.setText(self.fmt % min(max(val, self.min), self.max))

    def hasAcceptableValue(self):
        try:
            float(self.text())
        except Exception:
            return False
        return True

    def value(self):
#        f = QLocale().toFloat(self.text())[0]
#        if self.hasAcceptableInput():
#            return f
#        return max(self.validator().bottom(), min(self.validator().top(), f))
        v = self.default
        try:
            v = float(self.text())
        except Exception:
            pass
        return min(max(v, self.min), self.max)

