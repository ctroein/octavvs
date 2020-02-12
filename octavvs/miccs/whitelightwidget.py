#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 16:00:35 2020

@author: carl
"""

from PyQt5.QtWidgets import QSizePolicy
#from PyQt5.QtCore import pyqtSignal, pyqtSlot
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt

class WhiteLightWidget(FigureCanvas):
    def __init__(self, parent=None):
        self.fig = Figure()
        self.ax = self.fig.add_subplot(111)
        self.ax.set_axis_off()
        self.ax.set_position([0,0,1,1])
        FigureCanvas.__init__(self, self.fig)
        self.setParent(parent)
        FigureCanvas.setSizePolicy(self, QSizePolicy.Expanding, QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)
        self.img = None

    def load(self, filename, format=None):
        self.ax.clear()
        self.ax.set_axis_off()
        self.img = None
        if filename is not None:
            try:
                self.img = plt.imread(filename, format=format)
                self.ax.imshow(self.img)
            except FileNotFoundError:
                pass
        self.draw_idle()
