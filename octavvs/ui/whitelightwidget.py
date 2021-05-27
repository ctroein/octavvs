#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 16:00:35 2020

@author: carl
"""

from io import BytesIO
from PyQt5.QtWidgets import QSizePolicy
#from PyQt5.QtCore import pyqtSignal, pyqtSlot
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
# import matplotlib.patches as patches


class WhiteLightWidget(FigureCanvas):
    def __init__(self, parent=None):
        self.fig = Figure()
        self.ax = self.fig.add_subplot(111)
        self.ax.set_axis_off()
        self.ax.set_position([0,0,1,1])
        FigureCanvas.__init__(self, self.fig)
        self.setParent(parent)
        # FigureCanvas.setSizePolicy(self, QSizePolicy.Expanding, QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)
        self.img = None # Image data
        self.image = None # Image object
        # self.points = []

    # def draw_rectangles(self, pixelxy):
    #     for p in self.points:
    #         p.remove()
    #     self.points = []
    #     if pixelxy is None or self.image is None:
    #         return
    #     ixy = self.image.xy
    #     iwh = self.image.wh
    #     if not ixy or not iwh:
    #         return
    #     scx = self.ax.get_xlim()[1]
    #     scy = self.ax.get_ylim()[0]
    #     i = 0
    #     for xy in pixelxy:
    #         x = (xy[0] - ixy[0]) / iwh[0] + .5
    #         y = (xy[1] - ixy[1]) / iwh[1] + .5
    #         if x < 0 or x > 1 or y < 0 or y > 1:
    #             continue
    #         p = patches.Rectangle((x*scx, y*scy), 5, 5, fill=False,
    #                               color='b')
    #         self.ax.add_patch(p)
    #         self.points.append(p)
    #         i = i + 1
    #     self.draw_idle()

    def load(self, filename=None, image=None):
        self.ax.clear()
        self.ax.set_axis_off()
        self.img = None
        self.image = image

        if image is not None:
            if image.fmt is not None:
                self.img = plt.imread(BytesIO(image.data), format=image.fmt)
            elif image.data is not None:
                self.img = image.data
        elif filename is not None:
            try:
                self.img = plt.imread(filename)
            except FileNotFoundError:
                pass
        if self.img is not None:
            self.ax.imshow(self.img, cmap='terrain')
        self.draw_idle()

    def is_loaded(self):
        return self.img is not None
