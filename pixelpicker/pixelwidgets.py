#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 01:07:18 2019

@author: carl
"""

from PyQt5.QtWidgets import QSizePolicy
from PyQt5.QtCore import pyqtSignal, pyqtSlot
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from collections import Counter


class PixelWidget(FigureCanvas):
    def __init__(self, parent=None):
        self.fig = Figure()
        self.ax = self.fig.add_subplot(111)
        self.ax.set_axis_off()
        self.ax.set_position([0,0,1,1])
        FigureCanvas.__init__(self, self.fig)
        self.setParent(parent)
        self.mpl_connect('button_press_event', self.onclick)
        self.wh = (0, 0)


    def newRect(self, xy, color):
        return patches.Rectangle((xy[0]-.45, xy[1]-.45), .9, .9,
                                 fill=False, color=color, linewidth=1)

    def onclick(self, event):
        if event.inaxes != self.ax:
            return
        x, y = (int(round(event.xdata)), int(round(event.ydata)))
        if not (0 <= x < self.wh[0] and 0 <= y < self.wh[1]):
            return
        self.clickPoint(x, y)


class WhiteLightWidget(PixelWidget):
    clickedPoint = pyqtSignal(int, int)

    def __init__(self, parent=None):
        PixelWidget.__init__(self, parent)
        FigureCanvas.setSizePolicy(self, QSizePolicy.Expanding, QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)
        self.selected = {}  # Map from (x,y) to Patch
        self.img = None

    @pyqtSlot(int, int)
    def removePoint(self, x, y):
        self.selected.pop((x, y)).remove()
        self.draw_idle()

    @pyqtSlot(int, int, str)
    def addPoint(self, x, y, color):
        xy = (x, y)
        p = self.newRect(xy, color)
        self.ax.add_patch(p)
        self.selected[xy] = p
        self.draw_idle()

    def clickPoint(self, x, y):
        self.clickedPoint.emit(x, y)

    def load(self, filename):
        if self.img is not None:
            self.img.remove()
            self.img = None
        try:
            img = plt.imread(filename)
            self.wh = (64, 64)
            self.img = self.ax.imshow(img, extent=(-.5, self.wh[0] - .5,
                                                   self.wh[1] - .5, -.5))
        except FileNotFoundError:
            pass
        self.draw_idle()


class ProjectionWidget(PixelWidget):
    removedPoint = pyqtSignal(int, int)
    addedPoint = pyqtSignal(int, int, str)
    alteredCounts = pyqtSignal(Counter)

    def __init__(self, parent=None):
        PixelWidget.__init__(self, parent)

        self.wavenumber = None
        self.raw = None
        self.projection = np.random.rand(16,16)
        self.cmap = 'hot'
        self.selected = {}  # Map from (x,y) to (annotation, Patch)
        self.curannot = 0
        self.mainimg = self.ax.imshow(self.projection, self.cmap, zorder=0)
        self.drawannot = None
        self.mpl_connect('button_release_event', self.onrelease)
        self.mpl_connect('motion_notify_event', self.onmotion)

    def setData(self, wn, raw, wh):
        self.wavenumber = wn
        self.raw = raw
        self.wh = wh

    def setAnnotation(self, annot):
        """
        Set the annotation to use for subsequently clicked points
        """
        self.curannot = annot

    def getWavenumbers(self):
        return self.wavenumber

    def setProjection(self, method, wavenumix):
        if self.wavenumber is None:
            return
        if method == 0:
            data = np.trapz(self.raw, self.wavenumber, axis=1)
        elif method == 1:
            data = self.raw.max(axis=1)
        elif method == 2:
            data = self.raw[:, wavenumix]
        data = data.reshape(self.wh)

        if self.projection.shape != data.shape:
            self.clearSelected()
            self.ax.clear()
            self.ax.set_axis_off()
            self.mainimg = self.ax.imshow(data, self.cmap, zorder=0)
        elif self.mainimg is not None:
            self.mainimg.set_data(data)
            self.mainimg.autoscale()
        self.projection = data
        self.draw_idle()

    def getAnnotations(self):
        ann = np.zeros(self.wh, dtype=int)
        for key, val in self.selected.items():
            ann[key] = val[0] + 1
        return ann

    def setAnnotations(self, ann):
        self.clearSelected()
        xys = np.transpose(np.nonzero(ann))
        for xy in xys:
            self.addPoint(tuple(xy), ann[tuple(xy)] - 1)

    def removePoint(self, xy):
        self.selected.pop(xy)[1].remove()
        self.removedPoint.emit(*xy)
        self.updateCounts()
        self.draw_idle()

    def rectColor(self, annot):
        colors = 'bgmykcrw'
        return colors[annot % len(colors)]

    def addPoint(self, xy, annot = None):
        if annot is None:
            annot = self.curannot
        color = self.rectColor(annot)
        p = self.newRect(xy, color)
        self.ax.add_patch(p)
        self.selected[xy] = (annot, p)
        self.addedPoint.emit(*xy, color)
        self.updateCounts()
        self.draw_idle()

    def clearSelected(self):
        for xy in self.selected.keys():
            self.removedPoint.emit(*xy)
        for ap in self.selected.values():
            ap[1].remove()
        self.selected = {}
        self.updateCounts()
        self.draw_idle()

    def updateCounts(self):
        self.alteredCounts.emit(Counter([x[0] for x in self.selected.values()]))

    @pyqtSlot('QString')
    def setCmap(self, s):
        self.cmap = s
        if self.mainimg:
            self.mainimg.set_cmap(s)
        self.draw_idle()

    def clickOnePoint(self, x, y):
        if (x, y) in self.selected:
            self.removePoint((x, y))
        else:
            self.addPoint((x, y))

    def clickPoint(self, x, y):
        if (x, y) in self.selected:
            self.removePoint((x, y))
            self.drawannot = False
        else:
            self.addPoint((x, y))
            self.drawannot = True

    def onmotion(self, event):
        if self.drawannot is None:
            return
        if event.inaxes != self.ax:
            return
        x, y = (int(round(event.xdata)), int(round(event.ydata)))
        if not (0 <= x < self.wh[0] and 0 <= y < self.wh[1]):
            return
        if (x, y) in self.selected:
            if not self.drawannot:
                self.removePoint((x, y))
        elif self.drawannot:
            self.addPoint((x, y))

    def onrelease(self, event):
        self.drawannot = None

