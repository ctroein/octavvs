#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 01:07:18 2019

@author: carl
"""

from PyQt5.QtCore import pyqtSignal, pyqtSlot
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import random
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min

class ProjectionWidget(FigureCanvas):
    changedSelected = pyqtSignal(int)

    def __init__(self, parent=None):
        self.wavenumber = None
        self.raw = None
        self.wh = None
        self.projection = np.random.rand(16,16)
        self.fig = Figure()
        self.ax = self.fig.add_subplot(111)
        self.ax.set_axis_off()
        self.ax.set_position([0,0,1,1])
        FigureCanvas.__init__(self, self.fig)
        self.cmap = 'hot'
        self.mpl_connect('button_press_event', self.onclick)
        # (x,y) -> Rectangle for the main plot
        self.selected = {}
        self.clustered = False
        self.popax = None
        # (x,y) -> Rectangle for pop-out, iff popax is not None
        self.popsel = {}
        self.mainimg = self.ax.imshow(self.projection, self.cmap, zorder=0, aspect='equal')
        self.popimg = None
        self.popcb = None
        self.popfig = None
        self.refreshplots = True

    def setData(self, wn, raw, wh):
#        if self.projection.shape != data.shape:
        if self.raw is None or self.raw.shape != raw.shape:
            self.clearSelected()
#            self.mainimg = None
#            self.popax = None
        self.wavenumber = wn
        self.raw = raw
        self.setDimensions(wh)

    def setDimensions(self, wh):
        p = wh[0] * wh[1]
        if p <= 0:
            return
        if wh[0] > 8 * wh[1] or wh[1] > 8 * wh[0]:
            h = int(np.sqrt(p))
            self.wh = ((p + h - 1) // h, h)
        else:
            self.wh = wh
        self.refreshplots = True
        self.draw_idle()

    def getWavenumbers(self):
        return self.wavenumber

    def getSelectedData(self):
        if self.raw is None:
            return None
        return self.raw[self.getSelected(), :]

    def setProjection(self, method, wavenumix):
        if self.wavenumber is None:
            return
        if method == 0:
            data = np.trapz(self.raw, self.wavenumber, axis=1)
            if self.wavenumber[0] > self.wavenumber[-1]:
                data = -data
        elif method == 1:
            data = self.raw.max(axis=1)
        elif method == 2:
            data = self.raw[:, wavenumix]
        px = self.wh[0] * self.wh[1]
        if px != len(data):
            d2 = np.empty((px))
            d2[:len(data)] = data
            d2[len(data):] = data.min()
            data = d2

        data = data.reshape(self.wh[::-1])
        if data.shape != self.projection.shape:
            sel = self.getSelected()
            self.projection = data
            self.setSelected(sel)
        else:
            self.projection = data
        self.refreshplots = True
        self.draw_idle()

    def clearSelected(self):
        for p in self.selected.values():
            p.remove()
        for p in self.popsel.values():
            p.remove()
        self.selected = {}
        self.popsel = {}

    def getSelected(self):
        return [xy[1] * self.projection.shape[1] + xy[0] for xy in self.selected.keys()]

    def removePoint(self, xy):
        self.selected.pop(xy).remove()
        if self.popax:
            self.popsel.pop(xy).remove()

    def newRect(self, xy):
        return patches.Rectangle((xy[0], xy[1]), 1, 1, fill=False, color='b')

    def addPoint(self, xy):
        px = xy[1] * self.projection.shape[1] + xy[0]
        if px >= len(self.raw) or px < 0:
            return False
        p = self.newRect(xy)
        self.ax.add_patch(p)
        self.selected[xy] = p
        if self.popax:
            p = self.newRect(xy)
            self.popax.add_patch(p)
            self.popsel[xy] = p
        return True

    def setSelected(self, sel):
        w = self.projection.shape[1]
        self.clearSelected()
        for xy in { (s % w, s // w) for s in sel }:
            self.addPoint(xy)
        self.draw_idle()
        self.changedSelected.emit(len(self.selected))

    def setSelectedCount(self, n, clustered):
        if self.raw is None:
            return
        if (n == len(self.selected) and clustered == self.clustered) and not self.refreshplots:
            return
        self.clustered = clustered
        if clustered and n > 0 and self.raw is not None:
            # (not wasc or n != len(self.selected)) and
            km = KMeans(n_clusters=min(n, len(self.raw)), n_init=1)
            km.fit(self.raw)
            closest, _ = pairwise_distances_argmin_min(km.cluster_centers_, self.raw)
            w = self.projection.shape[0]
            self.clearSelected()
            for xy in { (s % w, s // w) for s in closest }:
                self.addPoint(xy)
        if n < len(self.selected):
            for xy in list(self.selected.keys())[n:]:
                self.removePoint(xy)
        while len(self.selected) < n and len(self.selected) < len(self.raw):
            x = random.randrange(self.projection.shape[1])
            y = random.randrange(self.projection.shape[0])
            if (x, y) not in self.selected:
                self.addPoint((x, y))
        self.draw_idle()
        self.changedSelected.emit(len(self.selected))

    @pyqtSlot('QString')
    def setCmap(self, s):
        self.cmap = s
        if self.mainimg:
            self.mainimg.set_cmap(s)
        if self.popimg:
            self.popimg.set_cmap(s)
        self.draw_idle()

    def onclick(self, event):
        if not (event.inaxes == self.ax or event.inaxes == self.popax):
            return
        if event.xdata is None or self.wh is None:
            return
        hw = self.projection.shape
        x, y = (int(event.xdata), int(event.ydata))
        if not (0 <= x < hw[1] and 0 <= y < hw[0]):
            return
        if (x, y) in self.selected:
            self.removePoint((x, y))
        elif not self.addPoint((x, y)):
            return
        FigureCanvas.draw_idle(self)
        self.changedSelected.emit(len(self.selected))

    def draw(self):
        if self.refreshplots:
            self.mainimg.set_data(self.projection)
            self.mainimg.set_extent((0, self.projection.shape[1], self.projection.shape[0], 0))
            self.mainimg.autoscale()

        FigureCanvas.draw(self)
        if plt.fignum_exists(self.objectName()):
            self.popOut()
        self.refreshplots = False


    def popOut(self):
        setext = True
        fig = plt.figure(self.objectName(), tight_layout=dict(pad=.6))
        if self.popfig != fig:
            self.popfig = fig
            fig.canvas.mpl_connect('button_press_event', self.onclick)
            self.popax = fig.gca()
            self.popimg = self.popax.imshow(self.projection, self.cmap)
            self.popcb = fig.colorbar(self.popimg, ax=self.popax)
            setext = True

            self.popsel = {}
            for k in self.selected:
                p = self.newRect(k)
                self.popsel[k] = p
                self.popax.add_patch(p)
            fig.show()

        elif self.refreshplots:
            self.popimg.set_data(self.projection)
            self.popcb.mappable.set_clim(vmin=self.projection.min(), vmax=self.projection.max())
            setext = True

        if setext:
            self.popimg.set_extent((0, self.projection.shape[1], self.projection.shape[0], 0))
            self.popimg.autoscale()
        fig.canvas.draw_idle()

