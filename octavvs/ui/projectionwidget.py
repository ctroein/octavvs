#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 01:07:18 2019

@author: carl
"""

from io import BytesIO
from PyQt5.QtCore import pyqtSignal, pyqtSlot
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib, cycler
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import OrderedDict
import numpy as np
import random
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min

class ProjectionWidgetBase(FigureCanvas):
    changedSelected = pyqtSignal(int)

    def __init__(self, parent=None):
        self.fig = Figure()
        self.ax = self.fig.add_subplot(111)
        self.ax.set_axis_off()
        self.ax.set_position([0,0,1,1])
        FigureCanvas.__init__(self, self.fig)
        self.wavenumber = None
        self.raw = None
        self.cmap = plt.get_cmap().name # For img/fill
        self.clustered = False
        # self.selected = OrderedDict() # contents vary by subclass
        self.plot_cmap = None # Name of cmap for borders, if any
        self.mpl_connect('button_press_event', self.onclick)

    def getWavenumbers(self):
        return self.wavenumber

    def getSelectedData(self):
        if self.raw is None:
            return None
        return self.raw[self.getSelected(), :]

    def setGlobalColorCycle(self, name):
        """
        Set the color used for line plots globally

        Parameters
        ----------
        name : str
            Name of cmap or "all [color]" or None to leave it unchanged.

        Returns
        -------
        cols : list
            List of colors to cycle through, or None if unchanged

        """
        if name is None:
            if self.plot_cmap is None:
                return plt.rcParams['axes.prop_cycle'].by_key()['color']
            name = self.plot_cmap
        self.plot_cmap = None
        if name[:4] == 'all ':
            cols = [matplotlib.colors.to_rgb(name[4:])]
        else:
            cm = plt.get_cmap(name)
            if hasattr(cm, 'colors'):
                cols = cm.colors
            else:
                n = max(len(self.selected), 1)
                d = .1 if name == 'gray' else 0
                cols = [cm(x) for x in np.linspace(d, 1-d, n)]
                self.plot_cmap = name
        plt.rcParams['axes.prop_cycle'] = cycler.cycler(
            'color', cols)
        return cols


class OldProjectionWidget(ProjectionWidgetBase):

    def __init__(self, parent=None):
        super().__init__(parent)
        self.wh = None
        self.projection = np.random.rand(16,16)

        # self.selected is (x,y) -> Rectangle for the main plot
        self.popax = None
        # (x,y) -> Rectangle for pop-out, iff popax is not None
        self.popsel = OrderedDict()
        self.mainimg = self.ax.imshow(self.projection, self.cmap,
                                      zorder=0, aspect='equal')
        self.popimg = None
        self.popcb = None
        self.popfig = None
        self.refreshplots = True

    def setData(self, wn, raw, wh):
        if self.raw is None or raw is None or self.raw.shape != raw.shape:
            self.clearSelected()
        self.wavenumber = wn
        self.raw = raw
        if wn is None:
            return
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
        self.selected = OrderedDict()
        self.popsel = OrderedDict()

    def getSelected(self):
        return [xy[1] * self.projection.shape[1] + xy[0]
                for xy in self.selected.keys()]

    def removePoint(self, xy):
        self.selected.pop(xy).remove()
        if self.popax:
            self.popsel.pop(xy).remove()

    def newRect(self, xy, colorix=0):
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        return patches.Rectangle((xy[0], xy[1]), .999, .999, fill=False,
                                 color=colors[colorix % len(colors)])

    def addPoint(self, xy):
        px = xy[1] * self.projection.shape[1] + xy[0]
        if px >= len(self.raw) or px < 0:
            return False
        pnum = len(self.selected)
        p = self.newRect(xy, pnum)
        self.ax.add_patch(p)
        self.selected[xy] = p
        if self.popax:
            p = self.newRect(xy, pnum)
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
        if n > len(self.raw):
            n = len(self.raw)
        if (n == len(self.selected) and clustered == self.clustered
            ) and not self.refreshplots:
            return
        self.clustered = clustered
        if clustered and n > 0:
            # (not wasc or n != len(self.selected)) and
            km = KMeans(n_clusters=min(n, len(self.raw)), n_init=1)
            km.fit(self.raw)
            closest, _ = pairwise_distances_argmin_min(
                km.cluster_centers_, self.raw)
            w = self.projection.shape[0]
            self.clearSelected()
            for xy in { (s % w, s // w) for s in closest }:
                self.addPoint(xy)
        if n < len(self.selected):
            for xy in list(self.selected.keys())[n:]:
                self.removePoint(xy)
        while len(self.selected) < n:
            x = random.randrange(self.projection.shape[1])
            y = random.randrange(self.projection.shape[0])
            if (x, y) not in self.selected:
                self.addPoint((x, y))
        self.updatePlotColors()
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

    def updatePlotColors(self, name=None):
        colors = self.setGlobalColorCycle(name)
        if colors is None:
            return
        i = 0
        for p in self.selected.values():
            p.set_color(colors[i % len(colors)])
            i = i + 1
        i = 0
        for p in self.popsel.values():
            p.set_color(colors[i % len(colors)])
            i = i + 1
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
        self.updatePlotColors()
        FigureCanvas.draw_idle(self)
        self.changedSelected.emit(len(self.selected))

    def draw(self):
        if self.refreshplots:
            self.mainimg.set_data(self.projection)
            self.mainimg.set_extent((0, self.projection.shape[1],
                                     self.projection.shape[0], 0))
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

            self.popsel = OrderedDict()
            for k in self.selected:
                p = self.newRect(k)
                self.popsel[k] = p
                self.popax.add_patch(p)
            fig.show()

        elif self.refreshplots:
            self.popimg.set_data(self.projection)
            self.popcb.mappable.set_clim(vmin=self.projection.min(),
                                         vmax=self.projection.max())
            setext = True

        if setext:
            self.popimg.set_extent((0, self.projection.shape[1],
                                    self.projection.shape[0], 0))
            self.popimg.autoscale()
        fig.canvas.draw_idle()



class ProjectionWidget(ProjectionWidgetBase):
    changedSelected = pyqtSignal(int)

    def __init__(self, parent=None):

        self.fig = Figure()
        self.ax = self.fig.add_subplot(111)
        self.ax.set_axis_off()
        self.ax.set_position([0,0,1,1])
        FigureCanvas.__init__(self, self.fig)

        self.wavenumber = None
        self.raw = None
        self.wavenumber = None  # Wavenumbers, used for area-under-curve
        self.wh = None          # Image shape for non-pixelxy
        self.pixelxy = None     # (x,y) for all pixels or None

        self.clustered = False
        self.selected = OrderedDict() # keys indicate selected pixels
        self.pixel_levels = []  # Current intensity of each pixel
        self.cmap = plt.get_cmap().name # For img/fill
        self.plot_cmap = None # Name of cmap for borders, if any
        self.patches = {}       # Patch objects indexed by pixel number
        self.pop_patches = {}   # Ditto in pop-out plot
        # self.white_patches = {} # Ditto in white light subplot
        self.rect_size = 4      # Size of rectancles in µm
        self.rect_color = 'b'   # Default patch edge color
        self.pixel_visible = None # None, or True for pixels within image
        self.image = None       # ..io.Image object
        # Main/background image with placeholder data
        self.bgimg = self.ax.imshow(
            np.random.rand(16, 16), cmap=self.cmap, zorder=0, aspect='equal')
        self.pop_bgimg = None   # Ditto in pop-out plot
        self.pop_fig = None     # The pop-out Figure
        self.pop_ax = None      # ...and its Axes
        self.pop_cb = None      # ...and its Colorbar
        self.mpl_connect('button_press_event', self.onclick)
        # self.selected is (x,y) -> Rectangle for the main plot
        # self.popax = None
        # (x,y) -> Rectangle for pop-out, iff popax is not None
        # self.popsel = OrderedDict()
        # self.mainimg = self.ax.imshow(self.projection, self.cmap,
                                      # zorder=0, aspect='equal')
        # self.refreshplots = True

    def setData(self, wn, raw, pixelxy=None):
        "Set spectral data and positions and clear selection"
        self.clear_selected() # New data implies cleared selection of pixels
        self.wavenumber = wn
        self.raw = raw
        self.pixelxy = pixelxy
        for p in self.patches.values():
            p.remove()
        self.patches = {}
        rs = self.rect_size
        if self.pixelxy is not None:
            for xy in self.pixelxy:
                rect = patches.Rectangle(
                    (xy[0] - rs/2, xy[1] - rs/2), rs, rs,
                    fill=False, color=self.rect_color)
                self.ax.add_patch(rect)
                self.patches[len(self.patches)] = rect
        self.pixels_visible = None

    def setDimensions(self, wh):
        "Set the shape of the image, with override if too unsquare"
        p = wh[0] * wh[1]
        if p <= 0:
            return
        if wh[0] > 8 * wh[1] or wh[1] > 8 * wh[0]:
            h = int(np.sqrt(p))
            self.wh = ((p + h - 1) // h, h)
        else:
            self.wh = wh
        self.draw_idle()

    def setImage(self, image):
        "Set the background or white light image"
        if image.fmt is not None:
            imgdata = plt.imread(BytesIO(image.data), format=image.fmt)
        elif image.data is not None:
            imgdata = image.data
        self.bgimg.set_data(imgdata)
        hwh = np.array(image.wh) / 2
        self.bgimg.set_extent((image.xy[0] - hwh[0], image.xy[0] + hwh[0],
                                image.xy[1] + hwh[1], image.xy[1] - hwh[1]))
        self.bgimg.autoscale()
        if self.pixelxy is not None:
            self.pixel_visible = [
                (np.abs(image.xy - xy) < (hwh + self.rect_size / 2)).all()
                for xy in self.pixelxy ]
            self.update_patch_fill()
        self.draw_idle()

    def setProjection(self, method, wavenumix):
        "Update pixel colors. Methods 0=area, 1=max, 2=wavenum"
        # if self.wavenumber is None:
        #     return
        if method == 0:
            assert self.wavenumber is not None
            data = np.trapz(self.raw, self.wavenumber, axis=1)
            if self.wavenumber[0] > self.wavenumber[-1]:
                data = -data
        elif method == 1:
            data = self.raw.max(axis=1)
        elif method == 2:
            data = self.raw[:, wavenumix]
        self.pixel_levels = data

        if self.pixelxy is None:
            px = self.wh[0] * self.wh[1]
            if px != len(data):
                d2 = np.empty((px))
                d2[:len(data)] = data
                d2[len(data):] = data.min()
                data = d2

            data = data.reshape(self.wh[::-1])
            self.bgimg.set_data(data)
            self.bgimg.set_extent((0, self.wh[0], self.wh[1], 0))
            self.bgimg.autoscale()
        else:
            self.update_patch_fill()
        self.draw_idle()

    def getWavenumbers(self):
        return self.wavenumber

    def update_patch_fill(self):
        "Update fill color of pixelxy patches, normalized to visible range"
        if self.pixel_visible is None:
            return
        vislevels = self.pixel_levels[self.pixel_visible]
        colors = matplotlib.cm.ScalarMappable(
            norm=matplotlib.colors.Normalize(
                vmin=vislevels.min(), vmax=vislevels.max(), clip=True),
            cmap=self.cmap).to_rgba(self.pixel_levels)
        for n, p in self.patches.items():
            p.set_facecolor(colors[n])
        for n, p in self.pop_patches.items():
            p.set_facecolor(colors[n])

    def setGlobalColorCycle(self, name):
        """
        Set the color used for line plots globally

        Parameters
        ----------
        name : str
            Name of cmap or "all [color]" or None to leave it unchanged.

        Returns
        -------
        cols : list
            List of colors to cycle through, or None if unchanged
        """
        if name is None:
            if self.plot_cmap is None:
                return plt.rcParams['axes.prop_cycle'].by_key()['color']
            name = self.plot_cmap
        self.plot_cmap = None
        if name[:4] == 'all ':
            cols = [matplotlib.colors.to_rgb(name[4:])]
        else:
            cm = plt.get_cmap(name)
            if hasattr(cm, 'colors'):
                cols = cm.colors
            else:
                n = max(len(self.selected), 1)
                d = .1 if name == 'gray' else 0
                cols = [cm(x) for x in np.linspace(d, 1-d, n)]
                self.plot_cmap = name
        plt.rcParams['axes.prop_cycle'] = cycler.cycler(
            'color', cols)
        return cols

    @pyqtSlot('QString')
    def setCmap(self, s):
        "Update colors of background image"
        self.cmap = s
        self.update_patch_fill()
        self.bgimg.set_cmap(s)
        self.draw_idle()

    def setFill(self, fill):
        "Fill/unfill pixelxy patches" # Todo: save/reset this state
        for p in self.patches.items():
            p.set_fill(fill)
        for p in self.pop_patches.items():
            p.set_fill(fill)
        self.draw_idle()

    def updatePlotColors(self, name=None):
        colors = self.setGlobalColorCycle(name)
        # if colors is None:
        #     return
        i = 0
        for s in self.selected.keys():
            c = colors[i % len(colors)]
            self.patches[s].set_edgecolor(c)
            if s in self.pop_patches:
                self.pop_patches[s].set_edgecolor(c)
            i = i + 1
        self.draw_idle()

    def getSelected(self):
        return list(self.selected.keys())

    def getSelectedData(self):
        if self.raw is None:
            return None
        return self.raw[self.getSelected(), :]


    def clear_selected(self):
        if self.pixelxy is None:
            for p in self.patches.values():
                p.remove()
            self.patches = {}
            for p in self.pop_patches.values():
                p.remove()
            self.pop_patches = {}
        else:
            for s in self.selected:
                self.patches[s].set_edgecolor(self.rect_color)
                if s in self.pop_patches:
                    self.pop_patches[s].set_edgecolor(self.rect_color)
        self.selected = OrderedDict()

    def new_patch(self, p, colorix):
        "Construct Patch for pixel p in non-pixelxy mode"
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        return patches.Rectangle(
            (p % self.wh[0], p // self.wh[0]), .999, .999, fill=False,
            color=colors[colorix % len(colors)])

    def color_patch(self, p, color):
        "Set edge color of patch(es) for pixel p"
        self.patches[p].set_edgecolor(color)
        if p in self.pop_patches:
            self.pop_patches[p].set_edgecolor(color)

    def select_point(self, p):
        assert p not in self.selected
        if self.pixelxy is None:
            if p < 0 or p >= len(self.raw):
                return False
            pnum = len(self.selected)
            pat = self.new_patch(p, pnum)
            self.patches[p] = pat
            self.ax.add_patch(pat)
            self.selected[p] = True
            if self.pop_ax is not None:
                pat = self.new_patch(p, pnum)
                self.pop_patches[p] = pat
                self.pop_ax.add_patch(pat)
        else:
            colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
            c = colors[len(self.selected) % len(colors)]
            self.color_patch(p, c)
            self.selected[p] = True
        return True

    def deselect_point(self, p):
        del self.selected[p]
        if self.pixelxy is None:
            self.patches.pop(p).remove()
            if p in self.pop_patches:
                self.pop_patches.pop(p).remove()
        else:
            self.color_patch(p, self.rect_color)

    def setSelectedCount(self, n, clustered):
        if self.raw is None:
            return
        if n > len(self.raw):
            n = len(self.raw)
        if n == len(self.selected) and clustered == self.clustered:
            return
        self.clustered = clustered
        if n == len(self.raw):
            for p in range(len(self.raw)):
                if p not in self.selected:
                    self.select_point(p)
        elif clustered and n > 0:
            # (not wasc or n != len(self.selected)) and
            km = KMeans(n_clusters=min(n, len(self.raw)), n_init=1)
            km.fit(self.raw)
            closest = pairwise_distances_argmin_min(
                km.cluster_centers_, self.raw)[0]
            self.clear_selected()
            for p in closest:
                self.select_point(p)
        if n < len(self.selected):
            for p in list(self.selected.keys())[n:]:
                self.deselect_point(p)
        while len(self.selected) < n:
            p = random.randrange(len(self.raw))
            if p not in self.selected:
                self.select_point(p)
        self.updatePlotColors()
        self.draw_idle()
        self.changedSelected.emit(len(self.selected))

    def onclick(self, event):
        if not (event.inaxes == self.ax or event.inaxes == self.pop_ax):
            return
        if event.xdata is None or self.raw is None:
            return
        changed = False
        if self.pixelxy is None:
            x, y = (int(event.xdata), int(event.ydata))
            if not (0 <= x < self.wh[0] and 0 <= y < self.wh[1]):
                return
            p = x + y * self.wh[0]
            if p in self.selected:
                self.deselect_point(p)
                changed = True
            else:
                changed = self.select_point(p)
        else:
            for p, pat in self.patches.items():
                if pat.contains(event)[0]:
                    changed = True
                    if p in self.selected:
                        self.deselect_point(p)
                    else:
                        self.select_point(p)
        if changed:
            self.updatePlotColors()
            FigureCanvas.draw_idle(self)
            self.changedSelected.emit(len(self.selected))


    def draw(self):
        FigureCanvas.draw(self)
        if plt.fignum_exists(self.objectName()):
            self.popOut()


    def popOut(self):
        pass
    #     setext = True
    #     fig = plt.figure(self.objectName(), tight_layout=dict(pad=.6))
    #     if self.pop_fig != fig:
    #         self.pop_fig = fig
    #         fig.canvas.mpl_connect('button_press_event', self.onclick)
    #         self.pop_ax = fig.gca()
    #         self.pop_img = self.pop_ax.imshow(self.projection, self.cmap)
    #         self.popcb = fig.colorbar(self.popimg, ax=self.popax)
    #         setext = True

    #         self.popsel = OrderedDict()
    #         for k in self.selected:
    #             p = self.newRect(k)
    #             self.popsel[k] = p
    #             self.popax.add_patch(p)
    #         fig.show()

    #     elif self.refreshplots:
    #         self.popimg.set_data(self.projection)
    #         self.popcb.mappable.set_clim(vmin=self.projection.min(),
    #                                      vmax=self.projection.max())
    #         setext = True

    #     if setext:
    #         self.popimg.set_extent((0, self.projection.shape[1],
    #                                 self.projection.shape[0], 0))
    #         self.popimg.autoscale()
    #     fig.canvas.draw_idle()
