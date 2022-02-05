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
from collections import OrderedDict
import numpy as np
import random
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min


# Silly naming covention in this class: camelCase for public functions,
# lower_case for private stuff
class ProjectionWidget(FigureCanvas):
    changedSelected = pyqtSignal(int)
    updatedProjection = pyqtSignal(np.ndarray)

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

        self.spatial = False    # Display in pixelxy-mode
        self.clustered = False  # Select based on cluster centers
        self.selected = OrderedDict() # keys indicate selected pixels
        self.pixel_levels = []  # Current intensity of each pixel
        self.cmap = plt.get_cmap().name # For img/fill
        self.plot_cmap = None # Name of cmap for borders, if any
        self.patches = {}       # Patch objects indexed by pixel number
        self.pop_patches = {}   # Ditto in pop-out plot
        # self.white_patches = {} # Ditto in white light subplot
        self.rect_size = 2      # Size of rectancles in µm
        self.rect_color = 'b'   # Default patch edge color
        self.pixel_visible = None # None, or True for pixels within image
        self.image = None       # ..io.Image object
        # Main/background image with placeholder data
        self.bgimg = self.ax.imshow(
            np.random.rand(16, 16), cmap=self.cmap, zorder=0, aspect='equal',
            origin='lower')
        self.pop_bgimg = None   # Ditto in pop-out plot
        self.pop_fig = None     # The pop-out Figure
        self.pop_ax = None      # ...and its Axes
        self.pop_cb = None      # ...and its Colorbar
        self.mpl_connect('button_press_event', self.onclick)

    def make_spatial_patches(self, patches, ax):
        rs = self.rect_size
        for xy in self.pixelxy:
            rect = matplotlib.patches.Rectangle(
                (xy[0] - rs/2, xy[1] - rs/2), rs, rs,
                fill=False, color=self.rect_color)
            ax.add_patch(rect)
            patches[len(patches)] = rect

    def setData(self, wn, raw, pixelxy=None):
        "Set spectral data and positions and clear selection"
        self.wavenumber = wn
        self.raw = raw
        self.pixelxy = pixelxy
        # New data implies cleared selection of pixels
        if self.spatial:
            for patches in [self.patches, self.pop_patches]:
                for p in patches.values():
                    p.remove()
            self.patches = {}
            self.pop_patches = {}
            self.selected = OrderedDict()
        else:
            self.clear_selected()
        self.spatial = pixelxy is not None
        if self.spatial:
            self.make_spatial_patches(self.patches, self.ax)
            if self.pop_fig:
                self.make_spatial_patches(self.pop_patches, self.pop_ax)
        self.pixel_visible = None

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
        self.trigger_redraw()

    def bgimglist(self):
        return [self.bgimg, self.pop_bgimg] if self.pop_bgimg \
            else [self.bgimg]

    def trigger_redraw(self):
        self.draw_idle()
        if self.pop_fig:
            self.pop_fig.canvas.draw_idle()


    def setImage(self, image):
        "Set the background or white light image"
        if image.fmt is not None:
            imgdata = plt.imread(BytesIO(image.data), format=image.fmt)
        elif image.data is not None:
            imgdata = image.data

        if self.spatial:
            hwh = np.array(image.wh) / 2
            for bgimg in self.bgimglist():
                bgimg.set_data(imgdata)
                bgimg.set_extent((image.xy[0] - hwh[0], image.xy[0] + hwh[0],
                                  image.xy[1] - hwh[1], image.xy[1] + hwh[1]))
                bgimg.autoscale()
            self.pixel_visible = [
                (np.abs(image.xy - xy) < (hwh + self.rect_size / 2)).all()
                for xy in self.pixelxy ]
            self.update_patch_fill()
        self.trigger_redraw()

    def setRectSize(self, size):
        if size <= 0:
            raise ValueError("Rectangle size must be positive")
        dpos = (self.rect_size - size) / 2
        self.rect_size = size
        for patches in [self.patches, self.pop_patches]:
            for p in patches.values():
                x, y = p.get_xy()
                p.set_bounds(x + dpos, y + dpos, size, size)
        self.trigger_redraw()

    def setProjection(self, method, wavenumix):
        "Update pixel colors. Methods 0=area, 1=max, 2=wavenum"
        # if self.wavenumber is None:
        #     return
        if method == 0:
            assert self.wavenumber is not None
            data = np.trapz(self.raw, self.wavenumber, axis=1)
            if not np.isfinite(data.sum()):
                data = np.trapz(np.nan_to_num(self.raw), self.wavenumber, axis=1)
            if self.wavenumber[0] > self.wavenumber[-1]:
                data = -data
        elif method == 1:
            data = np.nanmax(self.raw, axis=1)
        elif method == 2:
            data = self.raw[:, wavenumix]
        self.pixel_levels = data

        if self.spatial:
            self.update_patch_fill()
        else:
            px = self.wh[0] * self.wh[1]
            if px != len(data):
                d2 = np.empty((px))
                d2[:len(data)] = data
                d2[len(data):] = data.min()
                data = d2

            data = data.reshape(self.wh[::-1])
            for bgimg in self.bgimglist():
                bgimg.set_data(data)
                bgimg.set_extent((0, self.wh[0], 0, self.wh[1]))
                bgimg.autoscale()

        self.trigger_redraw()
        self.updatedProjection.emit(self.pixel_levels)

    def getWavenumbers(self):
        return self.wavenumber

    def visible_levels(self):
        "Get min and max visible pixel levels, for color range"
        vislevels = self.pixel_levels if self.pixel_visible is None \
            else self.pixel_levels[self.pixel_visible]
        return vislevels.min(), vislevels.max()

    def update_patch_fill(self):
        "Update fill color of pixelxy patches, normalized to visible range"
        if not self.spatial:
            return
        vislevels = self.visible_levels()
        colors = matplotlib.cm.ScalarMappable(
            norm=matplotlib.colors.Normalize(
                vmin=vislevels[0], vmax=vislevels[1], clip=True),
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
            if hasattr(cm, 'colors') and cm.N < 32:
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
        for bgimg in self.bgimglist():
            bgimg.set_cmap(s)
        self.trigger_redraw()

    def setFill(self, fill):
        "Fill/unfill pixelxy patches" # Todo: save/reset this state
        for p in self.patches.values():
            p.set_fill(fill)
        for p in self.pop_patches.values():
            p.set_fill(fill)
        self.trigger_redraw()

    def updatePlotColors(self, name=None):
        colors = self.setGlobalColorCycle(name)
        i = 0
        for s in self.selected.keys():
            c = colors[i % len(colors)]
            self.patches[s].set_edgecolor(c)
            if s in self.pop_patches:
                self.pop_patches[s].set_edgecolor(c)
            i = i + 1
        self.trigger_redraw()

    def getSelected(self):
        return list(self.selected.keys())

    def getSelectedData(self):
        if self.raw is None:
            return None
        return self.raw[self.getSelected(), :]

    def clear_selected(self):
        if self.spatial:
            for s in self.selected:
                self.patches[s].set_edgecolor(self.rect_color)
                if s in self.pop_patches:
                    self.pop_patches[s].set_edgecolor(self.rect_color)
        else:
            for p in self.patches.values():
                p.remove()
            self.patches = {}
            for p in self.pop_patches.values():
                p.remove()
            self.pop_patches = {}
        self.selected = OrderedDict()

    def new_patch(self, p, colorix):
        "Construct Patch for pixel p in non-spatial mode"
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        return matplotlib.patches.Rectangle(
            (p % self.wh[0], p // self.wh[0]), .999, .999, fill=False,
            color=colors[colorix % len(colors)])

    def color_patch(self, p, color):
        "Set edge color of patch(es) for pixel p"
        self.patches[p].set_edgecolor(color)
        if p in self.pop_patches:
            self.pop_patches[p].set_edgecolor(color)

    def select_point(self, p):
        assert p not in self.selected
        if self.spatial:
            colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
            c = colors[len(self.selected) % len(colors)]
            self.color_patch(p, c)
            self.selected[p] = True
        else:
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
        return True

    def deselect_point(self, p):
        del self.selected[p]
        if self.spatial:
            self.color_patch(p, self.rect_color)
        else:
            self.patches.pop(p).remove()
            if p in self.pop_patches:
                self.pop_patches.pop(p).remove()

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
        self.trigger_redraw()
        self.changedSelected.emit(len(self.selected))

    def onclick(self, event):
        if event.xdata is None:
            return
        if event.inaxes == self.ax:
            patches = self.patches
        elif event.inaxes == self.pop_ax:
            patches = self.pop_patches
        else:
            return
        changed = False  # Anything selected/deselected?
        if self.spatial:
            for p, pat in patches.items():
                if pat.contains(event)[0]:
                    changed = True
                    if p in self.selected:
                        self.deselect_point(p)
                    else:
                        self.select_point(p)
        elif self.wh is not None:
            x, y = (int(event.xdata), int(event.ydata))
            if not (0 <= x < self.wh[0] and 0 <= y < self.wh[1]):
                return
            p = x + y * self.wh[0]
            if p in self.selected:
                self.deselect_point(p)
                changed = True
            elif self.select_point(p):
                changed = True
        if changed:
            self.updatePlotColors()
            self.trigger_redraw()
            self.changedSelected.emit(len(self.selected))

    def onclose(self, event):
        "Pop-out window was close; stop drawing there"
        self.pop_fig = None
        self.pop_ax = None
        self.pop_bgimg = None
        self.pop_cb = None
        self.pop_patches = {}

    def draw(self):
        FigureCanvas.draw(self)
        # if plt.fignum_exists(self.objectName()):
        #     self.popOut()

    def popOut(self):
        fig = plt.figure(self.objectName(), tight_layout=dict(pad=.6))
        if self.pop_fig != fig:
            self.pop_fig = fig
            fig.canvas.mpl_connect('button_press_event', self.onclick)
            fig.canvas.mpl_connect('close_event', self.onclose)
            self.pop_ax = fig.gca()
            self.pop_bgimg = self.pop_ax.imshow(
                self.bgimg.get_array(), cmap=self.cmap,
                zorder=0, aspect='equal', extent=self.bgimg.get_extent(),
                origin='lower')
            self.pop_bgimg.autoscale()

            self.pop_patches = {}
            if self.spatial:
                self.make_spatial_patches(self.pop_patches, self.pop_ax)
            else:
                self.pop_cb = fig.colorbar(self.pop_bgimg, ax=self.pop_ax)
                for p in self.selected:
                    pat = self.new_patch(p, len(self.pop_patches))
                    self.pop_patches[p] = pat
                    self.pop_ax.add_patch(pat)

            fig.show()

