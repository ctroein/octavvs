from PyQt5.QtWidgets import QSizePolicy
from PyQt5.QtCore import pyqtSignal, pyqtSlot
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import math
import skimage.draw
from ..algorithms.util import pixels_fit

class BasePlotWidget(FigureCanvas):
    def __init__(self, parent=None):
        self.fig = Figure(tight_layout=dict(pad=0.4))
        self.ax = self.fig.add_subplot(111)
        FigureCanvas.__init__(self, self.fig)
        self.setParent(parent)
        FigureCanvas.setSizePolicy(
            self, QSizePolicy.Expanding, QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)

        # 2D layout
        self.wh = None
        self.pixelxy = None
        self.pixels = 0
        # What/how to plot
        self.cmap = plt.get_cmap().name
        self.pixel_size = 2

    def set_basic_data(self, pixels=None, wh=None, pixelxy=None):
        "Set shape/points and clears ROI etc"
        if wh is None and pixelxy is None:
            raise ValueError('Either wh or pixelxy must be specified')
        if pixelxy is not None:
            pixels = len(pixelxy)
        elif pixels is None:
            pixels = wh[0] * wh[1]
        else:
            assert pixels_fit(pixels, wh)
        self.wh = wh
        self.pixelxy = pixelxy
        self.pixels = pixels

    def rebox_data(self, data):
        "Make a reshaped data array that fits in w*h(*a)"
        nbox = self.wh[1] * self.wh[0]
        if nbox > self.pixels:
            shape = list(data.shape)
            shape[0] = nbox
            data2d = np.zeros(shape=shape, dtype=data.dtype)
            data2d[:self.pixels, ...] = data
            if len(shape) == 1:
                data2d[self.pixels:, ...] = data.min()
        else:
            data2d = data
        shape = [self.wh[1], self.wh[0]] + list(data.shape[1:])
        return data2d.reshape(shape)


class RoiPlotWidget(BasePlotWidget):
    updated = pyqtSignal(int, int)  # N out of M selected

    def __init__(self, parent=None):
        super().__init__(parent)

        self.window_title = 'Region of interest'
        # Data to plot
        self.data = None
        self.roi = None
        self.roi_alpha = np.array([130, 140, 180, 120], np.uint8) # RGBA
        self.polystart = None   # Patch created when starting polygon
        self.polyline = None    # Line2D for lines in polygon
        self.drawmode = 'click' # click, add, remove
        self.drawing = False    # Drawing with the mouse?
        self.patches = None     # list of Patch for xy-pixels
        self.mpl_connect('button_press_event', self.on_click)
        self.mpl_connect('button_release_event', self.on_release)
        self.mpl_connect('motion_notify_event', self.on_move)

    def emit_updated(self):
        n = 0 if self.roi is None else self.roi.sum()
        self.updated.emit(n, self.pixels)

    def set_basic_data(self, pixels=None, wh=None, pixelxy=None):
        "Set shape/points and clears ROI etc"
        super().set_basic_data(pixels=pixels, wh=wh, pixelxy=pixelxy)
        self.data = None
        self.roi = np.zeros((self.pixels), dtype=bool)
        self.polyline = None
        self.polystart = None
        self.patches = []
        self.ax.clear()
        self.emit_updated()
        self.draw_idle()

    def adjust_geometry(self, wh):
        "Update the plot geometry, keeping the pixel count"
        assert pixels_fit(self.pixels, wh)
        self.wh = wh
        self.set_data(self.data)

    def get_roi(self):
        "Return the ROI as a bool array (not a copy; beware)"
        return self.roi

    def set_roi(self, roi):
        "Set the ROI from an array or None"
        if roi is None:
            self.roi = np.zeros(self.pixels, dtype=bool)
        else:
            roi = np.array(roi, dtype=bool)
            if roi.shape != self.roi.shape:
                raise ValueError('ROI size must match the current geometry')
            self.roi = roi
        self.update_roi_image()
        self.emit_updated()
        self.draw_idle()

    @pyqtSlot(np.ndarray)
    def set_data(self, data):
        "Set intensities for all pixels"
        assert len(data) == self.pixels
        # (Re)create basic stuff in plot only when we get data
        if self.data is None:
            # Things common to both display modes
            self.ax.set_aspect('equal')
            self.ax.tick_params(bottom=False, labelbottom=False,
                                left=False, labelleft=False)
            self.ax.set_facecolor('#eee')

        if self.pixelxy is not None:
            cmap = plt.get_cmap(self.cmap)
            norm = matplotlib.colors.Normalize(
                vmin=data.min(), vmax=data.max())
            if self.data is None:
                minxy = np.min(self.pixelxy, axis=0)
                maxxy = np.max(self.pixelxy, axis=0)
                ps = self.pixel_size / 2
                for i in range(self.pixels):
                    p = matplotlib.patches.Rectangle(
                        self.pixelxy[i] - ps, ps, ps,
                        facecolor=cmap(norm(data[i])),
                        linewidth=2, zorder=0)
                    self.ax.add_patch(p)
                    self.patches.append(p)
                    self.select_pixel(i, self.roi[i])
                m = .05 * ((maxxy - minxy).max()) + ps
                self.ax.set_xlim(minxy[0] - m, maxxy[0] + m)
                self.ax.set_ylim(minxy[1] - m, maxxy[1] + m)
            else:
                for i in range(self.pixels):
                    self.patches[i].set_facecolor(cmap(norm(data[i])))
        else:
            data2d = self.rebox_data(data)
            if self.data is None:
                im = self.ax.imshow(data2d, cmap=self.cmap, zorder=0,
                                    origin='lower')
                im.set_extent((0, self.wh[0], 0, self.wh[1]))
                # Turn roi into rgba
                roi2d = np.outer(self.roi, self.roi_alpha)
                roi2d = self.rebox_data(roi2d)
                rim = self.ax.imshow(roi2d, zorder=2, origin='lower')
                rim.set_extent((0, self.wh[0], 0, self.wh[1]))
            else:
                im = self.ax.get_images()[0]
                im.set_data(data2d)
                # im.set_clim(data2d.min(), data2d.max()) # test
                im.set_extent((0, self.wh[0], 0, self.wh[1]))
                im = self.ax.get_images()[1]
                im.set_extent((0, self.wh[0], 0, self.wh[1]))
                im.autoscale()
            self.ax.autoscale()
            m = .05 * np.max(self.wh) + self.pixel_size
            self.ax.set_xlim(-m, self.wh[0] + m)
            self.ax.set_ylim(-m, self.wh[1] + m)

        self.data = data
        self.draw_idle()

    @pyqtSlot(float)
    def set_pixel_size(self, s):
        "Set the size of patches and redraw"
        dpos = (self.pixel_size - s) / 2
        self.pixel_size = s
        if self.patches is not None:
            for p in self.patches:
                x, y = p.get_xy()
                p.set_bounds(x + dpos, y + dpos, s, s)
        if self.polystart:
            self.polystart.set_radius(s)
        self.draw_idle()

    @pyqtSlot(str)
    def set_cmap(self, s):
        "Set colors for heatmaps"
        self.cmap = s
        for im in self.ax.get_images():
            im.set_cmap(s)
        self.draw_idle()

    def stop_polygon(self):
        "Remove polygon lines"
        if self.polyline is not None:
            self.ax.lines.remove(self.polyline)
            self.polyline = None
        if self.polystart is not None:
            self.polystart.remove()
            self.polystart = None

    def select_pixel(self, i, sign):
        self.roi[i] = sign
        self.patches[i].set_edgecolor('red' if sign else None)

    def update_roi_image(self):
        if self.data is None:
            return
        if self.pixelxy is not None:
            for i in range(self.pixels):
                self.select_pixel(i, self.roi[i])
        else:
            roi2d = np.outer(self.roi, self.roi_alpha)
            roi2d = self.rebox_data(roi2d)
            self.ax.get_images()[1].set_data(roi2d)

    def invert_roi(self):
        if self.data is None:
            return
        self.roi = ~self.roi
        self.update_roi_image()
        self.emit_updated()
        self.draw_idle()

    def clear(self):
        if self.data is None:
            return
        if self.pixelxy is not None:
            for i in range(self.pixels):
                if self.roi[i]:
                    self.select_pixel(i, False)
        else:
            self.roi.fill(False)
            self.update_roi_image()
        self.stop_polygon()
        self.emit_updated()
        self.draw_idle()

    def get_draw_mode(self):
        return self.drawmode

    def set_draw_mode(self, mode):
        assert mode in ['click', 'add', 'remove']
        if mode == 'click' and self.drawmode != mode:
            self.stop_polygon()
            self.draw_idle()
        self.drawmode = mode

    def erase_last_point(self):
        "Remove the last point from the unfinished polygon"
        if self.polyline is not None:
            xy = self.polyline.get_data(orig=True)
            if len(xy[0]) > 1:
                self.polyline.set_data(xy[0][:-1], xy[1][:-1])
            else:
                self.stop_polygon()
            self.draw_idle()

    def polygon_click(self, event):
        "Add polygon point"
        if event.button == 3: # Right
            self.erase_last_point()
        if event.button != 1: # Left
            self.drawing = False
            return

        upd = False
        exy = event.xdata, event.ydata
        if self.polyline is None:
            self.polyline = self.ax.plot(
                [exy[0]], [exy[1]], lw=1, color='k', zorder=1)[0]
            self.polystart = matplotlib.patches.RegularPolygon(
                exy, numVertices=5, radius=self.pixel_size, lw=2,
                ec='black', fc='yellow', fill=True, zorder=2)
            self.ax.add_patch(self.polystart)
            self.drawing = True
        elif self.polystart.contains(event)[0]:
            pxy = self.polyline.get_xydata()
            if len(pxy) > 2:
                sign = self.drawmode == 'add'
                if self.pixelxy is None:
                    ppx, ppy = skimage.draw.polygon(
                        pxy[:,0], pxy[:,1], shape=self.wh)
                    pp = ppx + ppy * self.wh[0]
                    self.roi[pp[pp < len(self.roi)]] = sign
                    self.update_roi_image()
                else:
                    pg = matplotlib.patches.Polygon(pxy)
                    for i in range(self.pixels):
                        if pg.contains_point(self.pixelxy[i]):
                            self.select_pixel(i, sign)
            # Do nothing only if we just started drawing
            if len(pxy) > 1 or not self.drawing:
                upd = True
                self.stop_polygon()
                self.drawing = False
        else:
            xy = self.polyline.get_data(orig=True)
            d2 = (xy[0][-1] - exy[0]) ** 2 + (xy[1][-1] - exy[1]) ** 2
            if not self.drawing or d2 > self.pixel_size ** 2:
                self.polyline.set_data(np.append(xy[0], exy[0]),
                                       np.append(xy[1], exy[1]))
                self.drawing = True
        if upd:
            self.emit_updated()
        self.draw_idle()

    def regular_click(self, event):
        upd = False
        if self.pixelxy is not None:
            for i in range(self.pixels):
                if self.patches[i].contains(event)[0]:
                    if not self.drawing:
                        self.drawsign = not self.roi[i]
                        self.drawing = True
                    if self.roi[i] != self.drawsign:
                        self.select_pixel(i, self.drawsign)
                        upd = True
        else:
            x, y = (math.floor(event.xdata), math.floor(event.ydata))
            if (0 <= x < self.wh[0] and 0 <= y < self.wh[1]):
                i = x + y * self.wh[0]
                if i < len(self.roi):
                    if not self.drawing:
                        self.drawsign = not self.roi[i]
                        self.drawing = True
                    if self.roi[i] != self.drawsign:
                        self.roi[i] = self.drawsign
                        upd = True
                        self.update_roi_image()
        if upd:
            self.emit_updated()
            self.draw_idle()

    def on_click(self, event):
        if self.data is None or event.inaxes != self.ax:
            return
        if self.drawing: # Click while drawing? We'd better just stop.
            self.drawing = False
        elif self.drawmode == 'click':
            self.regular_click(event)
        else:
            self.polygon_click(event)

    def on_release(self, event):
        self.drawing = False

    def on_move(self, event):
        if not self.drawing or event.inaxes != self.ax:
            return
        if self.drawmode == 'click':
            self.regular_click(event)
        else:
            self.polygon_click(event)


class DecompositionPlotWidget(BasePlotWidget):
    displayModesUpdated = pyqtSignal(list)

    def __init__(self, parent=None):
        super().__init__(parent)

        self.window_title = 'Decomposition'
        # Data to plot
        self.wn = None
        self.roi = None
        # What/how to plot
        self.discrete_cmap = 'tab10'
        self.display_mode = 0 # Index for now - should perhaps be string
        self.clear_data()

    def clear_data(self):
        self.error_log = []
        self.init_concentrations = None
        self.init_spectra = None
        self.concentrations = None
        self.spectra = None
        self.cluster_maps = {}
        self.update_display_modes()

    def mousePressEvent(self, event):
        self.popOut()

    def update_display_modes(self):
        "Create/update the list of display modes and signal the change"
        m = ['Error progress']
        if self.concentrations is not None:
            m = m + [ 'Initial spectra', 'Spectra', 'Contributions' ]
            for i in range(len(self.concentrations)):
                m.append('Contribution %d' % (i+1))
            m = m + list(self.cluster_maps.keys())
        self.display_modes = m
        if self.display_mode <= len(m):
            self.display_mode = len(m) - 1
        self.displayModesUpdated.emit(self.display_modes)

    @pyqtSlot(int)
    def set_display_mode(self, m):
        assert m < len(self.display_modes)
        self.display_mode = m
        self.draw_idle()

    def set_basic_data(self, wn, pixels=None, wh=None, pixelxy=None):
        "Set wavenumbers and geometric data that won't change for this file"
        super().set_basic_data(pixels=pixels, wh=wh, pixelxy=pixelxy)
        self.wn = wn
        self.clear_and_set_roi()

    def adjust_geometry(self, wh):
        "Change shape without altering pixel count"
        assert pixels_fit(self.pixels, wh)
        self.wh = wh
        self.draw_idle()

    def clear_and_set_roi(self, roi=None):
        "Set ROI and clear spectra/concentrations"
        if roi is not None:
            assert len(roi) == self.pixels
        self.roi = roi
        self.clear_data()
        self.draw_idle()

    def set_concentrations(self, concentrations):
        """
        Set concentration matrix from decomposition

        Parameters
        ----------
        concentrations : array(ncomponents, npixels)
            Note the order.

        Returns
        -------
        None.
        """
        assert self.wn is not None
        if concentrations is None:
            self.concentrations = None
            self.draw_idle()
            return
        comps, pixels = concentrations.shape
        udm = self.concentrations is None or len(self.concentrations) != comps
        if self.roi is None:
            assert pixels == self.pixels
            self.concentrations = concentrations
        else:
            assert pixels == self.roi.sum()
            if udm:
                self.concentrations = np.empty((comps, self.pixels))
            self.concentrations[...] = concentrations.min(1)[:,None]
            self.concentrations[:, self.roi] = concentrations

        self.add_clustering('Strongest contribution',
                            np.argmax(self.concentrations, axis=0))
        if udm:
            self.update_display_modes()
        self.draw_idle()

    def get_concentrations(self):
        return self.concentrations

    def set_spectra(self, spectra):
        self.spectra = spectra
        self.draw_idle()

    def set_initial_concentrations(self, concentrations):
        self.init_concentrations = concentrations
        self.draw_idle()

    def set_initial_spectra(self, spectra):
        self.init_spectra = spectra
        self.draw_idle()

    def set_errors(self, errors):
        "Set the list/array of errors"
        self.error_log = errors
        if self.display_mode == 0:
            self.draw_idle()

    @pyqtSlot(float)
    def set_pixel_size(self, s):
        "Set the size of patches and redraw"
        dpos = (self.pixel_size - s) / 2
        self.pixel_size = s
        for p in self.ax.patches:
            x, y = p.get_xy()
            p.set_bounds(x + dpos, y + dpos, s, s)
        self.draw_idle()

    @pyqtSlot('QString')
    def set_cmap(self, s):
        "Set colors for heatmaps"
        self.cmap = s
        self.draw_idle()

    @pyqtSlot('QString')
    def set_discrete_cmap(self, s):
        "Set colors for clustering etc"
        self.discrete_cmap = s
        self.draw_idle()

    def add_clustering(self, name, clustered):
        assert len(clustered) == self.pixels
        self.cluster_maps[name] = clustered
        if name not in self.display_modes:
            self.update_display_modes()

    def draw_error_log(self, ax):
        ax.set_yscale('log')
        ax.xaxis.set_major_locator(
            matplotlib.ticker.MaxNLocator(integer=True))
        ax.set_xlabel('Iteration')
        if self.error_log is None:
            return
        if len(self.error_log):
            ax.plot(self.error_log, label='Error')
            ax.legend()

    def draw_init_spectra(self, ax):
        ax.set_yscale('linear')
        if self.init_spectra is not None:
            for i, s in enumerate(self.init_spectra):
                ax.plot(self.wn, s,
                        label='Initial component %d' % (i + 1))
            ax.legend()

    def draw_spectra(self, ax):
        ax.set_yscale('linear')
        if self.spectra is not None:
            for i, s in enumerate(self.spectra):
                ax.plot(self.wn, s, label='Component %d' % (i + 1))
            ax.legend()

    def draw_concentrations(self, ax):
        ax.set_yscale('linear')
        if self.concentrations is not None:
            for i, s in enumerate(self.concentrations):
                if s.any():
                    ax.plot(s, label='Component %d' % (i + 1))
            ax.legend()

    def draw_heatmap(self, ax, data, discrete=False, cbfig=None):
        if discrete:
            lc = int(data.max())
            cminmax = (0, lc)
            cmap = self.discrete_cmap
            cm = plt.get_cmap(cmap)
            if hasattr(cm, 'colors') and cm.N < 32:
                # Reshape color list to match the input values
                cmap = matplotlib.colors.ListedColormap(cm.colors, N=lc+1)
        else:
            cmap = plt.get_cmap(self.cmap)
            cminmax = data.min(), data.max()

        if self.pixelxy is not None:
            norm = matplotlib.colors.Normalize(
                vmin=cminmax[0], vmax=cminmax[1])
            patches = ax.patches
            if patches:
                for i in range(len(patches)):
                    patches[i].set_facecolor(cmap(norm(data[i])))
            else:
                minxy = np.min(self.pixelxy, axis=0)
                maxxy = np.max(self.pixelxy, axis=0)
                r = self.pixel_size
                ax.clear()
                ax.set_aspect('equal')
                for i in range(len(self.pixelxy)):
                    xy = self.pixelxy[i]
                    p = matplotlib.patches.Rectangle(
                        xy - r / 2, r, r, facecolor=cmap(norm(data[i])))
                    ax.add_patch(p)
                m = .05 * np.max(maxxy - minxy) + r / 2
                ax.set_xlim(minxy[0] - m, maxxy[0] + m)
                ax.set_ylim(minxy[1] - m, maxxy[1] + m)
                # ax.set_extent((minxy[1], maxxy[1], maxxy[0], minxy[0]))
        else:
            data = self.rebox_data(data)
            imgs = ax.get_images()
            if imgs:
                im = imgs[0]
                im.set_data(data)
                im.set_cmap(cmap)
                im.set_clim(*cminmax)
                im.set_extent((-.5, self.wh[0]-.5, -.5, self.wh[1]-.5))
                im.autoscale()
            else:
                ax.clear()
                img = ax.imshow(data, cmap=cmap, zorder=0, aspect='equal',
                          vmin=cminmax[0], vmax=cminmax[1], origin='lower')
                if cbfig is not None:
                    cbfig.colorbar(img, ax=ax)

    def draw(self):
        self.draw_ax(self.ax)
        FigureCanvas.draw(self)

    def popOut(self):
        # obj_name = '%s%d' % (self.objectName(), self.display_mode)
        obj_name = None # New figure on every click
        fig = plt.figure(obj_name, tight_layout=dict(pad=.6))
        fig.canvas.set_window_title(self.window_title + ' ' +
                                    self.display_modes[self.display_mode])
        ax = fig.gca()
        self.draw_ax(ax, fig=fig)
        fig.canvas.draw_idle()
        fig.show()

    def draw_ax(self, ax, fig=None):
        mode = self.display_mode
        assert mode < len(self.display_modes)
        plotmodes = 4
        if mode < plotmodes:
            ax.clear()
            ax.set_aspect('auto')
            if mode == 0:
                self.draw_error_log(ax)
            elif mode == 1:
                self.draw_init_spectra(ax)
            elif mode == 2:
                self.draw_spectra(ax)
            elif mode == 3:
                self.draw_concentrations(ax)
        else:
            cnum = mode - plotmodes
            if cnum < len(self.concentrations):
                self.draw_heatmap(ax, self.concentrations[cnum],
                                  discrete=False, cbfig=fig)
            else:
                data = list(self.cluster_maps.values())[
                    cnum - len(self.concentrations)]
                self.draw_heatmap(ax, data, discrete=True, cbfig=fig)


class ClusterPlotWidget(BasePlotWidget):
    clicked = pyqtSignal(int)  # Clicked on cluster N

    def __init__(self, parent=None):
        super().__init__(parent)

        self.window_title = 'Clustering'
        # Data to plot
        self.clusters = None
        self.roi = None
        self.cmap = 'tab10'
        self.mpl_connect('button_press_event', self.on_click)

        # Things common to both display modes
        self.ax.set_aspect('equal')
        self.ax.tick_params(bottom=False, labelbottom=False,
                            left=False, labelleft=False)
        self.ax.set_facecolor('#eeeeee')


    def set_basic_data(self, pixels=None, wh=None, pixelxy=None):
        "Set shape/points and clears clusters"
        super().set_basic_data(pixels=pixels, wh=wh, pixelxy=pixelxy)
        # clusters_of_all: 0 or cluster+1 for all pixels
        self.clusters_of_all = np.zeros((self.pixels), dtype=np.uint8)
        self.set_roi_and_clusters(None, None)

    def adjust_geometry(self, wh):
        "Update the plot geometry, keeping the pixel count"
        assert pixels_fit(self.pixels, wh)
        self.wh = wh
        self.set_roi_and_clusters(self.roi, self.clusters)

    @pyqtSlot(float)
    def set_pixel_size(self, s):
        "Set the size of patches and redraw"
        self.pixel_size = s
        self.draw_idle()

    @pyqtSlot('QString')
    def set_cmap(self, s):
        "Set colors for clustering etc"
        self.cmap = s
        self.draw_idle()

    def clear_clusters(self):
        self.set_roi_and_clusters(None, None)

    def cluster_color(self, cluster):
        return self.cmap_object(self.norm(cluster))

    @pyqtSlot(np.ndarray)
    def set_roi_and_clusters(self, roi, clusters):
        "Set ROI and cluster labels for all pixels in ROI"
        if roi is not None:
            if len(roi) != self.pixels:
                raise ValueError('ROI size must match the current geometry')
        setup_axes = self.clusters is None
        self.roi = roi
        self.clusters = clusters
        if clusters is None:
            self.clusters_of_all.fill(0)
            self.ax.clear()
            self.draw_idle()
            return

        if roi is None:
            self.clusters_of_all[:] = clusters + 1
        else:
            self.clusters_of_all.fill(0)
            self.clusters_of_all[roi] = clusters + 1

        if setup_axes:
            ...

        lc = clusters.max()
        cminmax = (0, lc)
        cmap = plt.get_cmap(self.cmap)
        if hasattr(cmap, 'colors') and cmap.N < 32:
            # Reshape color list to match the input values
            cmap = matplotlib.colors.ListedColormap(cmap.colors, N=lc+1)
        self.cmap_object = cmap
        self.norm = matplotlib.colors.Normalize(
            vmin=cminmax[0], vmax=cminmax[1])

        if self.pixelxy is not None:
            unroi = (.5, .5, .5)
            if setup_axes:
                minxy = np.min(self.pixelxy, axis=0)
                maxxy = np.max(self.pixelxy, axis=0)
                r = self.pixel_size
                for i, xy in enumerate(self.pixelxy):
                    c = self.clusters_of_all[i]
                    fc = self.cluster_color(c - 1) if c else unroi
                    p = matplotlib.patches.Rectangle(
                        xy - r / 2, r, r, facecolor=fc)
                    self.ax.add_patch(p)
                m = .05 * np.max(maxxy - minxy) + r / 2
                self.ax.set_xlim(minxy[0] - m, maxxy[0] + m)
                self.ax.set_ylim(minxy[1] - m, maxxy[1] + m)
            else:
                for i, p in enumerate(self.ax.patches):
                    c = self.clusters_of_all[i]
                    p.set_facecolor(self.cluster_color(c - 1) if c else unroi)
        else:
            i2d = np.zeros((self.pixels, 4))
            i2d[roi, :] = self.cluster_color(clusters)
            i2d = self.rebox_data(i2d)
            if setup_axes:
                self.ax.imshow(i2d, zorder=0, origin='lower')
            else:
                im = self.ax.get_images()[0]
                im.set_data(i2d)
                im.set_extent((-.5, self.wh[0]-.5, -.5, self.wh[1]-.5))
        self.draw_idle()

    def on_click(self, event):
        if self.clusters is None or event.inaxes != self.ax:
            return
        if self.pixelxy is not None:
            for i, p in enumerate(self.ax.patches):
                if self.roi is None or self.roi[i]:
                    if p.contains(event)[0]:
                        self.clicked.emit(self.clusters_of_all[i] - 1)
        else:
            x, y = (int(event.xdata + .5), int(event.ydata + .5))
            if (0 <= x < self.wh[0] and 0 <= y < self.wh[1]):
                i = x + y * self.wh[0]
                if i < len(self.clusters_of_all):
                    c = self.clusters_of_all[i]
                    if c:
                        self.clicked.emit(c - 1)

