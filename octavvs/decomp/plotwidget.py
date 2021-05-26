from PyQt5.QtWidgets import QSizePolicy
from PyQt5.QtCore import pyqtSignal, pyqtSlot
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np



class DecompositionPlotWidget(FigureCanvas):
    clicked = pyqtSignal()
    displayModesUpdated = pyqtSignal(list)

    def __init__(self, parent=None):
        self.fig = Figure(tight_layout=dict(pad=0.4))
        self.ax = self.fig.add_subplot(111)
        FigureCanvas.__init__(self, self.fig)
        self.setParent(parent)
        FigureCanvas.setSizePolicy(
            self, QSizePolicy.Expanding, QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)
        # self.popax = None

        self.window_title = 'Decomposition'
        # Data to plot
        self.concentrations = []
        self.wn = None
        self.spectra = []
        self.init_spectra = []
        self.error_log = []
        self.cluster_maps = {}
        # 2D layout
        self.wh = None
        self.pixelxy = None
        self.pixels = 0
        # What/how to plot
        self.cmap = plt.get_cmap().name
        self.display_mode = 0 # Index for now - should perhaps be string
        self.update_display_modes()

    def mousePressEvent(self, event):
        self.clicked.emit()

    def draw(self):
        self.draw_ax(self.ax)
        FigureCanvas.draw(self)
        if plt.fignum_exists(self.objectName()):
            self.popOut()

    def popOut(self):
        # obj_name = '%s%d' % (self.objectName(), self.display_mode)
        obj_name = None # New figure on every click
        fig = plt.figure(obj_name, tight_layout=dict(pad=.6))
        fig.canvas.set_window_title(self.window_title + ' ' +
                                    self.display_modes[self.display_mode])
        ax = fig.gca()
        self.draw_ax(ax)
        fig.canvas.draw_idle()
        fig.show()
        # ax = fig.gca()
        # ax.clear()
        # self.draw_ax(ax)
        # fig.canvas.draw_idle()
        # if ax != self.popax:
        #     fig.show()
        #     self.popax = ax


    def update_display_modes(self):
        "Create/update the list of display modes and signal the change"
        m = [ 'Error progress', 'Initial spectra', 'Spectra',
             'Concentrations' ]
        for i in range(len(self.concentrations)):
            m.append('Concentration %d' % (i+1))
        m = m + list(self.cluster_maps.keys())
        self.display_modes = m
        if self.display_mode <= len(m):
            self.display_mode = len(m) - 1
        self.displayModesUpdated.emit(self.display_modes)

    # def get_display_modes(self):
    #     "Get display modes as list of descriptions"
    #     return self.display_modes

    # def get_display_mode(self):
    #     "Get current display mode"
    #     return self.display_mode

    @pyqtSlot(int)
    def set_display_mode(self, m):
        assert m < len(self.display_modes)
        self.display_mode = m
        self.draw_idle()

    def set_geometry(self, wh=None, pixelxy=None):
        if wh is None and pixelxy is None:
            raise ValueError('Either wh or pixelxy must be specified')
        # Clear if incompatible with new geometry
        if wh is not None:
            pixels = wh[0] * wh[1]
        else:
            pixels = len(pixelxy)
        if pixels != self.pixels:
            self.concentrations = []
            self.cluster_maps = {}
            self.update_display_modes()
        self.wh = wh
        self.pixelxy = pixelxy
        self.pixels = pixels

    def set_wavenumbers(self, wn):
        self.wn = wn
        self.spectra = []
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
        assert concentrations.ndim == 2
        assert concentrations.shape[1] == self.pixels
        udm = len(concentrations) != len(self.concentrations)
        self.concentrations = concentrations
        self.add_clustering('Strongest component',
                            np.argmax(self.concentrations, axis=0))
        if udm:
            self.update_display_modes()
        self.draw_idle()

    def get_concentrations(self):
        return self.concentrations

    def set_spectra(self, spectra):
        self.spectra = spectra
        self.draw_idle()

    def clear_errors(self):
        self.error_log = []

    def add_error(self, e):
        self.error_log.append(e)

    @pyqtSlot(int, np.ndarray, np.ndarray)
    def plot_progress(self, iteration, concentrations, spectra):
        if iteration:
            self.set_concentrations(concentrations)
            self.set_spectra(spectra)
        else:
            self.init_spectra = spectra

    def add_clustering(self, name, clustered):
        assert len(clustered) == self.pixels
        self.cluster_maps[name] = clustered
        if name not in self.display_modes:
            self.update_display_modes()

    @pyqtSlot('QString')
    def set_cmap(self, s):
        self.cmap = s
        for im in self.ax.get_images():
            im.set_cmap(s)
        # if self.popax is not None:
        #     for im in self.popax.get_images():
        #         im.set_cmap(s)
        self.draw_idle()

    def draw_ax(self, ax):
        mode = self.display_mode
        assert mode < len(self.display_modes)
        plotmodes = 4
        if mode < plotmodes:
            ax.clear()
            ax.set_aspect('auto')
            ax.set_yscale('linear')
            if mode == 0 and len(self.error_log):
                ax.set_yscale('log')
                ax.plot(self.error_log, label='Error')
                ax.legend()
            elif mode == 1 and len(self.init_spectra):
                for i in range(len(self.init_spectra)):
                    ax.plot(self.wn, self.init_spectra[i,:],
                            label='Initial component %d'%(i+1))
                ax.legend()
            elif mode == 2 and len(self.spectra):
                for i in range(len(self.spectra)):
                    s = self.spectra[i]
                    ax.plot(self.wn, s / s.mean(),
                            label='Component %d'%(i+1))
                ax.legend()
            else:
                for i in self.concentrations:
                    ax.plot(i / i.mean())
        else:
            cnum = mode - plotmodes
            if cnum < len(self.concentrations):
                data = self.concentrations[cnum]
                cmap = self.cmap
                cminmax = data.min(), data.max()
            else:
                # temp ugly hack
                data = list(self.cluster_maps.values())[
                    cnum - len(self.concentrations)]
                colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
                lc = data.max() + 1
                if len(colors) < lc:
                    colors = colors * (lc // len(colors) + 1)
                cminmax = (-.5, lc - .5)
                cmap = ListedColormap(colors[:lc])
            if self.wh is not None:
                data = data.reshape(self.wh)
                imgs = ax.get_images()
                if imgs:
                    im = imgs[0]
                    im.set_data(data)
                    im.set_cmap(cmap)
                    im.set_clim(*cminmax)
                    im.set_extent((0, self.wh[1], self.wh[0], 0))
                    im.autoscale()
                else:
                    ax.clear()
                    ax.imshow(data, cmap=cmap, zorder=0, aspect='equal',
                              vmin=cminmax[0], vmax=cminmax[1])
            elif self.pixelxy is not None:
                raise RuntimeWarning('not yet implemented')


