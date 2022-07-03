from PyQt5.QtWidgets import QSizePolicy
from PyQt5.QtCore import pyqtSignal, pyqtSlot
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
#import matplotlib.patches as patches
import numpy as np
#import random



class DataPlotWidget(FigureCanvas):
    updated = pyqtSignal()
    clicked = pyqtSignal()

    def __init__(self, parent=None):
        self.fig = Figure(tight_layout=dict(pad=0.4))
        self.ax = self.fig.add_subplot(111)
        FigureCanvas.__init__(self, self.fig)
        self.setParent(parent)
#        self.fig.canvas.mpl_connect('button_press_event', self.onclick)
        FigureCanvas.setSizePolicy(self, QSizePolicy.Expanding,
                                   QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)
        self.wavenumber = None
        self.inspectra = None
        self.spectra = None
        self.popax = None
        self.ltr = True
        self.window_title = self.__class__.__name__

    def mousePressEvent(self, event):
        self.clicked.emit()

    def setData(self, wavenumber, inspectra, spectra):
        if (self.wavenumber is wavenumber and
            self.inspectra is inspectra and
            self.spectra is spectra):
            return
        # Skip redraw if there is and was no outdata
        if self.spectra is not None or spectra is not None:
            self.draw_idle()
        self.wavenumber = wavenumber
        self.inspectra = inspectra
        self.spectra = spectra
        self.updated.emit()

    def getSpectra(self):
        return self.spectra if self.spectra is not None else self.inspectra

    def getWavenumbers(self):
        return self.wavenumber

    def setOrder(self, ltr):
        if ltr != self.ltr:
            if ltr == self.ax.xaxis_inverted():
                self.ax.invert_xaxis()
                FigureCanvas.draw(self)
            if self.popax and ltr == self.popax.xaxis_inverted():
                self.popax.invert_xaxis()
                self.popax.figure.canvas.draw()
            self.ltr = ltr

    def draw_ax(self, ax):
        pass

    def draw_ax_single(self, ax, args={}, spectra=None):
        if spectra is None:
            spectra = self.spectra
        lines = ax.get_lines()
        if len(lines) == len(spectra):
            for i, l in enumerate(lines):
                l.set_data(self.wavenumber, spectra[i])
            ax.relim()
            ax.autoscale_view()
        else:
            ax.clear()
            for i in spectra:
                ax.plot(self.wavenumber, i, linewidth=1, **args)

    def draw_ax_double(self, ax, label1, label2, spectra1=None,
                       spectra2=None, args1={}, args2={}):
        if spectra1 is None:
            spectra1 = self.inspectra
        if spectra2 is None:
            spectra2 = self.spectra
        lines = ax.get_lines()
        nspec = len(self.spectra)
        if len(lines) == nspec * 2:
            for i, l in enumerate(lines):
                l.set_data(
                    self.wavenumber, spectra1[i] if i < nspec
                    else spectra2[i - nspec])
            ax.relim()
            ax.autoscale_view()

            if ax.figure.canvas.toolbar is not None:
                tb = ax.figure.canvas.toolbar
                # print(dir(tb))
                print(dir(tb._nav_stack))
                # tb.update()
                # n = tb._nav_stack.back()
                print(tb._nav_stack._elements)
                # print(dir(tb._nav_stack._elements))
                if tb._nav_stack._elements:
                    wl = tb._nav_stack._elements[0]
                    for q,qq in wl.items():
                        print('q',qq)
                        print('qq',qq[0])
                        wl[q] = ((1000,4000,0,1), qq[1])

        else:
            ax.clear()
            for i, s in enumerate(spectra1):
                ax.plot(self.wavenumber, s, linewidth=1,
                        label=label1 if not i else None, **args1)
            for i, s in enumerate(spectra2):
                ax.plot(self.wavenumber, s, linewidth=1,
                        label=label2 if not i else None, **args2)
            ax.legend()


    def draw_or_clear(self, ax):
        if self.spectra is not None and len(self.spectra) > 0:
            self.draw_ax(ax)
            if self.ltr == ax.xaxis_inverted():
                ax.invert_xaxis()
        else:
            ax.clear()

    def draw(self):
        self.draw_or_clear(self.ax)
        FigureCanvas.draw(self)
        if plt.fignum_exists(self.objectName()):
            self.popOut()

    def popOut(self):
        fig = plt.figure(self.objectName(), tight_layout=dict(pad=.6))
        fig.canvas.set_window_title(self.window_title)
        ax = fig.gca()
        self.draw_or_clear(ax)
        fig.canvas.draw_idle()
        if ax != self.popax:
            fig.show()
            self.popax = ax


class SpectraPlotWidget(DataPlotWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.window_title = 'Raw spectra'

    def draw_ax(self, ax):
        self.draw_ax_single(ax)

class NormPlotWidget(SpectraPlotWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.window_title = 'Normalized spectra'

class ACPlotWidget(DataPlotWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.window_title = 'Atmospheric correction'

    def draw_ax(self, ax):
        self.draw_ax_double(ax, 'Raw', 'Atm. corr.',
                            args2=dict(color='r', linestyle='--'))


class SCPlotWidget(DataPlotWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.window_title = 'Scattering correction'
        self.progressMode = False
        self.runSelected = None
        self.showSelected = None

    def setData(self, wavenumber, inspectra, spectra, runSelected, showSelected):
        self.wavenumber = wavenumber
        self.inspectra = inspectra
        self.spectra = spectra
        self.runSelected = runSelected
        self.setSelected(showSelected)

    def setSelected(self, sel):
        if self.runSelected is not None:
            if sorted(sel) == sorted(self.runSelected):
                self.showSelected = range(len(self.spectra))
            else:
                self.showSelected = []
        else:
            self.showSelected = sel
        self.draw_idle()
        self.updated.emit()

    def getSpectra(self):
        if self.spectra is None or self.runSelected is not None:
            return self.spectra
        return self.spectra[self.showSelected]

    # Rather ugly hack to show progress in the main window only: the draw method is
    # made to bypass the DataPlotWidget draw.
    def prepareProgress(self, wn):
        self.wavenumber = wn
        self.ax.clear()
        self.progressMode = True
        self.draw_idle()

    def endProgress(self):
        if self.progressMode:
            self.progressMode = False
            self.draw_idle()

    @pyqtSlot(np.ndarray, tuple)
    def progressPlot(self, curves, progress):
        if self.progressMode:
            self.ax.plot(self.wavenumber, curves.T,
                         color=plt.cm.jet(progress[0] / progress[1]), linewidth=1)
            self.draw_idle()

    def draw(self):
        if self.progressMode:
            return FigureCanvas.draw(self)
        return DataPlotWidget.draw(self)

    def draw_ax(self, ax):
        ax.clear()
        i = 0
        for s in self.showSelected:
            ax.plot(self.wavenumber, self.inspectra[s],
                    color='black', linewidth=1, label='Input' if not i else None)
            ax.plot(self.wavenumber, self.spectra[s],
                    linewidth=1, label='Corrected' if not i else None)
            i += 1
        if i:
            ax.legend()


class SGFPlotWidget(DataPlotWidget):
    def __init__(self, parent=None):
        DataPlotWidget.__init__(self, parent=None)
        self.window_title = 'Savitzky-Golay filter'

    def draw_ax(self, ax):
        lines = ax.get_lines()
        nspec = len(self.spectra)
        if len(lines) == nspec * 2:
            for i, l in enumerate(lines):
                l.set_data(
                    self.wavenumber, self.inspectra[i] if i < nspec
                    else self.spectra[i - nspec])
            ax.relim()
            ax.autoscale_view()
        else:
            ax.clear()
            for i, s in enumerate(self.inspectra):
                ax.plot(self.wavenumber, s, linewidth=1,
                        color='b', label='Input' if not i else None)
            for i, s in enumerate(self.spectra):
                ax.plot(self.wavenumber, s, linewidth=1,
                        label='Smoothed' if not i else None)
            ax.legend()


class SRPlotWidget(DataPlotWidget):
    def __init__(self, parent=None):
        DataPlotWidget.__init__(self, parent=None)
        self.window_title = 'Spectral region selection'
        self.mincut=0
        self.maxcut=10000
        self.cutwn=np.zeros(0)
        self.vspans = [None, None]

    def setData(self, wavenumber, inspectra, spectra, cutwn, mincut, maxcut):
        self.wavenumber = wavenumber
        self.inspectra = inspectra
        self.spectra = spectra
        self.mincut=mincut
        self.maxcut=maxcut
        self.cutwn=cutwn
        self.draw_idle()
        self.updated.emit()

    def getWavenumbers(self):
        return self.cutwn if self.cutwn is not None else self.wavenumber

    def draw_ax(self, ax):
        self.draw_ax_single(ax, spectra=self.inspectra)

        for v in self.vspans:
            if v is not None:
                v.remove()
        self.vspans = [None, None]
        mmwn = self.wavenumber[[0, -1]]
        mmwn = (min(mmwn), max(mmwn))
        if mmwn[0] < self.mincut:
            self.vspans[0] = ax.axvspan(mmwn[0], self.mincut, color='gray')
        if self.maxcut < mmwn[1]:
            self.vspans[1] = ax.axvspan(self.maxcut, mmwn[1], color='gray')


class BCPlotWidget(DataPlotWidget):
    def __init__(self, parent=None):
        DataPlotWidget.__init__(self, parent=None)
        self.window_title = 'Baseline correction'

    def getSpectra(self):
        if self.spectra is not None:
            return self.inspectra - self.spectra
        return self.inspectra

    def draw_ax(self, ax):
        self.draw_ax_double(ax, 'Input', 'Baseline', args2=dict(color='k'))


class MCPlotWidget(DataPlotWidget):
    def __init__(self, parent=None):
        DataPlotWidget.__init__(self, parent=None)
        self.window_title = 'mIRage correction'

    def draw_ax(self, ax):
        ss1 = self.inspectra / self.inspectra.max(1)[:, None]
        ss2 = self.spectra / self.spectra.max(1)[:, None]
        self.draw_ax_double(
            ax, spectra1=ss1, label1='Input', args1=dict(color='b'),
            spectra2=ss2, label2='Corrected', args2=dict(color='r'))

class MCInfoPlotWidget(FigureCanvas):
    def __init__(self, parent=None):
        self.fig = Figure(tight_layout=dict(pad=0.4))
        self.ax = self.fig.add_subplot(111)
        FigureCanvas.__init__(self, self.fig)
        self.setParent(parent)
        FigureCanvas.setSizePolicy(self, QSizePolicy.Expanding,
                                   QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)
        self.logshifts = None # (nbreaks, npixels)
        self.breaks = None # (nbreaks, 2)
        # self.mode = 0

    # def mousePressEvent(self, event):
    #     self.mode = (self.mode + 1) % 2
    #     self.draw_idle()

    def setData(self, breaks, mcinfo):
        self.breaks = breaks
        if mcinfo is None:
            self.logshifts = None
            self.weights = None
        else:
            self.logshifts, self.weights = mcinfo
        self.draw_idle()

    def draw(self):
        ax = self.ax
        if self.logshifts is None:
            ax.clear()
        else:
            # if len(ax.collections) == len(self.logshifts):
            #     for i, l in enumerate(self.logshifts):
            #         ax.collections[i].set_offsets(np.column_stack(
            #             (self.weights[i]/self.weights[i].mean(), l)))
            #     ax.relim()
            #     ax.autoscale_view()
            # else:
            ax.clear()
            for i, l in enumerate(self.logshifts):
                ax.scatter(self.weights[i]/self.weights[i].mean(), l,
                           marker='x',
                           label='%g-%g' % tuple(self.breaks[i]))
                ax.set_ylabel(r'log$_{e}$ shift')
                ax.set_xlabel('Soft limit point reliability')
                ax.legend()
        # else:
            # lines = ax.get_lines()
            # logshifts = np.sort(self.logshifts, 1)
            # if len(lines) == len(logshifts):
            #     for i, l in enumerate(logshifts):
            #         lines[i].set_data(range(len(l)), l)
            #     ax.relim()
            #     ax.autoscale_view()
            # else:
            #     ax.clear()
            #     ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            #     ax.set_title(r'log$_{e}$ shift')
            #     for i, l in enumerate(logshifts):
            #         ax.plot(l, linewidth=1,
            #                      label='(%g - %g)' % tuple(self.breaks[i]))
            #     ax.legend()
        FigureCanvas.draw(self)
