from PyQt5.QtWidgets import QSizePolicy
from PyQt5.QtCore import pyqtSignal, pyqtSlot
from matplotlib.figure import Figure
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
        FigureCanvas.setSizePolicy(self, QSizePolicy.Expanding, QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)
        self.wavenumber = None
        self.inspectra = None
        self.spectra = None
        self.popax = None

    def mousePressEvent(self, event):
        self.clicked.emit()

    def setData(self, wavenumber, inspectra, spectra):
        if (self.wavenumber is wavenumber and
            self.inspectra is inspectra and
            self.spectra is spectra):
            return
        self.wavenumber = wavenumber
        self.inspectra = inspectra
        self.spectra = spectra
#        self.draw()
        self.draw_idle()
        self.updated.emit()

    def getSpectra(self):
        return self.spectra if self.spectra is not None else self.inspectra

    def getWavenumbers(self):
        return self.wavenumber

    def draw_ax(self, ax):
        pass

    def draw(self):
        self.ax.clear()
        if self.spectra is not None and len(self.spectra) > 0:
            self.draw_ax(self.ax)
        FigureCanvas.draw(self)
        if plt.fignum_exists(self.objectName()):
            self.popOut()

    def popOut(self):
        fig = plt.figure(self.objectName(), tight_layout=dict(pad=.6))
        ax = fig.gca()
        ax.clear()
        if self.spectra is not None and len(self.spectra) > 0:
            self.draw_ax(ax)
        fig.canvas.draw_idle()
        if ax != self.popax:
            fig.show()
            self.popax = ax


class RawPlotWidget(DataPlotWidget):
    def __init__(self, parent=None):
        DataPlotWidget.__init__(self, parent=None)

    def draw_ax(self, ax):
        for i in self.spectra:
            ax.plot(self.wavenumber, i, linewidth=1)

class NormPlotWidget(RawPlotWidget):
    pass

class ACPlotWidget(DataPlotWidget):
    def __init__(self, parent=None):
        DataPlotWidget.__init__(self, parent=None)

    def draw_ax(self, ax):
        for s in range(len(self.spectra)):
            ax.plot(self.wavenumber, self.inspectra[s], linewidth=1, label='Raw' if not s else None)
            ax.plot(self.wavenumber, self.spectra[s],
                    color='r', linewidth=1, linestyle='--', label='Atm corr.' if not s else None)
        ax.legend()


class SCPlotWidget(DataPlotWidget):
    def __init__(self, parent=None):
        DataPlotWidget.__init__(self, parent=None)
        self.progressMode = False
        self.runSelected = None
        self.showSelected = None

#    def getInputSpectra(self):
#        return self.inspectra

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

    def draw_ax(self, ax):
        for s in range(len(self.spectra)):
            ax.plot(self.wavenumber, self.inspectra[s],
                    color='black', linewidth=1, label='Input' if not s else None)
            ax.plot(self.wavenumber, self.spectra[s],
                    linewidth=1, label='Smoothed' if not s else None)
        if len(self.spectra):
            ax.legend()


class SRPlotWidget(DataPlotWidget):
    def __init__(self, parent=None):
        DataPlotWidget.__init__(self, parent=None)
        self.mincut=0
        self.maxcut=10000
        self.cutwn=np.zeros(0)

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
        if self.spectra is None:
            print('SR draw_ax called without spectra!')
        for s in self.inspectra:
            ax.plot(self.wavenumber, s, linewidth=1)
        if self.wavenumber[-1] < self.mincut:
            ax.axvspan(self.wavenumber[-1], self.mincut, color='gray')
        if self.maxcut < self.wavenumber[0]:
            ax.axvspan(self.maxcut, self.wavenumber[0], color='gray')


class BCPlotWidget(DataPlotWidget):
    def __init__(self, parent=None):
        DataPlotWidget.__init__(self, parent=None)

    def getSpectra(self):
        if self.spectra is not None:
            return self.inspectra - self.spectra
        return self.inspectra

    def draw_ax(self, ax):
        for s in range(len(self.spectra)):
            ax.plot(self.wavenumber, self.inspectra[s],
                    linewidth=1, label='Input' if not s else None)
            ax.plot(self.wavenumber, self.spectra[s],
                    color='k', linewidth=1, label='Baseline' if not s else None)
#            ax.plot(self.wavenumber, self.inspectra[s] - self.spectra[s],
#                    color='gray', linewidth=1, label='Corrected' if not s else None)
            ax.legend()

