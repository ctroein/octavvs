import os
import os.path
#import fnmatch
#import traceback
# from pkg_resources import resource_filename

#import numpy as np

# from PyQt5.QtWidgets import QWidget
#from PyQt5.QtCore import QThread, pyqtSignal, pyqtSlot, Qt
# from PyQt5 import uic
# import matplotlib.pyplot as plt

from . import constants, uitools
from octavvs.ui import NoRepeatStyle
# from ..io import Image


class ImageVisualizer():
    """
    Mixin class that adds hyperspectral image display to an OctavvsMainWindow.

    Things assumed to exist in self:
        imageProjection
        plot_raw
        data

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.ptirMode = False
        self.spinBoxSpectra.setStyle(NoRepeatStyle())

        self.pushButtonWhitelight.clicked.connect(self.loadWhite)
        self.pushButtonExpandProjection.clicked.connect(self.plot_raw.popOut)
        self.horizontalSliderImage.valueChanged.connect(self.selectWhiteLight)
        self.lineEditWavenumber.setFormat("%.2f")
        self.horizontalSliderWavenumber.valueChanged.connect(self.wavenumberSlide)
        self.lineEditWavenumber.editingFinished.connect(self.wavenumberEdit)
        self.comboBoxCmaps.currentTextChanged.connect(self.plot_raw.setCmap)

        self.comboBoxMethod.currentIndexChanged.connect(self.updateSpectralImage)
        self.checkBoxPixelFill.toggled.connect(self.plot_raw.setFill)
        self.whiteLightNames = {}

        self.plot_raw.changedSelected.connect(self.selectedSpectraUpdated)
        self.plot_spectra.clicked.connect(self.plot_spectra.popOut)
        self.spinBoxSpectra.valueChanged.connect(self.selectSpectra)
        self.checkBoxAutopick.toggled.connect(self.selectSpectra)
        self.comboBoxPlotCmap.currentTextChanged.connect(self.setPlotColors)


    def getParameters(self, p):
        "Copy from UI to some kind of parameters object"
        p.plotMethod = self.comboBoxMethod.currentIndex()
        p.plotColors = self.comboBoxCmaps.currentText()
        p.plotWavenum = self.lineEditWavenumber.value()
        p.spectraCount = self.spinBoxSpectra.value()
        p.spectraAuto = self.checkBoxAutopick.isChecked()

    def setParameters(self, p):
        "Copy from some kind of parameters object to UI"
        self.comboBoxMethod.setCurrentIndex(p.plotMethod)
        self.comboBoxCmaps.setCurrentText(p.plotColors)
        self.lineEditWavenumber.setValue(p.plotWavenum)
        self.checkBoxAutopick.setChecked(p.spectraAuto)
        self.spinBoxSpectra.setValue(p.spectraCount)

    def updatedFile(self):
        self.ptirMode = self.data.filetype == 'ptir'
        # self.checkBoxPixelFill.setHidden(not self.ptirMode)
        # self.stackedVisualizer.setCurrentIndex(int(self.ptirMode))

        self.plot_raw.setData(
            self.data.wavenumber, self.data.raw,
            self.data.pixelxy if self.ptirMode else None)
        self.plot_raw.setDimensions(self.data.wh)

        self.updateSpectralImage()
        self.updateWhiteLightImages()
        self.selectSpectra()

    def updateWavenumberRange(self):
        super().updateWavenumberRange()
        # Update sliders before boxes to avoid errors from triggered updates
        if self.data.wavenumber is not None:
            self.horizontalSliderWavenumber.setMaximum(len(self.data.wavenumber)-1)

        self.labelMinwn.setText("%.2f" % self.data.wmin)
        self.labelMaxwn.setText("%.2f" % self.data.wmax)
        wmin = min(self.data.wmin, constants.WMIN)
        wmax = max(self.data.wmax, constants.WMAX)
        self.lineEditWavenumber.setRange(wmin, wmax, default=.5*(wmin+wmax))

        # Make sure the sliders are in sync with the boxes
        self.wavenumberEdit()

    def wavenumberEdit(self):
        uitools.box_to_slider(
                self.horizontalSliderWavenumber, self.lineEditWavenumber,
                self.plot_raw.getWavenumbers(),
                uitools.ixfinder_nearest)

    def wavenumberSlide(self):
        uitools.slider_to_box(
                self.horizontalSliderWavenumber, self.lineEditWavenumber,
                self.plot_raw.getWavenumbers(),
                uitools.ixfinder_nearest)
        self.updateSpectralImage()

    def updateDimensions(self, wh):
        try:
            self.plot_raw.setDimensions(wh)
        except ValueError:
            pass
        self.updateSpectralImage()


    def selectedSpectraUpdated(self, n):
        self.spinBoxSpectra.setValue(n)
        self.plot_spectra.setData(self.plot_raw.getWavenumbers(), None,
                                  self.plot_raw.getSelectedData())

    def selectSpectra(self):
        self.plot_raw.setSelectedCount(
                self.spinBoxSpectra.value(),
                self.checkBoxAutopick.isChecked())

    def setPlotColors(self, cmap):
        self.plot_raw.updatePlotColors(cmap)
        # Trigger redraw
        self.plot_spectra.draw_idle()



    def updateSpectralImage(self):
        meth = self.comboBoxMethod.currentIndex()
        iswn = meth == 2
        self.lineEditWavenumber.setEnabled(iswn)
        self.horizontalSliderWavenumber.setEnabled(iswn)
        wn = self.data.wavenumber
        wix = len(wn)-1 - self.horizontalSliderWavenumber.value() if (
            wn[0] > wn[-1]) else self.horizontalSliderWavenumber.value()
        # print('wavenumix', wix, wn[wix])
        self.plot_raw.setProjection(meth, wix)

    def updateWhiteLightImages(self):
        if self.ptirMode:
            if self.data.images and len(self.data.images):
                self.horizontalSliderImage.setMaximum(len(self.data.images)-1)
                self.horizontalSliderImage.setEnabled(True)
                self.selectWhiteLight(self.horizontalSliderImage.value())
            else:
                self.horizontalSliderImage.setEnabled(False)
                self.horizontalSliderImage.setValue(0)
                self.lineEditImageInfo.setText('')
        else:
            fname = self.whiteLightNames[self.data.curFile] if \
                self.data.curFile in self.whiteLightNames else \
                    os.path.splitext(self.data.curFile)[0]+'.jpg'
            # self.plot_whitelight.load(filename=fname)

    # White light image stuff
    def loadWhite(self):
        filename = self.getImageFileName(title="Load white light image",
                                         settingname='whitelightDir')
        if not filename:
            return
        self.plot_whitelight.load(filename=filename)
        self.whiteLightNames[self.data.curFile] = filename

    def selectWhiteLight(self, num):
        if self.data.images and len(self.data.images):
            self.plot_raw.setImage(self.data.images[num])
            self.lineEditImageInfo.setText(self.data.images[num].name)
        # self.plot_whitelight.draw_rectangles(self.data.pixelxy)



