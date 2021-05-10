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
# from ..io import Image


class ImageVisualizer():
    """
    Mixin class that adds hyperspectral image display to an OctavvsMainWindow.

    Things assumed to exist in self:
        imageProjection
        plot_visual
        data

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.pushButtonWhitelight.clicked.connect(self.loadWhite)
        self.pushButtonExpandProjection.clicked.connect(self.plot_visual.popOut)
        self.horizontalSliderImage.valueChanged.connect(self.selectWhiteLight)
        self.lineEditWavenumber.setFormat("%.2f")
        self.horizontalSliderWavenumber.valueChanged.connect(self.wavenumberSlide)
        self.lineEditWavenumber.editingFinished.connect(self.wavenumberEdit)
        self.comboBoxCmaps.currentTextChanged.connect(self.plot_visual.setCmap)

        self.comboBoxMethod.currentIndexChanged.connect(self.updateSpectralImage)
        self.comboBoxCmaps.currentTextChanged.connect(self.plot_spatial.setCmap)
        self.checkBoxPixelFill.toggled.connect(self.plot_spatial.setFill)
        self.whiteLightNames = {}

    def getParameters(self, p):
        "Copy from UI to some kind of parameters object"
        p.plotMethod = self.comboBoxMethod.currentIndex()
        p.plotColors = self.comboBoxCmaps.currentText()
        p.plotWavenum = self.lineEditWavenumber.value()

    def setParameters(self, p):
        "Copy from some kind of parameters object to UI"
        self.comboBoxMethod.setCurrentIndex(p.plotMethod)
        self.comboBoxCmaps.setCurrentText(p.plotColors)
        self.lineEditWavenumber.setValue(p.plotWavenum)

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
                self.plot_visual.getWavenumbers(),
                uitools.ixfinder_nearest)

    def wavenumberSlide(self):
        uitools.slider_to_box(
                self.horizontalSliderWavenumber, self.lineEditWavenumber,
                self.plot_visual.getWavenumbers(),
                uitools.ixfinder_nearest)
        self.updateSpectralImage()

    def updateDimensions(self, wh):
        try:
            self.plot_visual.setDimensions(wh)
        except ValueError:
            pass
        self.updateSpectralImage()

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
                self.plot_spatial.load_white(None)
                self.lineEditImageInfo.setText('')
        else:
            fname = self.whiteLightNames[self.data.curFile] if \
                self.data.curFile in self.whiteLightNames else \
                    os.path.splitext(self.data.curFile)[0]+'.jpg'
            self.plot_whitelight.load(filename=fname)

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
            self.plot_spatial.setImage(self.data.images[num])
            self.lineEditImageInfo.setText(self.data.images[num].name)
        # self.plot_whitelight.draw_rectangles(self.data.pixelxy)



