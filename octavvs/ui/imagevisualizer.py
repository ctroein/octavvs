import os
import os.path
#import fnmatch
#import traceback
from pkg_resources import resource_filename

#import numpy as np

from PyQt5.QtWidgets import QWidget
#from PyQt5.QtCore import QThread, pyqtSignal, pyqtSlot, Qt
from PyQt5 import uic

from . import constants, uitools
# from ..io import Image

ImageVisualizerUi = uic.loadUiType(resource_filename(__name__, "imagevisualizer.ui"))[0]

class ImageVisualizerWidget(QWidget, ImageVisualizerUi):
    """
    Widget with a bunch of graphical components related to ...
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setupUi(self)
        self.lineEditWavenumber.setFormat("%.2f")

    def saveParameters(self, p):
        "Copy from UI to some kind of parameters object"
        p.plotMethod = self.comboBoxMethod.currentIndex()
        p.plotColors = self.comboBoxCmaps.currentText()
        p.plotWavenum = self.lineEditWavenumber.value()

    def loadParameters(self, p):
        "Copy to UI from some kind of parameters object"
        self.comboBoxMethod.setCurrentIndex(p.plotMethod)
        self.comboBoxCmaps.setCurrentText(p.plotColors)
        self.lineEditWavenumber.setValue(p.plotWavenum)


class ImageVisualizer():
    """
    Mixin class that adds hyperspectral image display to an OctavvsMainWindow.

    Things assumed to exist in self:
        imageVisualizer - ImageVisualizer (from UI)
        imageProjection
        plot_visual
        data

    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.pushButtonWhitelight.clicked.connect(self.loadWhite)
        self.horizontalSliderImage.valueChanged.connect(self.reloadWhiteLight)

        iv = self.imageVisualizer
        iv.comboBoxMethod.currentIndexChanged.connect(self.imageProjection)
        iv.horizontalSliderWavenumber.valueChanged.connect(self.wavenumberSlide)
        iv.lineEditWavenumber.editingFinished.connect(self.wavenumberEdit)
        iv.comboBoxCmaps.currentTextChanged.connect(self.plot_visual.setCmap)


    def updateWavenumberRange(self):
        super().updateWavenumberRange()
        # Update sliders before boxes to avoid errors from triggered updates
        iv = self.imageVisualizer
        if self.data.wavenumber is not None:
            iv.horizontalSliderWavenumber.setMaximum(len(self.data.wavenumber)-1)

        iv.labelMinwn.setText("%.2f" % self.data.wmin)
        iv.labelMaxwn.setText("%.2f" % self.data.wmax)
        wmin = min(self.data.wmin, constants.WMIN)
        wmax = max(self.data.wmax, constants.WMAX)
        iv.lineEditWavenumber.setRange(wmin, wmax, default=.5*(wmin+wmax))

        # Make sure the sliders are in sync with the boxes
        self.wavenumberEdit()

    def wavenumberEdit(self):
        iv = self.imageVisualizer
        uitools.box_to_slider(
                iv.horizontalSliderWavenumber, iv.lineEditWavenumber,
                self.plot_visual.getWavenumbers(),
                uitools.ixfinder_nearest)

    def wavenumberSlide(self):
        iv = self.imageVisualizer
        uitools.slider_to_box(
                iv.horizontalSliderWavenumber, iv.lineEditWavenumber,
                self.plot_visual.getWavenumbers(),
                uitools.ixfinder_nearest)
        self.imageProjection()

    def updateDimensions(self, wh):
        try:
            self.plot_visual.setDimensions(wh)
        except ValueError:
            pass
        self.imageProjection()

    def imageProjection(self):
        iv = self.imageVisualizer
        meth = iv.comboBoxMethod.currentIndex()
        iswn = meth == 2
        iv.lineEditWavenumber.setEnabled(iswn)
        iv.horizontalSliderWavenumber.setEnabled(iswn)
        self.plot_visual.setProjection(meth, -1 - iv.horizontalSliderWavenumber.value())

        if self.data.images and len(self.data.images) > 1:
            self.horizontalSliderImage.setMaximum(len(self.data.images)-1)
            self.horizontalSliderImage.show()
            self.lineEditImageInfo.show()
        else:
            self.horizontalSliderImage.hide()
            self.lineEditImageInfo.hide()
            self.horizontalSliderImage.setValue(0)
        self.reloadWhiteLight(self.horizontalSliderImage.value())

    # White light image stuff
    def loadWhite(self):
        filename = self.getImageFileName(title="Load white light image",
                                         settingname='whitelightDir')
        if filename:
            self.plot_whitelight.load(filename)

    def reloadWhiteLight(self, num=0):
        if self.data.images:
            self.plot_whitelight.load(image=self.data.images[num])
            self.lineEditImageInfo.setText(self.data.images[num].name)
            self.plot_whitelight.draw_rectangles(self.data.pixelxy)
        else:
            self.plot_whitelight.load(filename=os.path.splitext(self.data.curFile)[0]+'.jpg')
            self.lineEditImageInfo.setText('')



