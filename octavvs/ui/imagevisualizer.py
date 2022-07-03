import os
import os.path
#import fnmatch
#import traceback
from pkg_resources import resource_filename

#from PyQt5.QtCore import QThread, pyqtSignal, pyqtSlot, Qt
from PyQt5 import uic
from PyQt5.QtWidgets import QWidget
from PyQt5.QtCore import pyqtSignal
from . import constants, uitools
from octavvs.ui import NoRepeatStyle
# from ..io import Image

ImageVisualizerUi = uic.loadUiType(
    resource_filename(__name__, "imagevisualizer.ui"))[0]

class ImageVisualizerWidget(QWidget, ImageVisualizerUi):
    """
    Widget with a bunch of graphical components
    """
    updatePixelSize = pyqtSignal(float)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setupUi(self)
        self.spinBoxSpectra.setStyle(NoRepeatStyle())
        self.lineEditWavenumber.setFormat("%.2f")

        self.pushButtonExpandProjection.clicked.connect(self.plot_raw.popOut)
        self.comboBoxCmaps.currentTextChanged.connect(self.plot_raw.setCmap)
        self.comboBoxMethod.currentIndexChanged.connect(
            self.updateSpectralImage)
        self.checkBoxPixelFill.toggled.connect(self.plot_raw.setFill)
        self.lineEditPixelSize.setRange(1e-2, 1e3)
        self.lineEditPixelSize.setFormat('%g')
        self.lineEditPixelSize.editingFinished.connect(self.pixelSizeEdit)
        self.horizontalSliderWavenumber.valueChanged.connect(
            self.wavenumberSlide)
        self.lineEditWavenumber.editingFinished.connect(self.wavenumberEdit)
        self.comboBoxPlotCmap.currentTextChanged.connect(
            self.plot_raw.updatePlotColors)

        self.updatePixelSize.connect(self.plot_raw.setRectSize)
        self.plot_raw.changedSelected.connect(self.selectedSpectraUpdated)
        self.plot_spectra.clicked.connect(self.plot_spectra.popOut)
        self.spinBoxSpectra.valueChanged.connect(self.selectSpectra)
        self.checkBoxAutopick.toggled.connect(self.selectSpectra)

    def pixelSizeEdit(self):
        self.updatePixelSize.emit(self.lineEditPixelSize.value())

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

    def updateSpectralImage(self):
        meth = self.comboBoxMethod.currentIndex()
        iswn = meth == 2
        self.lineEditWavenumber.setEnabled(iswn)
        self.horizontalSliderWavenumber.setEnabled(iswn)
        wn = self.plot_raw.getWavenumbers()
        if wn is not None:
            wix = len(wn)-1 - self.horizontalSliderWavenumber.value() if (
                wn[0] > wn[-1]) else self.horizontalSliderWavenumber.value()
            self.plot_raw.setProjection(meth, wix)

    def selectedSpectraUpdated(self, n):
        self.spinBoxSpectra.setValue(n)
        self.plot_spectra.setData(self.plot_raw.getWavenumbers(), None,
                                  self.plot_raw.getSelectedData())

    def selectSpectra(self):
        self.plot_raw.setSelectedCount(
                self.spinBoxSpectra.value(),
                self.checkBoxAutopick.isChecked())

    def setImageCount(self, n):
        if n:
            self.horizontalSliderImage.setMaximum(n-1)
        else:
            self.horizontalSliderImage.setValue(0)
            self.lineEditImageInfo.setText('')
        self.horizontalSliderImage.setEnabled(n > 0)
        self.lineEditImageInfo.setEnabled(n > 0)
        self.horizontalSliderImage.setHidden(n < 2)

    def saveParameters(self, p):
        "Copy from UI to some kind of parameters object"
        p.plotMethod = self.comboBoxMethod.currentIndex()
        p.plotColors = self.comboBoxCmaps.currentText()
        p.plotWavenum = self.lineEditWavenumber.value()
        p.spectraCount = self.spinBoxSpectra.value()
        p.spectraAuto = self.checkBoxAutopick.isChecked()

    def loadParameters(self, p):
        "Copy from some kind of parameters object to UI"
        self.comboBoxMethod.setCurrentIndex(p.plotMethod)
        self.comboBoxCmaps.setCurrentText(p.plotColors)
        self.lineEditWavenumber.setValue(p.plotWavenum)
        self.checkBoxAutopick.setChecked(p.spectraAuto)
        self.spinBoxSpectra.setValue(p.spectraCount)
        self.wavenumberEdit()

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

        self.spatialMode = False

        self.imageVisualizer.pushButtonWhitelight.clicked.connect(
            self.loadWhite)
        self.imageVisualizer.horizontalSliderImage.valueChanged.connect(
            self.selectImage)
        self.whiteLightNames = {}

    def updateFile(self, num):
        super().updateFile(num)

        self.spatialMode = self.data.pixelxy is not None

        self.imageVisualizer.plot_raw.setData(
            self.data.wavenumber, self.data.raw,
            self.data.pixelxy)
        self.imageVisualizer.plot_raw.setDimensions(self.data.wh)

    def updateWavenumberRange(self):
        super().updateWavenumberRange()

        self.imageVisualizer.labelMinwn.setText("%.2f" % self.data.wmin)
        self.imageVisualizer.labelMaxwn.setText("%.2f" % self.data.wmax)
        wmin = min(self.data.wmin, constants.WMIN)
        wmax = max(self.data.wmax, constants.WMAX)
        self.imageVisualizer.lineEditWavenumber.setRange(
            wmin, wmax, default=.5*(wmin+wmax))
        # Make sure the sliders are in sync with the boxes
        self.imageVisualizer.wavenumberEdit()

    def updatedFile(self):
        self.imageVisualizer.updateSpectralImage()
        self.updateWhiteLightImages()
        self.imageVisualizer.selectSpectra()


    def updateDimensions(self, wh):
        try:
            self.imageVisualizer.plot_raw.setDimensions(wh)
        except ValueError:
            pass
        self.imageVisualizer.updateSpectralImage()

    def updateUnitsAndOrder(self, units, ltr):
        super().updateUnitsAndOrder(units, ltr)
        self.imageVisualizer.labelUnits1.setText(self.unitsRichText())
        self.imageVisualizer.plot_spectra.setOrder(ltr)
        self.imageVisualizer.horizontalSliderWavenumber.setInvertedAppearance(
            not ltr)

    def updateWhiteLightImages(self):
        imgcnt = len(self.data.images) if self.data.images else 0
        intern = self.spatialMode or imgcnt
        if intern:
            self.imageVisualizer.setImageCount(len(self.data.images))
            self.selectImage(self.imageVisualizer.horizontalSliderImage.value())
            self.imageVisualizer.plot_whitelight.setHidden(self.spatialMode)
        else:
            fname = self.whiteLightNames[self.data.curFile] if \
                self.data.curFile in self.whiteLightNames else \
                    os.path.splitext(self.data.curFile)[0]+'.jpg'
            self.imageVisualizer.plot_whitelight.load(filename=fname)
            self.imageVisualizer.plot_whitelight.setHidden(
                not self.imageVisualizer.plot_whitelight.is_loaded())
        self.imageVisualizer.layoutSpatial.setHidden(not self.spatialMode
                                                     and imgcnt < 2)
        self.imageVisualizer.pushButtonWhitelight.setHidden(intern)
        self.imageVisualizer.labelPixelSize.setHidden(not self.spatialMode)
        self.imageVisualizer.lineEditPixelSize.setHidden(not self.spatialMode)

    # White light image stuff
    def loadWhite(self):
        filename = self.getImageFileName(title="Load white light image",
                                         settingname='whitelightDir')
        if not filename:
            return
        self.imageVisualizer.plot_whitelight.load(filename=filename)
        ok = self.imageVisualizer.plot_whitelight.is_loaded()
        self.imageVisualizer.plot_whitelight.setHidden(not ok)
        if ok:
            self.whiteLightNames[self.data.curFile] = filename

    def selectImage(self, num):
        if self.data.images and len(self.data.images):
            if self.spatialMode:
                self.imageVisualizer.plot_raw.setImage(self.data.images[num])
            else:
                self.imageVisualizer.plot_whitelight.load(image=self.data.images[num])
            self.imageVisualizer.lineEditImageInfo.setText(
                self.data.images[num].name)


