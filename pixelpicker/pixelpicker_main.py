import matplotlib
matplotlib.use('QT5Agg')
import sys
from PyQt5 import uic
from PyQt5.QtWidgets import QApplication, QFileDialog, QErrorMessage
#from PyQt5.QtWidgets import QStyle, QProxyStyle, QMessageBox
from PyQt5.QtCore import pyqtSignal, Qt
from PyQt5.Qt import QMainWindow, qApp
#from os.path import basename, dirname
import numpy as np
import scipy.signal, scipy.io
import matplotlib.pyplot as plt
import os.path
from imagedata import ImageData

Ui_MainWindow = uic.loadUiType("pixelpicker_ui.ui")[0]

class MyMainWindow(QMainWindow, Ui_MainWindow):

    closing = pyqtSignal()

    fileOptions = QFileDialog.Options() | QFileDialog.DontUseNativeDialog

    def __init__(self,parent=None):
        super(MyMainWindow, self).__init__(parent)
        qApp.installEventFilter(self)
        self.setupUi(self)

        self.data = ImageData()

        self.errorMsg = QErrorMessage(self)
        self.errorMsg.setWindowModality(Qt.WindowModal)

        self.pushButtonLoadWL.clicked.connect(self.loadWhite)
        self.pushButtonLoadSpectra.clicked.connect(self.loadSpectra)
        self.pushButtonLoad.clicked.connect(self.loadAnnotations)
        self.pushButtonSave.clicked.connect(self.saveAnnotations)

        self.pushButtonClear.clicked.connect(self.plot_visual.clearSelected)

#        self.comboBoxMethod.currentIndexChanged.connect()
        self.horizontalSliderWn.valueChanged.connect(self.wavenumberSlide)
        self.lineEditWn.editingFinished.connect(self.wavenumberEdit)

        self.comboBoxVisual.currentIndexChanged.connect(self.imageProjection)
        self.comboBoxCmaps.currentTextChanged.connect(self.plot_visual.setCmap)

        self.lineEditWn.setFormat("%.2f")
#        self.lineEditWavenumber.setRange(min=wmin, max=wmax, default=.5*(wmin+wmax))

        self.comboBoxAnnotation.currentIndexChanged.connect(self.plot_visual.setAnnotation)

        self.plot_visual.addedPoint.connect(self.plot_whitelight.addPoint)
        self.plot_visual.removedPoint.connect(self.plot_whitelight.removePoint)
        self.plot_visual.alteredCounts.connect(self.updateCounts)
        self.plot_whitelight.clickedPoint.connect(self.plot_visual.clickOnePoint)

    def closeEvent(self, event):
        self.closing.emit()
        plt.close('all')
        self.deleteLater()
        qApp.quit()

    def loadSpectra(self):
        "Load a file or return (error message, traceback) from trying"
        fileName, _ = QFileDialog.getOpenFileName(self,
                                                  "Load FTIR image", "",
                                                  "Matlab files (*.mat);;All files (*)",
                                                  options=MyMainWindow.fileOptions)
        self.data.readmat(fileName)
        self.plot_visual.setData(self.data.wavenumber, self.data.raw, self.data.wh)
        self.horizontalSliderWn.setMaximum(len(self.data.wavenumber) - 1)
        self.lineEditWn.setRange(self.data.wmin, self.data.wmax)
        self.imageProjection()
        self.plot_whitelight.load(os.path.splitext(fileName)[0]+'.jpg')

    def loadAnnotations(self):
        fileName, _ = QFileDialog.getOpenFileName(self,
                                                  "Load annotation map", "",
                                                  "Matlab files (*.mat);;All files (*)",
                                                  options=MyMainWindow.fileOptions)
        if fileName:
            s = scipy.io.loadmat(fileName)['annotations']
            self.plot_visual.setAnnotations(s)

    def saveAnnotations(self):
        fileName, _ = QFileDialog.getSaveFileName(self,
                                                  "Save annotation map", "",
                                                  "Matlab files (*.mat);;All files (*)",
                                                  options=MyMainWindow.fileOptions)
        if fileName:
            scipy.io.savemat(fileName, {'annotations': self.plot_visual.getAnnotations()})


    # Image visualization
    def loadWhite(self):
        fileName, _ = QFileDialog.getOpenFileName(self,
                                                  "Load white light image", "",
                                                  "Image files (*.jpg *.png);;All files (*)",
                                                  options=MyMainWindow.fileOptions)
        if fileName:
            self.plot_whitelight.load(fileName)

    def imageProjection(self):
        meth = self.comboBoxVisual.currentIndex()
        iswn = meth == 2
        self.lineEditWn.setEnabled(iswn)
        self.horizontalSliderWn.setEnabled(iswn)
        self.plot_visual.setProjection(meth, -1-self.horizontalSliderWn.value())

    def updateCounts(self, counts):
        for i in range(self.comboBoxAnnotation.count()):
            t = self.comboBoxAnnotation.itemText(i)
            p = t.rfind('(')
            t = t[:p] if p >= 0 else t + ' '
            self.comboBoxAnnotation.setItemText(i, t + '(%d)' % counts[i])

    def sliderToBox(self, slider, box, ixfinder):
        wn = self.plot_visual.getWavenumbers()
        if wn is not None:
            if (not box.hasAcceptableInput() or
                    slider.value() != ixfinder(wn, box.value())):
                box.setValue(wn[-1-slider.value()])
        elif (not box.hasAcceptableInput() or
                slider.value() != int(round(slider.maximum() *
                (box.value() - self.wmin) / (self.wmax - self.wmin)))):
            box.setValue(self.wmin + (self.wmax - self.wmin) *
                         slider.value() / slider.maximum())

    def boxToSlider(self, slider, box, ixfinder):
        wn = self.plot_visual.getWavenumbers()
        if wn is not None:
            if box.hasAcceptableInput():
                slider.setValue(ixfinder(wn, box.value()))
            else:
                box.setValue(wn[-1-slider.value()])
        else:
            slider.setValue(int(round(slider.maximum() *
                  (box.value() - self.wmin) / (self.wmax - self.wmin))))

    def wavenumberSlide(self):
        self.sliderToBox(self.horizontalSliderWn, self.lineEditWn,
                         lambda wn, val: len(wn)-1 - (np.abs(wn - val)).argmin())
        self.plot_visual.setProjection(2, -1-self.horizontalSliderWn.value())

    def wavenumberEdit(self):
        self.boxToSlider(self.horizontalSliderWn, self.lineEditWn,
                         lambda wn, val: len(wn)-1 - (np.abs(wn - val)).argmin())




if __name__ == '__main__':
    app = QApplication.instance()
    if not app:
        app = QApplication(sys.argv)
    window = MyMainWindow()
    window.show()
    sys.exit(app.exec_())

