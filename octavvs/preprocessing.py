import os
import traceback
from os.path import basename, dirname
from pkg_resources import resource_filename
import argparse
from io import BytesIO

from PyQt5.QtWidgets import QFileDialog, QErrorMessage, QInputDialog, QDialog
from PyQt5.QtWidgets import QMessageBox
from PyQt5.QtCore import QThread, pyqtSignal, pyqtSlot, Qt
from PyQt5.Qt import qApp
from PyQt5 import uic

import numpy as np
import scipy.signal, scipy.io
import matplotlib
matplotlib.use('QT5Agg')
import matplotlib.pyplot as plt

from .prep.prepworker import PrepWorker, ABCWorker, PrepParameters
from .miccs import SpectralData, FileLoader
from .miccs import OctavvsMainWindow, NoRepeatStyle, run_octavvs_application

Ui_MainWindow = uic.loadUiType(resource_filename(__name__, "prep/preprocessing_ui.ui"))[0]
Ui_DialogSCAdvanced = uic.loadUiType(resource_filename(__name__, "prep/scadvanced.ui"))[0]


class DialogSCAdvanced(QDialog, Ui_DialogSCAdvanced):
    def __init__(self, parent=None):
        QDialog.__init__(self, parent)
        self.setupUi(self)

class MyMainWindow(FileLoader, OctavvsMainWindow, Ui_MainWindow):

    closing = pyqtSignal()
    startRmiesc = pyqtSignal(SpectralData, dict)
    startAC = pyqtSignal(np.ndarray, np.ndarray, dict)
    startBC = pyqtSignal(np.ndarray, np.ndarray, str, dict)
    startBatch = pyqtSignal(SpectralData, PrepParameters, str, bool)

    bcLambdaRange = np.array([0, 8])
    bcPRange = np.array([-5, 0])

    def __init__(self, parent=None, files=None):
        super().__init__(parent)

        self.data = SpectralData()
        self.rmiescRunning = 0   # 1 for rmiesc, 2 for batch

        # Avoid repeating spinboxes
        self.spinBoxItersBC.setStyle(NoRepeatStyle())
        self.spinBoxNclusScat.setStyle(NoRepeatStyle())
        self.spinBoxNIteration.setStyle(NoRepeatStyle())
        self.spinBoxPolyOrder.setStyle(NoRepeatStyle())
        self.spinBoxSpectra.setStyle(NoRepeatStyle())
        self.spinBoxWindowLength.setStyle(NoRepeatStyle())

        self.fileLoader.setSuffix('_prep')

        self.pushButtonWhitelight.clicked.connect(self.loadWhite)

        self.comboBoxMethod.currentIndexChanged.connect(self.imageProjection)
        self.horizontalSliderWavenumber.valueChanged.connect(self.wavenumberSlide)
        self.lineEditWavenumber.editingFinished.connect(self.wavenumberEdit)

        self.plot_spectra.clicked.connect(self.plot_spectra.popOut)
        self.spinBoxSpectra.valueChanged.connect(self.selectSpectra)
        self.checkBoxAutopick.toggled.connect(self.selectSpectra)
        self.plot_visual.changedSelected.connect(self.spinBoxSpectra.setValue)
        self.plot_visual.changedSelected.connect(self.selectedSpectraUpdated)
        self.comboBoxCmaps.currentTextChanged.connect(self.plot_visual.setCmap)
        self.pushButtonExpandProjection.clicked.connect(self.plot_visual.popOut)

        self.plot_spectra.updated.connect(self.updateAC)
        self.plot_AC.clicked.connect(self.plot_AC.popOut)
        self.checkBoxAC.toggled.connect(self.updateAC)
        self.checkBoxSpline.toggled.connect(self.updateAC)
        self.checkBoxLocalPeak.toggled.connect(self.updateAC)
        self.checkBoxSmoothCorrected.toggled.connect(self.updateAC)
        self.pushButtonACLoadReference.clicked.connect(self.loadACReference)
        self.lineEditACReference.editingFinished.connect(self.updateAC)

        self.plot_AC.updated.connect(self.updateSC)
        self.plot_SC.clicked.connect(self.plot_SC.popOut)
        self.plot_visual.changedSelected.connect(self.updateSCplot)
        self.pushButtonSCRefresh.clicked.connect(self.refreshSC)
        self.comboBoxReference.currentIndexChanged.connect(self.updateSC)
        self.pushButtonLoadOther.clicked.connect(self.loadOtherReference)
        self.comboBoxReference.currentIndexChanged.connect(self.updateSC)
        self.pushButtonLoadOther.clicked.connect(self.updateSC)
        self.spinBoxNIteration.valueChanged.connect(self.updateSC)
        self.checkBoxClusters.toggled.connect(self.scClustersToggle)
        self.spinBoxNclusScat.valueChanged.connect(self.updateSC)
        self.checkBoxStabilize.toggled.connect(self.updateSC)
        self.pushButtonSCStop.clicked.connect(self.scStop)
        self.pushButtonRunStop.clicked.connect(self.scStop)

        self.dialogSCAdvanced = DialogSCAdvanced()
        self.pushButtonSCAdvanced.clicked.connect(self.dialogSCAdvanced.show)
        self.pushButtonSCAdvanced.clicked.connect(self.dialogSCAdvanced.raise_)
        self.dialogSCAdvanced.comboBoxSCAlgo.currentIndexChanged.connect(self.updateSC)
        self.dialogSCAdvanced.spinBoxSCResolution.valueChanged.connect(self.updateSC)
        self.dialogSCAdvanced.lineEditSCamin.editingFinished.connect(self.updateSC)
        self.dialogSCAdvanced.lineEditSCamax.editingFinished.connect(self.updateSC)
        self.dialogSCAdvanced.lineEditSCdmin.editingFinished.connect(self.updateSC)
        self.dialogSCAdvanced.lineEditSCdmax.editingFinished.connect(self.updateSC)
        self.dialogSCAdvanced.checkBoxSCLinear.toggled.connect(self.updateSC)
        self.dialogSCAdvanced.checkBoxSCRenormalize.toggled.connect(self.updateSC)
        self.dialogSCAdvanced.spinBoxSCPCA.valueChanged.connect(self.updateSC)
        self.dialogSCAdvanced.radioButtonSCPCAFixed.toggled.connect(self.toggleSCPCA)
        self.dialogSCAdvanced.radioButtonSCPCADynamic.toggled.connect(self.toggleSCPCA)
        self.dialogSCAdvanced.spinBoxSCPCAMax.valueChanged.connect(self.updateSC)
        self.dialogSCAdvanced.lineEditSCPCAVariance.setFormat("%.8g")
        self.dialogSCAdvanced.lineEditSCPCAVariance.setRange(50, 99.999999)
        self.dialogSCAdvanced.lineEditSCPCAVariance.editingFinished.connect(self.updateSC)
        self.dialogSCAdvanced.checkBoxSCAutoIters.toggled.connect(self.updateSC)
        self.dialogSCAdvanced.lineEditSCMinImprov.setRange(0, 99)
        self.dialogSCAdvanced.lineEditSCMinImprov.setFormat("%g")
        self.dialogSCAdvanced.lineEditSCMinImprov.editingFinished.connect(self.updateSC)


        self.plot_AC.updated.connect(self.updateSGF)
        self.plot_SGF.clicked.connect(self.plot_SGF.popOut)
        self.checkBoxSC.toggled.connect(self.updateSGF)
        self.plot_SC.updated.connect(self.updateSGF)
        self.checkBoxSGF.toggled.connect(self.updateSGF)
        self.spinBoxWindowLength.valueChanged.connect(self.updateSGF)
        self.spinBoxPolyOrder.valueChanged.connect(self.updateSGF)

        self.plot_SGF.updated.connect(self.updateSR)
        self.plot_SR.clicked.connect(self.plot_SR.popOut)
        self.checkBoxSR.toggled.connect(self.updateSR)
        self.horizontalSliderMin.valueChanged.connect(self.srMinSlide)
        self.horizontalSliderMax.valueChanged.connect(self.srMaxSlide)
        self.lineEditMinwn.editingFinished.connect(self.srMinEdit)
        self.lineEditMaxwn.editingFinished.connect(self.srMaxEdit)

        self.plot_SR.updated.connect(self.updateBC)
        self.plot_BC.clicked.connect(self.plot_BC.popOut)
        self.comboBoxBaseline.currentIndexChanged.connect(self.bcMethod)
        self.horizontalSliderLambda.valueChanged.connect(self.bcLambdaSlide)
        self.lineEditLambda.editingFinished.connect(self.bcLambdaEdit)
        self.horizontalSliderP.valueChanged.connect(self.bcPSlide)
        self.lineEditP.editingFinished.connect(self.bcPEdit)
        self.spinBoxItersBC.valueChanged.connect(self.updateBC)
        self.bcLambdaEdit()
        self.bcPEdit()

        self.plot_BC.updated.connect(self.updateNorm)
        self.plot_norm.clicked.connect(self.plot_norm.popOut)
        self.comboBoxNormMethod.currentIndexChanged.connect(self.updateNorm)
        self.lineEditNormWavenum.editingFinished.connect(self.updateNorm)

        self.pushButtonSaveParameters.clicked.connect(self.saveParameters)
        self.pushButtonLoadParameters.clicked.connect(self.loadParameters)

        self.pushButtonRun.clicked.connect(self.runBatch)

        # Defaults when no data loaded
        self.default_wmin = 800
        self.default_wmax = 4000
        self.scSettings = {}

        self.worker = PrepWorker()
        self.workerThread = QThread()
        self.worker.moveToThread(self.workerThread)
        self.closing.connect(self.workerThread.quit)
        self.startRmiesc.connect(self.worker.rmiesc)
        self.worker.done.connect(self.scDone)
        self.worker.stopped.connect(self.scStopped)
        self.worker.failed.connect(self.scFailed)
        self.worker.progress.connect(self.scProgress)
        self.worker.batchProgress.connect(self.batchProgress)
        self.worker.progressPlot.connect(self.plot_SC.progressPlot)
        self.worker.fileLoaded.connect(self.updateFile)
        self.worker.loadFailed.connect(self.showLoadErrorMessage)
        self.startBatch.connect(self.worker.bigBatch)
        self.worker.batchDone.connect(self.batchDone)
        self.workerThread.start()

        self.bcNext = None
        self.acNext = None

        self.abcWorker = ABCWorker()
        self.abcWorkerThread = QThread()
        self.abcWorker.moveToThread(self.abcWorkerThread)
        self.closing.connect(self.abcWorkerThread.quit)
        self.startBC.connect(self.abcWorker.bc)
        self.startAC.connect(self.abcWorker.ac)
        self.abcWorker.acDone.connect(self.acDone)
        self.abcWorker.bcDone.connect(self.bcDone)
        self.abcWorker.acFailed.connect(self.acFailed)
        self.abcWorker.bcFailed.connect(self.bcFailed)
        self.abcWorkerThread.start()

        self.lineEditWavenumber.setFormat("%.2f")
        self.lineEditMinwn.setFormat("%.2f")
        self.lineEditMaxwn.setFormat("%.2f")
        self.lineEditLambda.setFormat("%.4g")
        self.lineEditP.setFormat("%.4g")
        self.lineEditLambda.setRange(*(10. ** self.bcLambdaRange))
        self.lineEditP.setRange(*(10. ** self.bcPRange))

        self.lineEditWidth.editingFinished.connect(lambda: self.updateDimensions(0))
        self.lineEditHeight.editingFinished.connect(lambda: self.updateDimensions(1))

        self.dialogSCAdvanced.lineEditSCamin.setFormat("%.4g")
        self.dialogSCAdvanced.lineEditSCamax.setFormat("%.4g")
        self.dialogSCAdvanced.lineEditSCdmin.setFormat("%.4g")
        self.dialogSCAdvanced.lineEditSCdmax.setFormat("%.4g")
        self.dialogSCAdvanced.lineEditSCamin.setRange(1., 4.)
        self.dialogSCAdvanced.lineEditSCamax.setRange(1., 4.)
        self.dialogSCAdvanced.lineEditSCdmin.setRange(.1, 100.)
        self.dialogSCAdvanced.lineEditSCdmax.setRange(.1, 100.)

        self.updateWavenumberRange()
        self.bcMethod()

        self.post_setup()
        if files is not None and files != []:
            self.updateFileList(files, False) # Load files passed as arguments

    def closeEvent(self, event):
        self.worker.halt = True
        self.abcWorker.haltBC = True
        self.closing.emit()
        plt.close('all')
        self.workerThread.wait()
        self.abcWorkerThread.wait()
        self.deleteLater()
        self.dialogSCAdvanced.close()
        qApp.quit()

    def updateWavenumberRange(self):
        if self.data.wavenumber is not None:
            wns = len(self.data.wavenumber)
            self.lineEditLength.setText(str(wns))
            # Update sliders before boxes to avoid errors from triggered updates
            self.horizontalSliderWavenumber.setMaximum(wns-1)
            self.horizontalSliderMin.setMaximum(wns-1)
            self.horizontalSliderMax.setMaximum(wns-1)

        self.labelMinwn.setText("%.2f" % self.data.wmin)
        self.labelMaxwn.setText("%.2f" % self.data.wmax)
        wmin = min(self.data.wmin, self.default_wmin)
        wmax = max(self.data.wmax, self.default_wmax)
        self.lineEditWavenumber.setRange(wmin, wmax, default=.5*(wmin+wmax))
        self.lineEditMinwn.setRange(wmin, wmax, default=wmin)
        self.lineEditMaxwn.setRange(wmin, wmax, default=wmax)
        self.lineEditNormWavenum.setRange(wmin, wmax, default=.5*(wmin+wmax))

        # Make sure the sliders are in sync with the boxes
        self.wavenumberEdit()
        self.srMinEdit()
        self.srMaxEdit()


    def loadFolder(self, *argv, **kwargs):
        assert self.rmiescRunning == 0
        super().loadFolder(*argv, **kwargs)

    @pyqtSlot(int)
    def updateFile(self, num):
        super().updateFile(num)

        self.lineEditWidth.setText(str(self.data.wh[0]))
        self.lineEditHeight.setText(str(self.data.wh[1]))

        self.plot_visual.setData(self.data.wavenumber, self.data.raw, self.data.wh)
        self.updateWavenumberRange()
        self.imageProjection()

        if self.bcNext:
            self.abcWorker.haltBC = True
            self.bcNext = True
        if self.acNext:
            self.abcWorker.haltBC = True
            self.acNext = True

        self.clearSC()
        self.imageProjection()
        self.selectSpectra()
        if self.data.image is not None:
            self.plot_whitelight.load(BytesIO(self.data.image[0]), self.data.image[1])
        else:
            self.plot_whitelight.load(os.path.splitext(self.data.curFile)[0]+'.jpg')

    @pyqtSlot(str, str, str)
    def showLoadErrorMessage(self, file, err, details):
        "Error message from loading a file in the worker thread, with abort option"
        q = self.loadErrorBox(file, (err, details if details else None))
        q.addButton('Skip file', QMessageBox.AcceptRole)
        q.addButton('Abort', QMessageBox.AcceptRole)
        if q.exec():
            self.scStop()

    def updateDimensions(self, dimnum):
        if dimnum == 0:
            self.data.setWidth(self.lineEditWidth.text())
        else:
            self.data.setHeight(self.lineEditHeight.text())
        self.lineEditWidth.setText(str(self.data.wh[0]))
        self.lineEditHeight.setText(str(self.data.wh[1]))
        try:
            self.plot_visual.setDimensions(self.data.wh)
        except ValueError:
            pass
        self.imageProjection()

    # Image visualization
    def loadWhite(self):
        filename = self.getImageFileName(title="Load white light image",
                                         settingname='whitelightDir')
        if filename:
            self.plot_whitelight.load(filename)

    def imageProjection(self):
        meth = self.comboBoxMethod.currentIndex()
        iswn = meth == 2
        self.lineEditWavenumber.setEnabled(iswn)
        self.horizontalSliderWavenumber.setEnabled(iswn)
        self.plot_visual.setProjection(meth, -1-self.horizontalSliderWavenumber.value())

    def wavenumberSlide(self):
        self.sliderToBox(self.horizontalSliderWavenumber, self.lineEditWavenumber,
                         lambda: self.plot_visual.getWavenumbers(),
                         lambda wn, val: len(wn)-1 - (np.abs(wn - val)).argmin())
        self.plot_visual.setProjection(2, -1-self.horizontalSliderWavenumber.value())

    def wavenumberEdit(self):
        self.boxToSlider(self.horizontalSliderWavenumber, self.lineEditWavenumber,
                         lambda: self.plot_visual.getWavenumbers(),
                         lambda wn, val: len(wn)-1 - (np.abs(wn - val)).argmin())

    # Selection of spectra
    def selectedSpectraUpdated(self):
        self.plot_spectra.setData(self.plot_visual.getWavenumbers(), None,
                                  self.plot_visual.getSelectedData())

    def selectSpectra(self):
        self.plot_visual.setSelectedCount(self.spinBoxSpectra.value(),
                                          self.checkBoxAutopick.isChecked())


    # AC, Atmospheric correction
    def loadACReference(self):
        startdir = os.path.realpath(os.path.join(
                os.getcwd(), os.path.dirname(__file__), 'miccs', 'reference'))
        ref, _ = QFileDialog.getOpenFileName(self, "Load atmospheric reference spectrum",
                                             directory=self.settings.value('atmRefDir', startdir),
                                             filter="Matrix file (*.mat)",
                                             options=MyMainWindow.fileOptions)
        self.lineEditACReference.setText(ref)
        self.settings.setValue('atmRefDir', dirname(ref))
        self.updateAC()

    def updateAC(self):
        wn = self.plot_spectra.getWavenumbers()
        if wn is None:
            return
        indata = self.plot_spectra.getSpectra()
        if not self.checkBoxAC.isChecked() or len(indata) == 0:
            self.plot_AC.setData(wn, indata, None)
            self.labelACInfo.setText('')
            return
        opt = dict(cut_co2=self.checkBoxSpline.isChecked(),
                   extra=self.checkBoxLocalPeak.isChecked(),
                   smooth=self.checkBoxSmoothCorrected.isChecked(),
                   ref=self.lineEditACReference.text())
        args = [ wn, indata, opt ]
        if self.acNext:
            self.acNext = args
        else:
            self.startAC.emit(*args)
            self.acNext = True

    @pyqtSlot(np.ndarray, np.ndarray, np.ndarray, np.ndarray)
    def acDone(self, wavenumber, indata, corr, factors):
        self.plot_AC.setData(wavenumber, indata, corr)
        if self.acNext and self.acNext is not True:
            self.startAC.emit(*self.acNext)
            self.acNext = True
        else:
            self.acNext = None
        if factors is not None:
            h = 'H<sub>2</sub>O: %.2f%%' % (factors[0] * 100)
            if factors[0] > .05:
                h = '<span style="color:red">%s</span>' % (h)
            c = 'CO<sub>2</sub>: %.2f%%' % (factors[1] * 100)
            self.labelACInfo.setText('Mean correction: %s %s' % (h, c))
        else:
            self.labelACInfo.setText('')

    @pyqtSlot(str)
    def acFailed(self, err):
        if err != '':
            self.errorMsg.showMessage('AC failed: ' + err)
        self.acDone(None, None, None, None)


    # SC, Scattering correction
    def loadOtherReference(self):
        ref, _ = QFileDialog.getOpenFileName(self, "Open reference spectrum",
                                             filter="Matrix file (*.mat)",
                                             directory=self.settings.value('scRefDir', None),
                                             options=MyMainWindow.fileOptions)
        self.lineEditReferenceName.setText(ref)
        self.settings.setValue('scRefDir', dirname(ref))
        self.comboBoxReference.setCurrentIndex(3)

    def scClustersToggle(self):
        self.spinBoxNclusScat.setEnabled(self.checkBoxClusters.isChecked())
        self.checkBoxStabilize.setEnabled(self.checkBoxClusters.isChecked())
        self.updateSC()

    def getSCSettings(self):
        """ Returns SC settings as a dict, to enable comparison with latest run.
        """
        params = self.getParameters()
        p = { k: v for k, v in vars(params).items() if k.startswith('sc')}
        # Make shadowed options irrelevant to the comparison
        if p['scRef'] != 'Other':
            p['scOtherRef'] = ''
        if not p['scClustering']:
            p['scClusters'] = 0
            p['scStable'] = False
        if p['scPCADynamic']:
            p['scPCA'] = 0
        else:
            p['scPCAMax'] = 0
            p['scPCAVariance'] = 0
        if not p['scAutoIters']:
            p['scMinImprov'] = 0
        # Include atm correction only if used
        if params.acDo:
            p.update({ k: v for k, v in vars(params).items() if k.startswith('ac')})
        else:
            p['acDo'] = False
        # Selection of pixels to correct, or all if using clustering.
        p['selected'] = self.plot_visual.getSelected() if not p['scClustering'] else None
        return p

    def scSettingsChanged(self, settings=None):
        """ Are these settings/data different from the previous ones used for SC?
            settings=None reads settings from the UI
        """
        if self.plot_SC.getWavenumbers() is None:
            return False
        if not self.scSettings:
            return True
        if not settings:
            settings = self.getSCSettings()
        return settings != self.scSettings

    def toggleSCPCA(self):
#        fixed = self.dialogSCAdvanced.radioButtonSCPCAFixed.isChecked()
#        self.dialogSCAdvanced.spinBoxSCPCA.setEnabled(fixed)
#        self.dialogSCAdvanced.spinBoxSCPCAMax.setEnabled(not fixed)
#        self.dialogSCAdvanced.lineEditSCPCAVariance.setEnabled(not fixed)
        self.updateSC()

    def updateSC(self):
        if self.rmiescRunning:
            return
        changed = self.scSettingsChanged()
        self.pushButtonSCRefresh.setEnabled(changed)

    def updateSCplot(self):
        self.plot_SC.setSelected(self.plot_visual.getSelected())

    def toggleRunning(self, newstate):
        onoff = not newstate
        self.fileLoader.setEnabled(onoff)
        self.pushButtonRun.setEnabled(onoff)
        self.pushButtonSCRefresh.setEnabled(onoff)
        if newstate:
            self.progressBarSC.setValue(0)
        self.pushButtonSCStop.setEnabled(newstate == 1)
        self.pushButtonRunStop.setEnabled(newstate == 2)
        self.rmiescRunning = newstate

    def refreshSC(self):
        if self.rmiescRunning:
            return
        self.scNewSettings = self.getSCSettings()
        self.toggleRunning(1)
        self.startRmiesc.emit(self.data, self.scNewSettings)


    def clearSC(self):
        if self.rmiescRunning == 1:
            return
        self.plot_SC.setData(self.plot_visual.getWavenumbers(), None, None, None, None)
        self.scSettings = None

    def scStop(self):
        self.worker.halt = True
        self.pushButtonSCStop.setEnabled(False)
        self.pushButtonRunStop.setEnabled(False)

    @pyqtSlot()
    def scStopped(self):
        self.plot_SC.endProgress()
        self.worker.halt = False
        if self.rmiescRunning == 1:
            self.scNewSettings = None
            self.toggleRunning(0)
        self.updateSC()

    @pyqtSlot(str, str)
    def scFailed(self, err, trace):
        self.errorMsg.showMessage('RMieSC failed:\n' + err + "\n\n" + trace)
        self.scStopped()

    @pyqtSlot(np.ndarray, np.ndarray, np.ndarray)
    def scDone(self, wavenumber, indata, corr):
        self.scSettings = self.scNewSettings
        self.plot_SC.setData(wavenumber, indata, corr, self.scSettings['selected'],
                             self.plot_visual.getSelected())
        self.scStopped()

    @pyqtSlot(int, int)
    def scProgress(self, a, b):
        if a < 0:
            self.progressBarSC.setFormat(['', 'Atm %p%', 'RMieSC %p%', 'SGF %p%', 'Baseline'][-a])
        if a == -2:
            self.plot_SC.prepareProgress(self.plot_visual.getWavenumbers())
        self.progressBarSC.setValue(max(a, 0))
        self.progressBarSC.setMaximum(b)


    # SGF, Denoising with Savtisky-Golay filter
    def updateSGF(self):
        window = self.spinBoxWindowLength.value()
        if not window & 1:
            window += 1
            self.spinBoxWindowLength.setValue(window)
        order = self.spinBoxPolyOrder.value()
        if self.checkBoxSC.isChecked():
            wn = self.plot_SC.getWavenumbers()
            indata = self.plot_SC.getSpectra()
        else:
            wn = self.plot_AC.getWavenumbers()
            indata = self.plot_AC.getSpectra()
        if (self.checkBoxSGF.isChecked() and indata is not None and
            len(indata) > 0 and window > order and indata.shape[1] >= window):
            corr = scipy.signal.savgol_filter(indata, window, order, axis=1)
        else:
            corr = None
        self.plot_SGF.setData(wn, indata, corr)


    # SR, Spectral region cutting
    def updateSR(self):
        wn = self.plot_SGF.getWavenumbers()
        indata = self.plot_SGF.getSpectra()
        if indata is None or not self.checkBoxSR.isChecked():
            self.plot_SR.setData(wn, indata, None, None, 1234, 2345)
            return
        cut = range(len(wn)-1 - self.horizontalSliderMax.value(),
                    len(wn) - self.horizontalSliderMin.value())
        if len(cut) < 2:
            self.plot_SR.setData(wn, indata, None, None, 1234, 2345)
        else:
            self.plot_SR.setData(wn, indata,
                                 self.plot_SGF.getSpectra()[:, cut],
                                 self.plot_SGF.getWavenumbers()[cut],
                                 self.lineEditMinwn.value(), self.lineEditMaxwn.value())

    def srMinSlide(self):
        self.sliderToBox(self.horizontalSliderMin, self.lineEditMinwn,
                         lambda: self.plot_SGF.getWavenumbers(),
                         lambda wn, val: wn[::-1].searchsorted(val, 'left'))
        self.updateSR()

    def srMinEdit(self):
        self.boxToSlider(self.horizontalSliderMin, self.lineEditMinwn,
                         lambda: self.plot_SGF.getWavenumbers(),
                         lambda wn, val: wn[::-1].searchsorted(val, 'left'))

    def srMaxSlide(self):
        self.sliderToBox(self.horizontalSliderMax, self.lineEditMaxwn,
                         lambda: self.plot_SGF.getWavenumbers(),
                         lambda wn, val: wn[::-1].searchsorted(val, 'right')-1)
        self.updateSR()

    def srMaxEdit(self):
        self.boxToSlider(self.horizontalSliderMax, self.lineEditMaxwn,
                         lambda: self.plot_SGF.getWavenumbers(),
                         lambda wn, val: wn[::-1].searchsorted(val, 'right')-1)

    def bcName(self):
        return ['none', 'rubberband', 'concaverubberband', 'asls', 'arpls'
                ][self.comboBoxBaseline.currentIndex()]
    def bcIndex(self, val):
        return ['none', 'rubberband', 'concaverubberband', 'asls', 'arpls'
                ].index(val)

    # BC, Baseline Correction
    def updateBC(self):
        indata = self.plot_SR.getSpectra()
        wn = self.plot_SR.getWavenumbers()
        bc = self.bcName()
        if indata is None or bc == 'none':
            self.plot_BC.setData(wn, indata, None)
            return
        if bc == 'asls':
            param = {'lam': self.lineEditLambda.value(), 'p': self.lineEditP.value()}
        elif bc == 'arpls':
            param = {'lam': self.lineEditLambda.value()}
        elif bc == 'rubberband':
            param = {}
        elif bc == 'concaverubberband':
            param = {'iters': self.spinBoxItersBC.value()}
        if self.bcNext:
            self.abcWorker.haltBC = True
            self.bcNext = [ wn, indata, bc, param ]
        else:
            self.startBC.emit(wn, indata, bc, param)
            self.bcNext = True

    @pyqtSlot(np.ndarray, np.ndarray, np.ndarray)
    def bcDone(self, wavenumber, indata, corr):
        if(wavenumber is not None):
            self.plot_BC.setData(wavenumber, indata, corr)
        self.abcWorker.haltBC = False
        if not self.bcNext or self.bcNext is True:
            self.bcNext = None
        else:
            self.startBC.emit(*self.bcNext)
            self.bcNext = True

    @pyqtSlot(str)
    def bcFailed(self, err):
        if err != '':
            self.errorMsg.showMessage('BC failed: ' + err)
        self.bcDone(None, None, None)

    def bcMethod(self):
        params = dict(none=(0,0,0), asls=(1,1,0), arpls=(1,0,0),
                      rubberband=(0,0,0), concaverubberband=(0,0,1))
        params = params[self.bcName()]
        self.horizontalSliderLambda.setEnabled(params[0])
        self.lineEditLambda.setEnabled(params[0])
        self.labelLambda.setEnabled(params[0])
        self.horizontalSliderP.setEnabled(params[1])
        self.lineEditP.setEnabled(params[1])
        self.labelP.setEnabled(params[1])
        self.spinBoxItersBC.setEnabled(params[2])
        self.labelItersBC.setEnabled(params[2])
        self.updateBC()

    def expSliderPos(self, box, slider, rng):
        return round(slider.maximum() * (np.log10(box.value()) - rng[0]) / (rng[1] - rng[0]))

    def expBoxVal(self, slider, rng):
        return 10.**(slider.value() / slider.maximum() * (rng[1] - rng[0]) + rng[0])

    def bcLambdaSlide(self):
        if (not self.lineEditLambda.hasAcceptableInput() or
            self.expSliderPos(self.lineEditLambda, self.horizontalSliderLambda, self.bcLambdaRange) !=
            self.horizontalSliderLambda.value()):
            self.lineEditLambda.setValue(self.expBoxVal(self.horizontalSliderLambda,
                                                        self.bcLambdaRange))
            self.updateBC()

    def bcLambdaEdit(self):
        if self.lineEditLambda.hasAcceptableInput():
            self.horizontalSliderLambda.setValue(self.expSliderPos(
                    self.lineEditLambda, self.horizontalSliderLambda, self.bcLambdaRange))
        else:
            self.lineEditLambda.setValue(self.expBoxVal(self.horizontalSliderLambda,
                                                        self.bcLambdaRange))
        self.updateBC()

    def bcPSlide(self):
        if (not self.lineEditP.hasAcceptableInput() or
            self.expSliderPos(self.lineEditP, self.horizontalSliderP, self.bcPRange) !=
            self.horizontalSliderP.value()):
            self.lineEditP.setValue(self.expBoxVal(self.horizontalSliderP, self.bcPRange))
            self.updateBC()

    def bcPEdit(self):
        if self.lineEditP.hasAcceptableInput():
            self.horizontalSliderP.setValue(self.expSliderPos(
                    self.lineEditP, self.horizontalSliderP, self.bcPRange))
        else:
            self.lineEditP.setValue(self.expBoxVal(self.horizontalSliderP, self.bcPRange))
        self.updateBC()


    # Normalization
    def normName(self):
        return ['none', 'area', 'wn', 'mean', 'max'][self.comboBoxNormMethod.currentIndex()]
    def normIndex(self, val):
        return ['none', 'area', 'wn', 'mean', 'max'].index(val)

    def updateNorm(self):
        meth = self.normName()
        self.lineEditNormWavenum.setEnabled(meth == 'wn')
        wn = self.plot_BC.getWavenumbers()
        indata = self.plot_BC.getSpectra()
        if indata is None or len(indata) == 0:
            self.plot_norm.setData(wn, indata, indata)
            return
        if not self.lineEditNormWavenum.hasAcceptableInput():
            self.lineEditNormWavenum.setValue()
        if meth == 'none':
            self.plot_norm.setData(wn, indata, indata)
        elif meth == 'mean':
            self.plot_norm.setData(wn, indata, (indata.T / indata.mean(axis=1)).T)
        elif meth == 'area':
            self.plot_norm.setData(wn, indata, (indata.T / -np.trapz(indata, wn, axis=1)).T)
        elif meth == 'wn':
            idx = (np.abs(wn-self.lineEditNormWavenum.value())).argmin()
            self.plot_norm.setData(wn, indata, (indata.T / indata[:, idx]).T)
        elif meth == 'max':
            self.plot_norm.setData(wn, indata, (indata.T / indata.max(axis=1)).T)


    # Loading and saving parameters
    def getParameters(self):
        p = PrepParameters()
        self.fileLoader.saveParameters(p)

        p.plotMethod = self.comboBoxMethod.currentIndex()
        p.plotColors = self.comboBoxCmaps.currentText()
        p.plotWavenum = self.lineEditWavenumber.value()
        p.spectraCount = self.spinBoxSpectra.value()
        p.spectraAuto = self.checkBoxAutopick.isChecked()
        p.acDo = self.checkBoxAC.isChecked()
        p.acSpline = self.checkBoxSpline.isChecked()
        p.acLocal = self.checkBoxLocalPeak.isChecked()
        p.acSmooth = self.checkBoxSmoothCorrected.isChecked()
        p.acReference = self.lineEditACReference.text()
        p.scDo = self.checkBoxSC.isChecked()
        p.scRef = self.comboBoxReference.currentText()
        p.scOtherRef = self.lineEditReferenceName.text()
        p.scIters = self.spinBoxNIteration.value()
        p.scClustering = self.checkBoxClusters.isChecked()
        p.scClusters = self.spinBoxNclusScat.value()
        p.scStable = self.checkBoxStabilize.isChecked()
        p.scAlgorithm = self.dialogSCAdvanced.comboBoxSCAlgo.currentText()
        p.scResolution = self.dialogSCAdvanced.spinBoxSCResolution.value()
        p.scAmin = self.dialogSCAdvanced.lineEditSCamin.value()
        p.scAmax = self.dialogSCAdvanced.lineEditSCamax.value()
        p.scDmin = self.dialogSCAdvanced.lineEditSCdmin.value()
        p.scDmax = self.dialogSCAdvanced.lineEditSCdmax.value()
        p.scLinear = self.dialogSCAdvanced.checkBoxSCLinear.isChecked()
        p.scRenormalize = self.dialogSCAdvanced.checkBoxSCRenormalize.isChecked()
        p.scPCADynamic = self.dialogSCAdvanced.radioButtonSCPCADynamic.isChecked()
        p.scPCA = self.dialogSCAdvanced.spinBoxSCPCA.value()
        p.scPCAMax = self.dialogSCAdvanced.spinBoxSCPCAMax.value()
        p.scPCAVariance = self.dialogSCAdvanced.lineEditSCPCAVariance.value()
        p.scAutoIters = self.dialogSCAdvanced.checkBoxSCAutoIters.isChecked()
        p.scMinImprov = self.dialogSCAdvanced.lineEditSCMinImprov.value()
        p.sgfDo = self.checkBoxSGF.isChecked()
        p.sgfWindow = self.spinBoxWindowLength.value()
        p.sgfOrder = self.spinBoxPolyOrder.value()
        p.srDo = self.checkBoxSR.isChecked()
        p.srMin = self.lineEditMinwn.value()
        p.srMax = self.lineEditMaxwn.value()
        p.bcMethod = self.bcName()
        p.bcIters = self.spinBoxItersBC.value()
        p.bcLambda = self.lineEditLambda.value()
        p.bcP = self.lineEditP.value()
        p.normMethod = self.normName()
        p.normWavenum = self.lineEditNormWavenum.value()
        return p

    def saveParameters(self):
        filename = self.getSaveFileName(
                "Save preprocessing settings",
                filter="Setting files (*.pjs);;All files (*)",
                settingname='settingsDir',
                suffix='pjs')
        if filename:
            try:
                self.getParameters().save(filename)
            except Exception as e:
                self.showDetailedErrorMessage(
                        "Error saving settings to "+filename+": "+repr(e),
                        traceback.format_exc())

    def setParameters(self, p):
        self.spinBoxSpectra.setValue(0)
        self.fileLoader.loadParameters(p)
        self.comboBoxMethod.setCurrentIndex(p.plotMethod)
        self.comboBoxCmaps.setCurrentText(p.plotColors)
        self.lineEditWavenumber.setValue(p.plotWavenum)
        self.checkBoxAutopick.setChecked(p.spectraAuto)
        self.checkBoxAC.setChecked(p.acDo)
        self.checkBoxSpline.setChecked(p.acSpline)
        self.checkBoxLocalPeak.setChecked(p.acLocal)
        self.checkBoxSmoothCorrected.setChecked(p.acSmooth)
        self.lineEditACReference.setText(p.acReference)
        self.checkBoxSC.setChecked(p.scDo)
        self.comboBoxReference.setCurrentText(p.scRef)
        self.lineEditReferenceName.setText(p.scOtherRef)
        self.spinBoxNIteration.setValue(p.scIters)
        self.spinBoxNclusScat.setValue(p.scClusters)
        self.checkBoxClusters.setChecked(p.scClustering)
        self.checkBoxStabilize.setChecked(p.scStable)
        self.dialogSCAdvanced.comboBoxSCAlgo.setCurrentText(p.scAlgorithm)
        self.dialogSCAdvanced.spinBoxSCResolution.setValue(p.scResolution)
        self.dialogSCAdvanced.lineEditSCamin.setValue(p.scAmin)
        self.dialogSCAdvanced.lineEditSCamax.setValue(p.scAmax)
        self.dialogSCAdvanced.lineEditSCdmin.setValue(p.scDmin)
        self.dialogSCAdvanced.lineEditSCdmax.setValue(p.scDmax)
        self.dialogSCAdvanced.checkBoxSCLinear.setChecked(p.scLinear)
        self.dialogSCAdvanced.checkBoxSCRenormalize.setChecked(p.scRenormalize)
        self.dialogSCAdvanced.radioButtonSCPCADynamic.setChecked(p.scPCADynamic)
        self.dialogSCAdvanced.spinBoxSCPCA.setValue(p.scPCA)
        self.dialogSCAdvanced.spinBoxSCPCAMax.setValue(p.scPCAMax)
        self.dialogSCAdvanced.lineEditSCPCAVariance.setValue(p.scPCAVariance)
        self.dialogSCAdvanced.checkBoxSCAutoIters.setChecked(p.scAutoIters)
        self.dialogSCAdvanced.lineEditSCMinImprov.setValue(p.scMinImprov)
        self.checkBoxSGF.setChecked(p.sgfDo)
        self.spinBoxWindowLength.setValue(p.sgfWindow)
        self.spinBoxPolyOrder.setValue(p.sgfOrder)
        self.checkBoxSR.setChecked(p.srDo)
        self.lineEditMinwn.setValue(p.srMin)
        self.lineEditMaxwn.setValue(p.srMax)
        self.comboBoxBaseline.setCurrentIndex(self.bcIndex(p.bcMethod))
        self.spinBoxItersBC.setValue(p.bcIters)
        self.lineEditLambda.setValue(p.bcLambda)
        self.lineEditP.setValue(p.bcP)
        self.comboBoxNormMethod.setCurrentIndex(self.normIndex(p.normMethod))
        self.lineEditNormWavenum.setValue(p.normWavenum)

        self.wavenumberEdit()
        self.srMinEdit()
        self.srMaxEdit()

        self.updateSGF()
        self.spinBoxSpectra.setValue(p.spectraCount)

    def loadParameters(self):
        filename = self.getLoadFileName("Load preprocessing settings",
                                        filter="Settings files (*.pjs);;All files (*)",
                                        settingname='settingsDir')
        if filename:
            p = PrepParameters()
            try:
                p.load(filename)
                self.setParameters(p)
            except Exception as e:
                self.showDetailedErrorMessage("Error loading settings from "+filename+": "+repr(e),
                                              traceback.format_exc())


    def runBatch(self):
        assert not self.rmiescRunning
        if len(self.data.filenames) < 1:
            self.errorMsg.showMessage('Load some data first')
            return
        params = self.getParameters()

        foldername = self.getDirectoryName("Select save directory",
                                           settingname='spectraDir',
                                           savesetting=False)
        if not foldername:
            return
        preservepath = False

        print('foldername', foldername)
        all_paths = { os.path.dirname(f) for f in self.data.filenames }
        folder_within_input = any(p.startswith(foldername) for p in all_paths)

        if folder_within_input:
            if params.saveExt == '':
                self.errorMsg.showMessage('A filename save suffix must be specified '+
                                          'when loading and saving files in the same directory')
                return
            elif len(self.data.filenames) > 1:
                yn = QMessageBox.question(self, 'Identical directory',
                                          'Are you sure you want to '+
                                          'save output files in the input directory')
                if yn != QMessageBox.Yes:
                    return
            preservepath = True
        elif len(all_paths) > 1:
            yn = QMessageBox.question(self, 'Multiple directories',
                      'Input files are in multiple directories. Should this directory structure '+
                      'be replicated in the output directory?')
            preservepath = yn == QMessageBox.Yes

        self.toggleRunning(2)
        self.scNewSettings = self.getSCSettings()
        self.startBatch.emit(self.data, params, foldername, preservepath)

    @pyqtSlot(int, int)
    def batchProgress(self, a, b):
        self.progressBarRun.setValue(a)
        self.progressBarRun.setMaximum(b)

    @pyqtSlot(str)
    def batchDone(self, err):
        self.worker.halt = False
        self.progressBarSC.setValue(0)
        self.progressBarRun.setValue(0)
        if err != '':
            self.errorMsg.showMessage(err)
        self.toggleRunning(0)


def main():
    parser = argparse.ArgumentParser(
            description='Graphical application for preprocessing of hyperspectral data.')
    parser.add_argument('files', metavar='file', type=str, nargs='*',
                        help='initial hyperspectral images to load')
    run_octavvs_application('Preprocessing', windowclass=MyMainWindow,
                            parser=parser, parameters=['files'])


