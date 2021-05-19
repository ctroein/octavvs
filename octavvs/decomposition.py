import os
import traceback
from pkg_resources import resource_filename
import argparse

from PyQt5.QtWidgets import QDialog, QMessageBox
from PyQt5.QtCore import QThread, pyqtSignal, pyqtSlot
from PyQt5 import uic

import numpy as np
import scipy.signal, scipy.io
import matplotlib
matplotlib.use('QT5Agg')
import matplotlib.pyplot as plt

from .decomp.decompworker import DecompWorker
# import octavvs.io
from octavvs.algorithms import normalization
from octavvs.io import SpectralData, Parameters
from octavvs.ui import (FileLoader, ImageVisualizer, OctavvsMainWindow,
                        NoRepeatStyle, uitools)



DecompositionMainWindow = uic.loadUiType(resource_filename(
    __name__, "decomp/decomposition.ui"))[0]

class MyMainWindow(FileLoader, ImageVisualizer, OctavvsMainWindow,
                   DecompositionMainWindow):

    closing = pyqtSignal()
    startDecomp = pyqtSignal(SpectralData, Parameters)
    startBatch = pyqtSignal(SpectralData, Parameters, str, bool)

    @classmethod
    def program_name(cls):
        "Return the name of the program that this main window represents"
        return 'Decomposition'

    def __init__(self, parent=None, files=None, paramFile=None, savePath=None):
        super().__init__(parent)

        self.data = SpectralData()
        self.workerRunning = False

        # Avoid repeating spinboxes
        self.spinBoxComponents.setStyle(NoRepeatStyle())
        self.spinBoxIterations.setStyle(NoRepeatStyle())

        self.fileLoader.setSuffix('_dec')

        self.plot_spectra.updated.connect(self.updateDC)
        # self.plot_MC.clicked.connect(self.plot_MC.popOut)
        # self.checkBoxMC.toggled.connect(self.updateMC)
        # self.spinBoxMCEndpoints.valueChanged.connect(self.updateMC)
        # self.lineEditMCSlopefactor.editingFinished.connect(self.updateMC)

        # self.pushButtonSaveParameters.clicked.connect(self.saveParameters)
        # self.pushButtonLoadParameters.clicked.connect(self.loadParameters)

        # self.pushButtonRun.clicked.connect(self.runBatch)

        # Defaults when no data loaded
        self.scSettings = {}

        self.worker = DecompWorker()
        self.workerThread = QThread()
        self.worker.moveToThread(self.workerThread)
        self.closing.connect(self.workerThread.quit)
        self.startDecomp.connect(self.worker.decompose)
        self.worker.done.connect(self.dcDone)
        self.worker.stopped.connect(self.dcStopped)
        self.worker.failed.connect(self.dcFailed)
        self.worker.progress.connect(self.dcProgress)
        self.worker.batchProgress.connect(self.batchProgress)
        # self.worker.progressPlot.connect(self.plot_SC.progressPlot)
        self.worker.fileLoaded.connect(self.updateFile)
        self.worker.loadFailed.connect(self.showLoadErrorMessage)
        self.startBatch.connect(self.worker.startBatch)
        self.worker.batchDone.connect(self.batchDone)
        self.workerThread.start()

        self.lineEditSimplismaNoise.setFormat("%g")
        self.lineEditSimplismaNoise.setRange(1e-6, 1)
        self.lineEditTolerance.setFormat("%g")
        self.lineEditTolerance.setRange(1e-6, 1)

        self.updateWavenumberRange()

        self.post_setup()
        if files is not None and files != []:
            self.updateFileList(files, False) # Load files passed as arguments

        if paramFile is not None: # Loads the parameter file passed as argument
            self.loadParameters(filename=paramFile)

        if savePath is not None:
            if paramFile is None:
                self.errorMsg.showMessage(
                    "Running from the command line without passing a "+
                    "parameter file does nothing.")
            savePath = os.path.normpath(savePath)
            self.runBatch(foldername=savePath)

    def closeEvent(self, event):
        self.worker.halt = True
        self.closing.emit()
        plt.close('all')
        self.workerThread.wait()
        self.deleteLater()


    def loadFolder(self, *argv, **kwargs):
        assert self.workerRunning == 0
        super().loadFolder(*argv, **kwargs)

    @pyqtSlot(int)
    def updateFile(self, num):
        super().updateFile(num)
        self.updateWavenumberRange()
        super().updatedFile()


    @pyqtSlot(str, str, str)
    def showLoadErrorMessage(self, file, err, details):
        "Error message from loading a file in the worker thread, with abort option"
        q = self.loadErrorBox(file, (err, details if details else None))
        q.addButton('Skip file', QMessageBox.AcceptRole)
        q.addButton('Abort', QMessageBox.AcceptRole)
        if q.exec():
            self.dcStop()


    # DC, Decomposition
    def updateDC(self):
        "Respond to changes by user in UI"
        if self.workerRunning:
            return
        self.pushButtonStart.setEnabled(True)

    def toggleRunning(self, newstate):
        "Move between states: 0=idle, 1=run one, 2=run batch"
        onoff = not newstate
        self.fileLoader.setEnabled(onoff)
        self.pushButtonRun.setEnabled(onoff)
        self.pushButtonStart.setEnabled(onoff)
        if newstate:
            self.progressBarIteration.setValue(0)
        self.pushButtonStop.setEnabled(newstate == 1)
        self.pushButtonRunStop.setEnabled(newstate == 2)
        self.rmiescRunning = newstate

    def getDCSettings(self):
        "Get relevant parameters for decomposition, as a dict"
        params = self.getParameters()
        return params.filtered('dc')

    def startDC(self):
        "Decompose the current data only"
        if self.workerRunning:
            return
        settings = self.getDCSettings()
        self.progressBarIteration.setMaximum(settings['dcIterations'])
        self.progressBarIteration.setFormat('initializing')
        self.toggleRunning(1)
        self.startDecomp.emit(self.data, Parameters(settings.items()))

    def clearDC(self):
        if self.rmiescRunning == 1:
            return
        self.progressBarIteration.setValue(0)
        self.progressBarIteration.setFormat('idle')

    def dcStop(self):
        self.worker.halt = True
        self.pushButtonStop.setEnabled(False)
        self.pushButtonRunStop.setEnabled(False)

    @pyqtSlot()
    def dcStopped(self):
        self.worker.halt = False
        if self.workterRunning == 1:
            self.toggleRunning(0)
        self.updateDC()

    @pyqtSlot(str, str)
    def dcFailed(self, err, trace):
        self.errorMsg.showMessage(
            'Processing failed:<pre>\n' + err + "\n\n" + trace + '</pre>')
        self.dcStopped()

    @pyqtSlot(np.ndarray, np.ndarray, np.ndarray)
    def dcDone(self, wavenumber, indata, corr):
        self.plot_SC.setData(wavenumber, indata, corr,
                             self.scSettings['selected'],
                             self.plot_visual.getSelected())
        self.dcStopped()

    @pyqtSlot(int, float)
    def dcProgress(self, it, err):
        if it == 0:
            self.progressBarIteration.setFormat('%v / %m')
        self.progressBarIteration.setValue(it)
        self.lineEditRelError.setValue(err)



    # Loading and saving parameters
    def getParameters(self):
        "Copy from UI to some kind of parameters object"
        p = Parameters()
        self.fileLoader.saveParameters(p)
        ImageVisualizer.getParameters(self, p)

        p.dcDo = self.checkBoxDecomp.isChecked()
        p.dcAlgorithm = self.comboBoxAlgorithm.currentText()
        p.dcComponents = self.spinBoxComponents.value()
        p.dcStartingPoint = self.comboBoxStartingPoint.currentText()
        p.dcInitialValues = self.comboBoxInitialValues.currentText()
        p.dcSimplismaNoise = self.lineEditSimplismaNoise.value()
        p.dcIterations = self.spinBoxIterations.value()
        p.dcTolerance = self.lineEditTolerance.value()
        return p

    def saveParameters(self):
        filename = self.getSaveFileName(
                "Save decomposition settings",
                filter="Setting files (*.djs);;All files (*)",
                settingname='settingsDir',
                suffix='djs')
        if filename:
            try:
                self.getParameters().save(filename)
            except Exception as e:
                self.showDetailedErrorMessage(
                        "Error saving settings to "+filename+": "+repr(e),
                        traceback.format_exc())

    def setParameters(self, p):
        "Copy from some kind of parameters object to UI"
        self.spinBoxSpectra.setValue(0)
        self.fileLoader.loadParameters(p)

        self.checkBoxDecomp.setChecked(p.dcDo)
        self.comboBoxAlgorithm.setCurrentText(p.dcAlgorithm)
        self.spinBoxComponents.setValue(p.dcComponents)
        self.comboBoxStartingPoint.setCurrentText(p.dcStartingPoint)
        self.comboBoxInitialValues.setCurrentText(p.dcInitialValues)
        self.lineEditSimplismaNoise.setValue(p.dcSimplismaNoise)
        self.spinBoxIterations.setValue(p.dcIterations)
        self.lineEditTolerance.setValue(p.dcTolerance)

        ImageVisualizer.setParameters(self, p)

    def loadParameters(self, checked=False, filename=None):
        """
        Load parameters from a file, showing a dialog if no filename is given.
        The 'checked' parameter is just a placeholder to match QPushButton.clicked().
        """
        if not filename:
            filename = self.getLoadFileName(
                    "Load decomposition settings",
                    filter="Settings files (*.djs);;All files (*)",
                    settingname='settingsDir')
        if filename:
            p = self.getParameters()
            try:
                p.load(filename)
                self.setParameters(p)
            except Exception as e:
                self.showDetailedErrorMessage(
                        "Error loading settings from "+filename+": "+repr(e),
                        traceback.format_exc())


    def runBatch(self, checked=False, foldername=None):
        """
        Run the selected preprocessing steps, showing a dialog if no output
        path is given. The 'checked' parameter is just a placeholder to
        match QPushButton.clicked().
        """
        assert not self.workerRunning
        if len(self.data.filenames) < 1:
            self.errorMsg.showMessage('Load some data first')
            return
        params = self.getParameters()

        if foldername is None:
            foldername = self.getDirectoryName("Select save directory",
                                               settingname='spectraDir',
                                               savesetting=False)
        if not foldername:
            return
        preservepath = False

        all_paths = { os.path.dirname(f) for f in self.data.filenames }
        folder_within_input = any(p.startswith(foldername) for p in all_paths)

        if folder_within_input:
            if params.saveExt == '':
                self.errorMsg.showMessage(
                    'A filename save suffix must be specified '+
                    'when loading and saving files in the same directory')
                return
            elif len(self.data.filenames) > 1:
                yn = QMessageBox.question(
                    self, 'Identical directory',
                    'Are you sure you want to '+
                    'save output files in the input directory')
                if yn != QMessageBox.Yes:
                    return
            preservepath = True
        elif len(all_paths) > 1:
            yn = QMessageBox.question(
                self, 'Multiple directories',
                'Input files are in multiple directories. Should this '+
                'directory structure be replicated in the output directory?')
            preservepath = yn == QMessageBox.Yes

        self.toggleRunning(2)
        self.startBatch.emit(self.data, params, foldername, preservepath)


    @pyqtSlot(int, int)
    def batchProgress(self, a, b):
        self.progressBarRun.setValue(a)
        self.progressBarRun.setMaximum(b)

    @pyqtSlot(bool)
    def batchDone(self, success):
        self.worker.halt = False
        self.progressBarIteration.setValue(0)
        self.progressBarIteration.setFormat('done' if success else 'failed')
        # if err != '':
        #     self.errorMsg.showMessage(err)
        self.toggleRunning(0)


def main():
    parser = argparse.ArgumentParser(
            description='Graphical application for decomposition of '+
            'hyperspectral data.')
    parser.add_argument('files', metavar='file', nargs='*',
                        help='initial hyperspectral images to load')
    parser.add_argument('-p', '--params', metavar='file.djs', dest='paramFile',
                        help='parameter file to load')
    parser.add_argument('-r', '--run', metavar='output_dir', nargs='?',
                        dest='savePath', const='./',
                        help='runs and saves to the output directory (params '+
                        'argument must also be passed)')
    MyMainWindow.run_octavvs_application(
        parser=parser, parameters=['files', 'paramFile', 'savePath'])


