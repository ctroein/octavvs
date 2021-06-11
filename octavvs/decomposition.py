import os
import traceback
from pkg_resources import resource_filename
import argparse

from PyQt5.QtWidgets import QDialog, QMessageBox, QStyle
from PyQt5.QtCore import Qt, QThread, pyqtSignal, pyqtSlot
from PyQt5 import uic

import numpy as np
import scipy.signal
import sklearn.cluster
import matplotlib
matplotlib.use('QT5Agg')
import matplotlib.pyplot as plt

from .decomp.decompworker import DecompWorker
# import octavvs.io
# from octavvs.algorithms import normalization
from octavvs.io import DecompositionData, Parameters
from octavvs.ui import (FileLoader, ImageVisualizer, OctavvsMainWindow,
                        NoRepeatStyle, uitools)



DecompositionMainWindow = uic.loadUiType(resource_filename(
    __name__, "decomp/decomposition.ui"))[0]

class MyMainWindow(ImageVisualizer, FileLoader, OctavvsMainWindow,
                   DecompositionMainWindow):

    closing = pyqtSignal()
    startDecomp = pyqtSignal(DecompositionData, Parameters)
    startBatch = pyqtSignal(DecompositionData, Parameters, str, bool)

    @classmethod
    def program_name(cls):
        "Return the name of the program that this main window represents"
        return 'Decomposition'

    def __init__(self, parent=None, files=None, paramFile=None, savePath=None):
        super().__init__(parent)

        self.data = DecompositionData()
        self.workerRunning = False

        # Avoid repeating spinboxes
        self.spinBoxComponents.setStyle(NoRepeatStyle())
        self.spinBoxIterations.setStyle(NoRepeatStyle())

        self.comboBoxDirectory.currentIndexChanged.connect(
            self.dirModeCheck)
        self.pushButtonDirectory.clicked.connect(self.dirSelect)

        self.imageVisualizer.comboBoxCmaps.currentTextChanged.connect(
            self.plot_roi.set_cmap)
        self.imageVisualizer.plot_raw.updatedProjection.connect(
            self.plot_roi.set_data)
        self.pushButtonRoiClear.clicked.connect(self.roiClear)
        self.pushButtonRoiAdd.clicked.connect(self.roiAddArea)
        self.pushButtonRoiRemove.clicked.connect(self.roiRemoveArea)
        self.pushButtonRoiErase.clicked.connect(
            self.plot_roi.erase_last_point)
        self.pushButtonRoiInvert.clicked.connect(self.plot_roi.invert_roi)
        self.plot_roi.updated.connect(self.roiUpdateSelected)
        self.pushButtonRoiLoad.clicked.connect(self.roiLoad)
        self.pushButtonRoiSave.clicked.connect(self.roiSave)

        self.imageVisualizer.plot_spectra.updated.connect(self.updateDC)
        self.pushButtonStart.clicked.connect(self.startDC)
        self.pushButtonStop.clicked.connect(self.stopDC)

        self.imageVisualizer.comboBoxCmaps.currentTextChanged.connect(
            self.plot_decomp.set_cmap)
        self.comboBoxPlotMode.currentIndexChanged.connect(
            self.plot_decomp.set_display_mode)
        self.plot_decomp.displayModesUpdated.connect(
            self.updateDCPlotModes)

        self.pushButtonCluster.clicked.connect(self.clusterDC)

        # self.toolButtonAnnotationPlus.setIcon(
        #     self.style().standardIcon(QStyle.SP_DialogYesButton))
        self.toolButtonAnnotationMinus.setIcon(
            self.style().standardIcon(QStyle.SP_TrashIcon))
        def dragMove(event):
            if event.target == self.listWidgetClusterAnnotations:
                event.accept()
            else:
                event.ignore()
        self.listWidgetClusterAnnotations.dragMoveEvent = dragMove

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
        self.worker.progressPlot.connect(self.plot_decomp.plot_progress)
        self.worker.batchProgress.connect(self.batchProgress)
        self.worker.fileLoaded.connect(self.updateFile)
        self.worker.loadFailed.connect(self.showLoadErrorMessage)
        self.startBatch.connect(self.worker.startBatch)
        self.worker.batchDone.connect(self.batchDone)
        self.workerThread.start()

        self.lineEditSimplismaNoise.setFormat("%g")
        self.lineEditSimplismaNoise.setRange(1e-6, 1)
        self.lineEditTolerance.setFormat("%g")
        self.lineEditTolerance.setRange(1e-10, 1)
        self.lineEditRelError.setFormat("%.4g")

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
        self.roiAutosaveCheck()
        self.worker.halt = True
        self.closing.emit()
        plt.close('all')
        self.workerThread.wait()
        self.deleteLater()


    def loadFolder(self, *argv, **kwargs):
        assert self.workerRunning == 0
        super().loadFolder(*argv, **kwargs)

    def updateFileList(self, *argv, **kwargs):
        ret = super().updateFileList(*argv, **kwargs)
        if ret:
            self.roiDirModeCheck()
        return ret

    def loadFile(self, file):
        "Save ROI before proceeding"
        self.roiAutosaveCheck()
        return super().loadFile(file)

    @pyqtSlot(int)
    def updateFile(self, num):
        super().updateFile(num)
        self.updateWavenumberRange()
        self.plot_roi.set_geometry(
            wh=self.data.wh, pixelxy=self.data.pixelxy)
        self.roiLoad(auto=True)
        self.plot_decomp.set_geometry(
            wh=self.data.wh, pixelxy=self.data.pixelxy)
        self.plot_decomp.set_wavenumbers(
            self.imageVisualizer.plot_raw.getWavenumbers())
        self.updatedFile()

    @pyqtSlot(str, str, str, bool)
    def showLoadErrorMessage(self, file, err, details, warning):
        """
        Error message from loading a file in the worker thread,
        with abort option
        """
        q = self.loadErrorBox(file, (err, details if details else None),
                              warning)
        q.addButton('Ignore' if warning else 'Skip file',
                    QMessageBox.AcceptRole)
        abort = q.addButton('Abort', QMessageBox.AcceptRole)
        q.exec()
        if q.clickedButton() == abort:
            self.dcStop()

    def updateDimensions(self, wh):
        super().updateDimensions(wh)
        self.plot_roi.set_geometry(
            wh=self.data.wh, pixelxy=self.data.pixelxy)
        self.plot_decomp.set_geometry(
            wh=self.data.wh, pixelxy=self.data.pixelxy)

    def setPlotColors(self, cmap):
        super().setPlotColors(cmap)
        self.plot_decomp.draw_idle()

    # Common output directory handling
    def dirCurrent(self):
        if self.comboBoxDirectory.currentIndex():
            return self.lineEditDirectory.text()
        return None

    def dirModeCheck(self):
        other = self.comboBoxDirectory.currentIndex()
        self.lineEditDirectory.setEnabled(other)
        if other:
            dups = self.data.get_duplicate_filenames()
            if dups:
                q = QMessageBox(self)
                q.setIcon(QMessageBox.Warning)
                q.setWindowTitle('Warning: identical filenames')
                q.setText('Some input files have identical names so their '
                          'regions of interest will attempt to use '
                          'the same files.')
                q.setTextFormat(Qt.PlainText)
                q.setDetailedText('Examples:\n' + '\n'.join(list(dups)[:10]))
                q.exec()
            if not self.lineEditDirectory.text():
                self.dirSelect()

    def dirSelect(self):
        rdir = self.getDirectoryName(
            "Select ROI directory", settingname='roiDir')
        if rdir is not None:
            self.lineEditDirectory.setText(rdir)
            self.comboBoxDirectory.setCurrentIndex(1)


    # Roi, Region of interest
    def roiClear(self):
        self.pushButtonRoiAdd.setChecked(False)
        self.pushButtonRoiRemove.setChecked(False)
        self.plot_roi.set_draw_mode('click')
        self.plot_roi.clear()

    def roiAddArea(self, checked):
        self.pushButtonRoiRemove.setChecked(False)
        self.plot_roi.set_draw_mode('add' if checked else 'click')

    def roiRemoveArea(self, checked):
        self.pushButtonRoiAdd.setChecked(False)
        self.plot_roi.set_draw_mode('remove' if checked else 'click')

    def roiUpdateSelected(self, n, m, from_draw):
        self.labelRoiSelected.setText('Selected: %d / %d' % (n, m))
        if from_draw:
            self.labelRoiChanged.setText('*')


    def roiLoad(self, auto=False):
        if not self.data.curFile:
            return
        filename = self.data.roi_filename(filedir=self.dirCurrent())
        if auto:
            self.data.roi = None
            try:
                self.data.load_roi(filename)
            except FileNotFoundError:
                pass
            except Exception as e:
                print('Warning:', e)
        else:
            path = os.path.split(filename)
            filename = self.getLoadFileName(
                "Load ROI from file",
                filter="Decomposition HDF5 files (*.odd);;All files (*)",
                settingname='roiDir',
                directory=path[0], defaultfilename=path[1])
            if not filename:
                return
            self.data.load_roi(filename)
        self.plot_roi.set_roi(self.data.roi)

    def roiSave(self, auto=False):
        if not self.data.curFile:
            return
        changed = self.data.set_roi(self.plot_roi.get_roi())
        if auto and not changed:
            return
        filename = self.data.roi_filename(filedir=self.dirCurrent())
        if not auto:
            path = os.path.split(filename)
            filename = self.getSaveFileName(
                "Save ROI to file",
                filter="Decomposition HDF5 files (*.odd);;All files (*)",
                settingname='roiDir',
                directory=path[0], defaultfilename=path[1])
            if not filename:
                return
        self.data.save_roi(filename)

    def roiAutosaveCheck(self):
        if self.checkBoxRoiAutosave.isChecked() and self.data.curFile:
            self.roiSave(auto=True)


    # DC, Decomposition
    def updateDC(self):
        "Respond to changes by user in UI"
        if self.workerRunning:
            return
        self.pushButtonStart.setEnabled(True)

    @pyqtSlot(list)
    def updateDCPlotModes(self, modes):
        "Refresh the list of plot modes"
        # Quick and dirty
        ct = self.comboBoxPlotMode.currentText()
        self.comboBoxPlotMode.clear()
        self.comboBoxPlotMode.insertItems(0, modes)
        try:
            self.comboBoxPlotMode.setCurrentText(ct)
        except ValueError:
            pass

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
        self.workerRunning = newstate

    def getDCSettings(self):
        "Get relevant parameters for decomposition, as a dict"
        params = self.getParameters()
        return params.filtered('dc')

    def startDC(self):
        "Decompose the current data only"
        if self.workerRunning:
            return
        settings = self.getDCSettings()
        self.data.set_roi(self.plot_roi.get_roi())
        self.plot_decomp.set_roi(self.data.roi)
        self.progressBarIteration.setMaximum(settings['dcIterations'])
        self.progressBarIteration.setFormat('initializing')
        self.toggleRunning(1)
        self.startDecomp.emit(self.data, Parameters(settings.items()))

    def clearDC(self):
        if self.workerRunning == 1:
            return
        self.progressBarIteration.setValue(0)
        self.progressBarIteration.setFormat('idle')

    def stopDC(self):
        self.worker.halt = True
        self.pushButtonStop.setEnabled(False)
        self.pushButtonRunStop.setEnabled(False)

    @pyqtSlot()
    def dcStopped(self):
        self.worker.halt = False
        if self.workerRunning == 1:
            self.toggleRunning(0)
        self.updateDC()

    @pyqtSlot(str, str)
    def dcFailed(self, err, trace):
        self.errorMsg.showMessage(
            'Processing failed:<pre>\n' + err + "\n\n" + trace + '</pre>')
        self.dcStopped()

    @pyqtSlot(np.ndarray, np.ndarray)
    def dcDone(self, concentrations, spectra):
        self.plot_decomp.set_spectra(spectra)
        self.plot_decomp.set_concentrations(concentrations)
        self.dcStopped()

    @pyqtSlot(int, float)
    def dcProgress(self, it, err):
        if it == 0:
            self.progressBarIteration.setFormat('%v / %m')
            self.plot_decomp.clear_errors()
        else:
            self.progressBarIteration.setValue(it)
            self.lineEditRelError.setValue(err)
            self.plot_decomp.add_error(err)

    def clusterDC(self):
        data = self.plot_decomp.get_concentrations().T
        data = data / data.mean(axis=0)
        kmeans = sklearn.cluster.MiniBatchKMeans(
            self.spinBoxClusters.value())
        labels = kmeans.fit_predict(data)
        self.plot_decomp.add_clustering('K-means', labels)
        self.comboBoxPlotMode.setCurrentText('K-means')
        self.plot_decomp.draw_idle()

    # Loading and saving parameters
    def getParameters(self):
        "Copy from UI to some kind of parameters object"
        p = Parameters()
        self.fileLoader.saveParameters(p)
        self.imageVisualizer.saveParameters(p)

        p.dirMode = self.comboBoxDirectory.currentIndex()
        p.directory = self.lineEditDirectory.text()

        p.roiAutosave = self.checkBoxRoiAutosave.isChecked()

        p.dcAlgorithm = self.comboBoxAlgorithm.currentText()
        p.dcComponents = self.spinBoxComponents.value()
        p.dcStartingPoint = self.comboBoxStartingPoint.currentText()
        p.dcRoi = self.comboBoxUseRoi.currentIndex()
        p.dcInitialValues = self.comboBoxInitialValues.currentText()
        p.dcSimplismaNoise = self.lineEditSimplismaNoise.value()
        p.dcIterations = self.spinBoxIterations.value()
        p.dcTolerance = self.lineEditTolerance.value()
        p.dcClusters = self.spinBoxClusters.value()
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
        self.imageVisualizer.loadParameters(p)

        self.comboBoxDirectory.setCurrentIndex(p.dirMode)
        self.lineEditDirectory.setText(p.directory)

        self.checkBoxRoiAutosave.setChecked(p.roiAutosave)

        self.comboBoxAlgorithm.setCurrentText(p.dcAlgorithm)
        self.spinBoxComponents.setValue(p.dcComponents)
        self.comboBoxStartingPoint.setCurrentText(p.dcStartingPoint)
        self.comboBoxUseRoi.setCurrentIndex(p.dcRoi)
        self.comboBoxInitialValues.setCurrentText(p.dcInitialValues)
        self.lineEditSimplismaNoise.setValue(p.dcSimplismaNoise)
        self.spinBoxIterations.setValue(p.dcIterations)
        self.lineEditTolerance.setValue(p.dcTolerance)
        self.spinBoxClusters.setValue(p.dcClusters)


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


