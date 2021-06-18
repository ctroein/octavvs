import os
import traceback
from pkg_resources import resource_filename
import argparse

from PyQt5.QtWidgets import QMessageBox, QStyle, QTableWidget, \
    QTableWidgetItem, QMainWindow, QDialog, QDialogButtonBox, \
    QListWidgetItem
from PyQt5 import QtGui
from PyQt5.QtCore import Qt, QThread, pyqtSignal, pyqtSlot
from PyQt5 import uic

import numpy as np
# import scipy.signal
import sklearn.cluster
import matplotlib
matplotlib.use('QT5Agg')
import matplotlib.pyplot as plt

from .decomp.decompworker import DecompWorker
# import octavvs.io
# from octavvs.algorithms import normalization
from octavvs.io import DecompositionData, Parameters
from octavvs.ui import (FileLoader, ImageVisualizer, OctavvsMainWindow,
                        NoRepeatStyle)



DecompositionMainWindow = uic.loadUiType(resource_filename(
    __name__, "decomp/decomposition.ui"))[0]
SettingsTableWindow = uic.loadUiType(resource_filename(
    __name__, "decomp/settings_table.ui"))[0]

class DialogSettingsTable(QDialog, SettingsTableWindow):
    def __init__(self, parent=None):
        QDialog.__init__(self, parent)
        self.setupUi(self)

class MyMainWindow(ImageVisualizer, FileLoader, OctavvsMainWindow,
                   DecompositionMainWindow):

    closing = pyqtSignal()
    startDecomp = pyqtSignal(DecompositionData, Parameters)
    startBatch = pyqtSignal(DecompositionData, Parameters)

    @classmethod
    def program_name(cls):
        "Return the name of the program that this main window represents"
        return 'Decomposition'

    def __init__(self, parent=None, files=None, paramFile=None):
        super().__init__(parent)

        self.data = DecompositionData()
        self.workerRunning = False

        # Avoid repeating spinboxes
        self.spinBoxComponents.setStyle(NoRepeatStyle())
        self.spinBoxIterations.setStyle(NoRepeatStyle())

        self.comboBoxDirectory.currentIndexChanged.connect(
            self.dirModeCheck)
        self.pushButtonDirectory.clicked.connect(self.dirSelect)
        self.lineEditDirectory.editingFinished.connect(self.dirEditCheck)
        self.pushButtonSaveParameters.clicked.connect(self.saveParameters)
        self.pushButtonLoadParameters.clicked.connect(self.loadParameters)

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
        self.pushButtonLoad.clicked.connect(self.dcLoad)
        self.pushButtonSave.clicked.connect(self.dcSave)
        self.pushButtonRun.clicked.connect(self.dcBatchRun)
        self.pushButtonRunStop.clicked.connect(self.stopDC)

        self.pushButtonSettingsInfo.clicked.connect(self.dcSettingsShow)
        self.dialogSettingsTable = DialogSettingsTable()
        self.dialogSettingsTable.buttonBox.clicked.connect(
            self.dcSettingsClicked)

        self.imageVisualizer.comboBoxCmaps.currentTextChanged.connect(
            self.plot_decomp.set_cmap)
        self.comboBoxPlotMode.currentIndexChanged.connect(
            self.plot_decomp.set_display_mode)
        self.plot_decomp.displayModesUpdated.connect(
            self.updateDCPlotModes)

        self.pushButtonCluster.clicked.connect(self.caStartClustering)
        # self.toolButtonAnnotationPlus.setIcon(
        #     self.style().standardIcon(QStyle.SP_DialogYesButton))
        self.toolButtonAnnotationMinus.setIcon(
            self.style().standardIcon(QStyle.SP_TrashIcon))
        self.toolButtonAnnotationPlus.clicked.connect(self.caAddAnnotation)
        self.toolButtonAnnotationMinus.clicked.connect(self.caDelAnnotation)
        self.listWidgetClusterUsed.itemDoubleClicked.connect(
            self.caClickedUsed)
        self.listWidgetClusterUnused.itemDoubleClicked.connect(
            self.caClickedUnused)
        self.plot_cluster.clicked.connect(self.caSelectCluster)
        self.pushButtonClusterLoad.clicked.connect(self.caLoad)
        self.pushButtonClusterSave.clicked.connect(self.caSave)

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
        self.worker.batchProgress.connect(self.dcBatchProgress)
        self.worker.fileLoaded.connect(self.updateFile)
        self.worker.loadFailed.connect(self.showLoadErrorMessage)
        self.startBatch.connect(self.worker.startBatch)
        self.worker.batchDone.connect(self.dcBatchDone)
        self.workerThread.start()

        self.lineEditSimplismaNoise.setFormat("%g")
        self.lineEditSimplismaNoise.setRange(1e-6, 1)
        self.lineEditTolerance.setFormat("%g")
        self.lineEditTolerance.setRange(1e-10, 1)
        self.lineEditRelError.setFormat("%.4g")

        self.updateWavenumberRange()

        self.post_setup()
        if paramFile is not None: # Loads the parameter file passed as argument
            self.loadParameters(filename=paramFile)
        if files is not None and files != []:
            self.updateFileList(files, False) # Load files passed as arguments


    def closeEvent(self, event):
        self.dialogSettingsTable.close()
        self.autosaveCheck()
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
            self.dirModeCheck()
        return ret

    def loadFile(self, file):
        "Save ROI before proceeding"
        self.autosaveCheck()
        return super().loadFile(file)

    @pyqtSlot(int)
    def updateFile(self, num):
        super().updateFile(num)
        self.updateWavenumberRange()
        self.plot_roi.set_basic_data(
            wh=self.data.wh, pixelxy=self.data.pixelxy)
        self.plot_decomp.set_basic_data(
            wn=self.imageVisualizer.plot_raw.getWavenumbers(),
            wh=self.data.wh, pixelxy=self.data.pixelxy)
        self.plot_cluster.set_basic_data(
            wh=self.data.wh, pixelxy=self.data.pixelxy)
        self.genericLoad(auto=True)
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
        self.plot_roi.adjust_geometry(wh)
        self.plot_decomp.adjust_geometry(wh)

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
            "Select ROI directory", settingname='rdcDir')
        if rdir is not None:
            self.lineEditDirectory.setText(rdir)
            self.comboBoxDirectory.setCurrentIndex(1)
            self.dirEditCheck()

    def dirEditCheck(self):
        if not self.lineEditDirectory.text():
            self.comboBoxDirectory.setCurrentIndex(0)
        if self.data.curFile:
            filename = self.data.rdc_filename(filedir=self.dirCurrent())
            if os.path.exists(filename):
                yn = QMessageBox.question(
                    self, 'Directory changed',
                    '(Re)load saved data from selected directory?')
                if yn == QMessageBox.Yes:
                    self.data.load_rdc(filename)
                    self.plot_roi.set_roi(self.data.roi)


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

    def roiUpdateSelected(self, n, m):
        self.labelRoiSelected.setText('Selected: %d / %d' % (n, m))


    def genericLoad(self, auto=False, what='all', description='data'):
        if not self.data.curFile:
            return
        filename = self.data.rdc_filename(filedir=self.dirCurrent())
        if auto:
            try:
                self.data.load_rdc(filename, what=what)
            except OSError:
                return
            except Exception as e:
                print('Warning:', e)
                return
        else:
            path = os.path.split(filename)
            filename = self.getLoadFileName(
                "Load %s from file" % description,
                filter="Decomposition HDF5 files (*.odd);;All files (*)",
                directory=path[0], defaultfilename=path[1])
            if not filename:
                return
            self.data.load_rdc(filename, what=what)
        if what in ['roi', 'all']:
            self.plot_roi.set_roi(self.data.roi)
        if what in ['decomposition', 'all']:
            self.dcSettingsUpdate()
            self.plot_decomp.clear_and_set_roi(self.data.decomposition_roi)
            self.plot_decomp.set_errors(self.data.decomposition_errors)
            dd = self.data.decomposition_data
            if dd is not None and len(dd):
                print('saved iters: ', dd.keys())
                lastiter = max(dd.keys())
                ds = dd[lastiter]
                self.plot_decomp.set_concentrations(ds['concentrations'])
                self.plot_decomp.set_spectra(ds['spectra'])
                if 0 in dd:
                    self.plot_decomp.set_initial_spectra(dd[0]['spectra'])
        if what in ['clustering', 'all']:
            self.plot_cluster.set_roi_and_clusters(
                self.data.clustering_roi, self.data.clustering_labels)
            self.caPopulateLists()

    def genericSave(self, auto=False, what='all', description='data'):
        if not self.data.curFile:
            return

        filename = self.data.rdc_filename(filedir=self.dirCurrent())
        if not auto:
            path = os.path.split(filename)
            filename = self.getSaveFileName(
                "Save %s to file" % description,
                filter="Decomposition HDF5 files (*.odd);;All files (*)",
                directory=path[0], defaultfilename=path[1])
            if not filename:
                return

        if what in ['roi', 'all']:
            self.data.set_roi(self.plot_roi.get_roi())
        if what in ['decomposition', 'all']:
            ...
        if what in ['clustering', 'all']:
            # from list to data...
            ...
        self.data.save_rdc(filename, what=what)


    def roiLoad(self, auto=False):
        self.genericLoad(auto, what='roi', description='ROI')

    def roiSave(self, auto=False):
        self.genericSave(auto, what='roi', description='ROI')


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
        params = Parameters(self.getDCSettings().items())
        self.data.set_roi(self.plot_roi.get_roi())
        if params.dcRoi == 'require' and self.data.roi is None:
            self.errorMsg.showMessage(
                'Non-empty ROI is required')
            return
        self.data.set_decomposition_settings(params)
        self.dcSettingsUpdate()
        self.plot_decomp.clear_and_set_roi(self.data.decomposition_roi)
        self.progressBarIteration.setMaximum(params.dcIterations)
        self.progressBarIteration.setFormat('initializing')
        self.toggleRunning(1)
        self.startDecomp.emit(self.data, params)

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

    @pyqtSlot(np.ndarray, np.ndarray, np.ndarray)
    def dcDone(self, concentrations, spectra, errors):
        self.data.add_decomposition_data(len(errors), spectra, concentrations)
        self.data.set_decomposition_errors(errors)
        self.plot_decomp.set_spectra(spectra)
        self.plot_decomp.set_concentrations(concentrations)
        self.plot_decomp.set_errors(errors)
        self.dcStopped()
        self.caClear()

    @pyqtSlot(list)
    def dcProgress(self, errors):
        if not errors:
            self.progressBarIteration.setFormat('%v / %m')
        else:
            self.progressBarIteration.setValue(len(errors))
            self.lineEditRelError.setValue(errors[-1])
        self.plot_decomp.set_errors(errors)

    # def clusterDC(self):
    #     data = self.plot_decomp.get_concentrations().T
    #     data = data / data.mean(axis=0)
    #     kmeans = sklearn.cluster.MiniBatchKMeans(
    #         self.spinBoxClusters.value())
    #     labels = kmeans.fit_predict(data)
    #     self.plot_decomp.add_clustering('K-means', labels)
    #     self.comboBoxPlotMode.setCurrentText('K-means')
    #     self.plot_decomp.draw_idle()

    def dcLoad(self, auto=False):
        self.genericLoad(auto, what='decomposition',
                         description='decomposition data')

    def dcSave(self, auto=False):
        self.genericSave(auto, what='decomposition',
                         description='decomposition data')

    def dcSettingsUpdate(self):
        "Update the table of most recent dc settings"
        dds = self.data.decomposition_settings
        if dds is None:
            dds = {}
        tw = self.dialogSettingsTable.tableWidget
        tw.setRowCount(len(dds))
        for i, k in enumerate(dds.keys()):
            tw.setItem(i, 0, QTableWidgetItem(k))
            tw.setItem(i, 1, QTableWidgetItem(str(dds[k])))
        tw.resizeColumnsToContents()

    def dcSettingsShow(self, button):
        "Show the table of dc settings"
        self.dialogSettingsTable.show()
        self.dialogSettingsTable.raise_()

    def dcSettingsClicked(self, button):
        if button == QDialogButtonBox.Apply:
            ...
        elif button == QDialogButtonBox.Close:
            self.dialogSettingsTable.hide()

    def dcBatchRun(self, checked=False):
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
        self.toggleRunning(2)
        self.startBatch.emit(self.data, params)


    @pyqtSlot(int, int)
    def dcBatchProgress(self, a, b):
        self.progressBarRun.setValue(a)
        self.progressBarRun.setMaximum(b)

    @pyqtSlot(bool)
    def dcBatchDone(self, success):
        self.worker.halt = False
        self.progressBarIteration.setValue(0)
        self.progressBarIteration.setFormat('done' if success else 'failed')
        self.toggleRunning(0)

    # CA, Clustering and annotation
    def caStartClustering(self):
        if self.data.raw is None:
            return
        p = self.getParameters()
        if p.caInput == 'raw':
            inp = self.data.raw
            roi = self.data.roi
        else:
            dd = self.data.decomposition_data
            lastiter = max(dd.keys())
            inp = dd[lastiter]['concentrations'].T
            roi = self.data.decomposition_roi
        if p.caNormalizeMean:
            inp = inp - inp.mean(0)
        if p.caNormalizeVariance:
            inp = inp / inp.std(0)
        kmeans = sklearn.cluster.MiniBatchKMeans(
            p.caClusters, n_init=p.caRestarts)
        labels = kmeans.fit_predict(inp)
        self.data.set_clustering_settings(p, roi)
        self.data.set_clustering_labels(labels)
        self.plot_cluster.set_roi_and_clusters(
            self.data.clustering_roi, self.data.clustering_labels)
        self.caPopulateLists()

    def caAddAnnotation(self, checked, name='New annotation', edit=True):
        item = QListWidgetItem(name, parent=self.listWidgetClusterAnnotations)
        item.setFlags(item.flags() | Qt.ItemIsEditable)
        self.listWidgetClusterAnnotations.addItem(item)
        if edit:
            self.listWidgetClusterAnnotations.editItem(item)

    def caPopulateLists(self):
        for l in [self.listWidgetClusterUsed, self.listWidgetClusterUnused,
                  self.listWidgetClusterAnnotations]:
            while l.count():
                l.takeItem(0)
        if self.data.clustering_annotations is None:
            return
        used = set()
        for aname, clusters in self.data.clustering_annotations.items():
            self.caAddAnnotation(name=aname, edit=False)
            used.update(clusters)
        if self.data.clustering_labels is not None:
            unused = set(self.data.clustering_labels) - used
            for u in unused:
                item = QListWidgetItem('Cluster %d' % u)
                cc = np.asarray(self.plot_cluster.cluster_color(u)[:3]) * 255
                print(cc)
                color = QtGui.QColor(*cc)
                print('color',color.rgb())
                item.setBackground(QtGui.QBrush(color))
                # item.setForeground(QtGui.QBrush(color))
                self.listWidgetClusterUnused.addItem(item)

    def caDelAnnotation(self):
        sel = self.listWidgetClusterAnnotations.selectedItems()
        if not sel:
            return
        for s in sel:
            self.listWidgetClusterAnnotations.takeItem(
                self.listWidgetClusterAnnotations.row(s))


    def caClickedUsed(self, item):
        print('double c1')
        self.listWidgetClusterUsed.takeItem(
            self.listWidgetClusterUsed.row(item))
        self.listWidgetClusterUnused.addItem(item)

    def caClickedUnused(self, item):
        print('double c2')
        self.listWidgetClusterUnused.takeItem(
            self.listWidgetClusterUnused.row(item))
        self.listWidgetClusterUsed.addItem(item)

    def caSelectCluster(self, num):
        ...

    def caClear(self):
        "A bit brutal as response to changes, maybe"
        input = self.caInputNames[self.comboBoxClusterInput.currentIndex()]
        enabled = self.data.raw is not None and (
            input == 'raw' or self.data.decomposition_data is not None)
        self.pushButtonCluster.setEnabled(enabled)
        self.plot_cluster.clear_clusters()
        for l in [self.listWidgetClusterUsed, self.listWidgetClusterUnused]:
            while l.count():
                l.takeItem(0)

    def caLoad(self, auto=False):
        self.genericLoad(auto, what='clustering', description='Clustering')

    def caSave(self, auto=False):
        self.genericSave(auto, what='clustering', description='Clustering')

    def autosaveCheck(self):
        if self.data.curFile:
            if self.checkBoxRoiAutosave.isChecked():
                self.roiSave(auto=True)
            if self.checkBoxClusterAutosave.isChecked():
                self.caSave(auto=True)

    useRoiNames = ['ignore', 'ifdef', 'require']
    caInputNames = ['raw', 'decomposition']
    caMethodNames = ['kmeans']

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
        p.dcRoi = self.useRoiNames[self.comboBoxUseRoi.currentIndex()]
        p.dcInitialValues = self.comboBoxInitialValues.currentText()
        p.dcSimplismaNoise = self.lineEditSimplismaNoise.value()
        p.dcIterations = self.spinBoxIterations.value()
        p.dcTolerance = self.lineEditTolerance.value()

        p.caInput = self.caInputNames[self.comboBoxClusterInput.currentIndex()]
        p.caNormalizeMean = self.checkBoxClusterNormMean.isChecked()
        p.caNormalizeVariance = self.checkBoxClusterNormVar.isChecked()
        p.caMethod = self.caMethodNames[
            self.comboBoxClusterMethod.currentIndex()]
        p.caClusters = self.spinBoxClusterClusters.value()
        p.caRestarts = self.spinBoxClusterRestarts.value()
        p.caAnnotations = []
        for i in range(self.listWidgetClusterAnnotations.count()):
            item = self.listWidgetClusterAnnotations.item(i)
            p.caAnnotations.append(item.text())
        p.caAutosave = self.checkBoxClusterAutosave.isChecked()

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
        self.comboBoxUseRoi.setCurrentIndex(self.useRoiNames.index(p.dcRoi))
        self.comboBoxInitialValues.setCurrentText(p.dcInitialValues)
        self.lineEditSimplismaNoise.setValue(p.dcSimplismaNoise)
        self.spinBoxIterations.setValue(p.dcIterations)
        self.lineEditTolerance.setValue(p.dcTolerance)

        self.comboBoxClusterInput.setCurrentIndex(
            self.caInputNames.index(p.caInput))
        self.checkBoxClusterNormMean.setChecked(p.caNormalizeMean)
        self.checkBoxClusterNormVar.setChecked(p.caNormalizeVariance)
        self.comboBoxClusterMethod.setCurrentIndex(
            self.caMethodNames.index(p.caMethod))
        self.spinBoxClusterClusters.setValue(p.caClusters)
        self.spinBoxClusterRestarts.setValue(p.caRestarts)
        annots = set()
        for i in range(self.listWidgetClusterAnnotations.count()):
            item = self.listWidgetClusterAnnotations.item(i)
            annots.add(item.text())
        for a in p.caAnnotations:
            if a not in annots:
                self.caAddAnnotation(name=a)


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



def main():
    parser = argparse.ArgumentParser(
            description='Graphical application for decomposition of '+
            'hyperspectral data.')
    parser.add_argument('files', metavar='file', nargs='*',
                        help='initial hyperspectral images to load')
    parser.add_argument('-p', '--params', metavar='file.djs', dest='paramFile',
                        help='parameter file to load')
    MyMainWindow.run_octavvs_application(
        parser=parser, parameters=['files', 'paramFile'])


