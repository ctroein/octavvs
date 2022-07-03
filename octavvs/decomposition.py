import os
import traceback
from functools import partial
from pkg_resources import resource_filename
import argparse

from PyQt5.QtWidgets import QMessageBox, QStyle, QTableWidget, \
    QTableWidgetItem, QMainWindow, QDialog, QDialogButtonBox, \
    QListWidgetItem, QMenu, QAction
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
from octavvs.algorithms import normalization
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

        # Not really needed
        self.pushButtonSaveParameters.clicked.connect(self.saveParameters)
        self.pushButtonLoadParameters.clicked.connect(self.loadParameters)

        self.imageVisualizer.plot_raw.updatedProjection.connect(
            self.plot_roi.set_data)
        self.pushButtonRoiClear.clicked.connect(self.roiClear)
        self.pushButtonRoiDraw.clicked.connect(self.roiDrawFree)
        self.pushButtonRoiAdd.clicked.connect(self.roiAddArea)
        self.pushButtonRoiRemove.clicked.connect(self.roiRemoveArea)
        self.pushButtonRoiErase.clicked.connect(
            self.plot_roi.erase_last_point)
        self.pushButtonRoiInvert.clicked.connect(self.plot_roi.invert_roi)
        self.plot_roi.updated.connect(self.roiUpdateSelected)

        # Connect all the data loading/saving stuff
        for att, what in {'Roi': 'roi', 'Dc': 'decomposition',
                          'Ca': 'clustering'}.items():
            getattr(self, 'pushButton%sDirectory' % att).clicked.connect(
                partial(self.genericDirSelect, what=what))
            getattr(self, 'lineEdit%sDirectory' % att).editingFinished.connect(
                partial(self.reloadCheck, what=what))
            getattr(self, 'pushButton%sLoad' % att).clicked.connect(
                partial(self.genericLoad, what=what))
            getattr(self, 'pushButton%sSave' % att).clicked.connect(
                partial(self.genericSave, what=what))

        self.imageVisualizer.plot_spectra.updated.connect(self.updateDC)
        self.pushButtonStart.clicked.connect(self.startDC)
        self.pushButtonStop.clicked.connect(self.stopDC)
        self.pushButtonRun.clicked.connect(self.dcBatchRun)
        self.pushButtonRunStop.clicked.connect(self.stopDC)

        self.pushButtonSettingsInfo.clicked.connect(self.dcSettingsShow)
        self.dialogSettingsTable = DialogSettingsTable()
        self.dialogSettingsTable.buttonBox.clicked.connect(
            self.dcSettingsClicked)
        # todo: Preset editing - save in application settings?

        self.comboBoxInitialValues.currentIndexChanged.connect(
            self.dcInitialValuesChanged)
        self.lineEditInitSkew.setFormat('%g')
        self.lineEditInitSkew.setRange(1, 1000)
        self.lineEditInitNoise.setFormat('%g')
        self.lineEditInitNoise.setRange(.001, 1)
        self.comboBoxDerivative.currentIndexChanged.connect(
            self.dcDerivativeChanged)
        self.checkBoxContrast.toggled.connect(self.dcContrastChanged)
        self.lineEditContrast.setFormat("%g")
        self.lineEditContrast.setRange(0, .1)
        self.lineEditContrast.editingFinished.connect(self.dcContrastEdit)
        self.horizontalSliderContrast.valueChanged.connect(
            self.dcContrastSlide)

        self.comboBoxPlotMode.currentIndexChanged.connect(
            self.plot_decomp.set_display_mode)
        self.plot_decomp.displayModesUpdated.connect(
            self.updateDCPlotModes)

        for plot in [self.plot_roi, self.plot_decomp, self.plot_cluster]:
            self.imageVisualizer.updatePixelSize.connect(
                plot.set_pixel_size)
            self.imageVisualizer.comboBoxCmaps.currentTextChanged.connect(
                plot.set_cmap)
        self.imageVisualizer.pixelSizeEdit()

        self.pushButtonCluster.clicked.connect(self.caStartClustering)
        # self.toolButtonAnnotationPlus.setIcon(
        #     self.style().standardIcon(QStyle.SP_DialogYesButton))
        self.toolButtonAnnotationMinus.setIcon(
            self.style().standardIcon(QStyle.SP_TrashIcon))
        self.toolButtonAnnotationPlus.clicked.connect(self.caAddAnnotation)
        self.toolButtonAnnotationMinus.clicked.connect(self.caDelAnnotation)
        self.listWidgetClusterAnnotations.itemSelectionChanged.connect(
            self.caSelectAnnotation)
        self.listWidgetClusterAnnotations.itemChanged.connect(
            self.caEditedAnnotation)
        self.listWidgetClusterUsed.itemDoubleClicked.connect(
            self.caClickedUsed)
        self.listWidgetClusterUnused.itemDoubleClicked.connect(
            self.caClickedUnused)
        self.plot_cluster.clicked.connect(self.caSelectCluster)

        exportMenu = QMenu()
        for txt, a, b in [['Annotation averages', 'annotation', False],
                         ['Cluster averages', 'cluster', False],
                         ['Individual spectra', None, False],
                         ['Batch, annotation averages', 'annotation', True],
                         ['Batch, cluster averages', 'cluster', True],
                         ['Batch, individual spectra', None, True]]:
            exportAction = QAction(txt, self)
            exportAction.triggered.connect(
                partial(self.caExport, average=a, batch=b))
            exportAction.setIconVisibleInMenu(False)
            exportMenu.addAction(exportAction)
        self.toolButtonClusterExport.setMenu(exportMenu)

        self.worker = DecompWorker()
        self.workerThread = QThread()
        self.worker.moveToThread(self.workerThread)
        self.closing.connect(self.workerThread.quit)
        self.startDecomp.connect(self.worker.decompose)
        self.worker.done.connect(self.dcDone)
        self.worker.stopped.connect(self.dcStopped)
        self.worker.failed.connect(self.dcFailed)
        self.worker.progress.connect(self.dcProgress)
        self.worker.progressPlot.connect(self.dcProgressPlot)
        self.worker.batchProgress.connect(self.dcBatchProgress)
        self.worker.fileLoaded.connect(self.updateFile)
        self.worker.loadFailed.connect(self.showLoadErrorMessage)
        self.startBatch.connect(self.worker.startBatch)
        self.worker.batchDone.connect(self.dcBatchDone)
        self.workerThread.start()

        self.lineEditTolerance.setFormat("%g")
        self.lineEditTolerance.setRange(1e-10, 1)
        # self.lineEditRelError.setFormat("%.4g")

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
            dups = self.data.get_duplicate_filenames()
            if dups:
                q = QMessageBox(self)
                q.setIcon(QMessageBox.Warning)
                q.setWindowTitle('Warning: identical filenames')
                q.setText('Some input files have identical names, which '
                          'will cause problems with overwritten output files.')
                q.setTextFormat(Qt.PlainText)
                q.setDetailedText('Examples:\n' + '\n'.join(list(dups)[:10]))
                q.exec()
        return ret

    def loadFile(self, file):
        "Save ROI before proceeding"
        self.autosaveCheck()
        return super().loadFile(file)

    def populatePlots(self):
        "Send data to plots etc after loading rdc"
        self.plot_roi.set_roi(self.data.roi)
        self.dcSettingsUpdate()
        self.plot_decomp.clear_and_set_roi(self.data.decomposition_roi)
        self.plot_decomp.set_errors(self.data.decomposition_errors)
        dd = self.data.decomposition_data
        if dd:
            if 0 in dd:
                self.plot_decomp.set_initial_spectra(dd[0]['spectra'])
            lastiter = max(dd.keys())
            ds = dd[lastiter]
            self.plot_decomp.set_concentrations(ds['concentrations'])
            self.plot_decomp.set_spectra(ds['spectra'])
        self.plot_cluster.set_roi_and_clusters(
            self.data.clustering_roi, self.data.clustering_labels)
        self.caPopulateLists()

    @pyqtSlot(int)
    def updateFile(self, num):
        super().updateFile(num)
        self.updateWavenumberRange()
        geom = dict(pixels=len(self.data.raw),
                    wh=self.data.wh, pixelxy=self.data.pixelxy)
        self.plot_roi.set_basic_data(**geom)
        self.plot_decomp.set_basic_data(
            wn=self.imageVisualizer.plot_raw.getWavenumbers(), **geom)
        self.plot_cluster.set_basic_data(**geom)
        self.autoLoad()
        self.populatePlots()
        self.updatedFile()

    @pyqtSlot(str, str, str)
    def showLoadErrorMessage(self, file, err, details):
        """
        Error message from loading a file in the worker thread,
        with abort option
        """
        warning=False
        q = self.loadErrorBox(file, (err, details if details else None),
                              warning)
        q.addButton('Ignore' if warning else 'Skip file',
                    QMessageBox.AcceptRole)
        abort = q.addButton('Abort', QMessageBox.AcceptRole)
        q.exec()
        if q.clickedButton() == abort:
            self.stopDC()

    def updateDimensions(self, wh):
        super().updateDimensions(wh)
        self.plot_roi.adjust_geometry(wh)
        self.plot_decomp.adjust_geometry(wh)
        self.plot_cluster.adjust_geometry(wh)

    def setPlotColors(self, cmap):
        super().setPlotColors(cmap)
        self.plot_decomp.draw_idle()

    # Loading and saving data
    def suggestedFilename(self, what):
        if not self.data.curFile:
            return None
        fn = os.path.splitext(os.path.basename(self.data.curFile))[0]
        if what == 'roi':
            return os.path.join(self.lineEditRoiDirectory.text(), fn) + '.roi'
        if what == 'decomposition':
            return os.path.join(self.lineEditDcDirectory.text(), fn) + '.dcn'
        if what == 'clustering':
            return os.path.join(self.lineEditCaDirectory.text(), fn) + '.can'

    def fileFormats(self, what):
        if what == 'roi':
            ff = "ROI HDF5 files (*.roi);;ROI MATLAB files (*.mat);;"\
                "ROI CVS files (*.csv);;"
            descr = 'ROI'
        elif what == 'decomposition':
            ff = "Decomposition HDF5 files (*.dcn);;"
            descr = 'decomposition data'
        elif what == 'clustering':
            ff = "Clustering/annotation HDF5 files (*.can);;"
            descr = 'clustering/annotations'
        else:
            ff = ""
            descr = 'data'
        return ff + "Combined HDF5 files (*.odd);;All files (*)", descr

    def genericLoad(self, what='all'):
        if not self.data.curFile:
            return
        path = os.path.split(self.suggestedFilename(what))
        filter, description = self.fileFormats(what)
        filename = self.getLoadFileName(
            "Load %s from file" % description,
            filter=filter,
            directory=path[0], defaultfilename=path[1])
        if not filename:
            return
        self.data.load_rdc(filename, what=what)
        self.populatePlots()

    def autoLoad(self):
        for what in ['roi', 'decomposition', 'clustering']:
            filename = self.suggestedFilename(what)
            if filename and os.path.exists(filename):
                self.data.load_rdc(filename, what=what)

    def genericSave(self, what='all'):
        if not self.data.curFile:
            return
        path = os.path.split(self.suggestedFilename(what))
        filter, description = self.fileFormats(what)
        filename = self.getSaveFileName(
            "Save %s to file" % description,
            filter=filter,
            directory=path[0], defaultfilename=path[1])
        if not filename:
            return

        # Make sure self.data is up-to-date
        if what in ['roi', 'all']:
            self.data.set_roi(self.plot_roi.get_roi())
        if what in ['decomposition', 'all']:
            ...
        if what in ['clustering', 'all']:
            self.caSelectAnnotation()
        ext = os.path.splitext(filename)[1]
        if what == 'roi':
            self.data.save_roi(filename=filename, fmt=ext)
        else:
            self.data.save_rdc(filename=filename, what=what)

    def autosaveCheck(self):
        if self.data.curFile and not self.workerRunning:
            save = []
            if self.checkBoxRoiAutosave.isChecked():
                self.data.set_roi(self.plot_roi.get_roi())
                save.append('roi')
            if self.checkBoxDcAutosave.isChecked():
                save.append('decomposition')
            if self.checkBoxCaAutosave.isChecked():
                self.caSelectAnnotation()
                save.append('clustering')
            for what in save:
                self.data.save_rdc(self.suggestedFilename(
                    what=what), what=what)

    def reloadCheck(self, what):
        filename = self.suggestedFilename(what)
        if filename:
            descr = self.fileFormats(what)[1]
            if os.path.exists(filename):
                yn = QMessageBox.question(
                    self, 'Directory changed',
                    '(Re)load saved %s from selected directory?' % descr)
                if yn == QMessageBox.Yes:
                    self.data.load_rdc(filename, what=what)
                    self.populatePlots()

    def genericDirSelect(self, what):
        descr = self.fileFormats(what)[1]
        short = {'roi': 'roi', 'decomposition': 'dc',
                 'clustering': 'ca', 'all': 'rdc'}[what]
        rdir = self.getDirectoryName(
            'Select %s directory' % descr, settingname=short+'Dir',
            default=os.path.dirname(self.data.curFile))
        if rdir is not None:
            Short = short[0].upper() + short[1:]
            getattr(self, 'lineEdit%sDirectory' % Short).setText(rdir)
            self.reloadCheck(what=what)

    # Roi, Region of interest
    def roiClear(self):
        self.roiDrawFree(True)
        self.plot_roi.clear()

    def roiCheckButtons(self):
        mode = self.plot_roi.get_draw_mode()
        self.pushButtonRoiDraw.setChecked(mode == 'click')
        self.pushButtonRoiAdd.setChecked(mode == 'add')
        self.pushButtonRoiRemove.setChecked(mode == 'remove')

    def roiDrawFree(self, checked):
        self.plot_roi.set_draw_mode('click')
        self.roiCheckButtons()

    def roiAddArea(self, checked):
        self.plot_roi.set_draw_mode('add' if checked else 'click')
        self.roiCheckButtons()

    def roiRemoveArea(self, checked):
        self.plot_roi.set_draw_mode('remove' if checked else 'click')
        self.roiCheckButtons()

    def roiUpdateSelected(self, n, m):
        self.labelRoiSelected.setText('Selected: %d / %d' % (n, m))


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
        self.pushButtonRoiLoad.setEnabled(onoff)
        self.pushButtonRoiSave.setEnabled(onoff)
        self.pushButtonDcLoad.setEnabled(onoff)
        self.pushButtonDcSave.setEnabled(onoff)
        self.pushButtonCaLoad.setEnabled(onoff)
        self.pushButtonCaSave.setEnabled(onoff)
        self.pushButtonRun.setEnabled(onoff)
        self.pushButtonStart.setEnabled(onoff)
        # if newstate:
        #     self.progressBarIteration.setValue(0)
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
        # self.progressBarIteration.setMaximum(params.dcIterations)
        # self.progressBarIteration.setFormat('initializing')
        self.lineEditIteration.setText('init')
        self.toggleRunning(1)
        self.startDecomp.emit(self.data, params)

    def clearDC(self):
        if self.workerRunning == 1:
            return
        self.lineEditIteration.setText('idle')
        # self.progressBarIteration.setValue(0)
        # self.progressBarIteration.setFormat('idle')

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
    def dcDone(self, spectra, concentrations, errors):
        self.plot_decomp.set_spectra(spectra)
        self.plot_decomp.set_concentrations(concentrations)
        self.plot_decomp.set_errors(errors)
        self.lineEditIteration.setText('%d/done' % len(errors))
        self.dcStopped()
        self.caClear()

    @pyqtSlot(int, float)
    def dcProgress(self, iteration, error):
        if iteration:
            self.lineEditIteration.setText('%d' % iteration)
            self.lineEditRelError.setText('%.4f' % (error * 100))

    @pyqtSlot(int, np.ndarray, np.ndarray, list)
    def dcProgressPlot(self, iteration, spectra, concentrations, errors):
        if iteration:
            self.plot_decomp.set_concentrations(concentrations)
            self.plot_decomp.set_spectra(spectra)
        else:
            # self.data.add_decomposition_data(0, spectra, None)
            self.plot_decomp.set_initial_spectra(spectra)
        self.plot_decomp.set_errors(errors)

    def dcInitialValuesChanged(self):
        self.stackedWidgetInit.setCurrentIndex(
            self.comboBoxInitialValues.currentIndex())

    def dcDerivativeChanged(self):
        en = self.comboBoxDerivative.currentIndex() > 0
        self.spinBoxDerivativePoly.setEnabled(en)
        self.spinBoxDerivativeWindow.setEnabled(en)

    def dcContrastChanged(self):
        en = self.checkBoxContrast.isChecked()
        self.comboBoxContrast.setEnabled(en)
        self.horizontalSliderContrast.setEnabled(en)
        self.lineEditContrast.setEnabled(en)

    def dcContrastSlide(self):
        self.lineEditContrast.setValue(
            self.horizontalSliderContrast.value() * .001)

    def dcContrastEdit(self):
        self.horizontalSliderContrast.setValue(
            self.lineEditContrast.value() * 1000)

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
        self.progressBarRun.setMaximum(b-1)

    @pyqtSlot(bool)
    def dcBatchDone(self, success):
        self.worker.halt = False
        self.lineEditIteration.setText('done' if success else 'fail')
        self.toggleRunning(0)


    # CA, Clustering and annotation
    def caStartClustering(self):
        if self.data.raw is None:
            return
        p = self.getParameters()
        if p.caInput == 'decomposition':
            dd = self.data.decomposition_data
            if dd is None:
                return
            lastiter = max(dd.keys())
            if not lastiter:
                return # Only initial spectra available
            inp = dd[lastiter]['concentrations'].T
            roi = self.data.decomposition_roi
        else:
            inp = self.data.raw
            roi = self.data.roi if p.caInput == 'raw-roi' else None
            if roi is not None:
                inp = inp[roi]
        inp = normalization.normalize_features(inp, p.caNormalization)

        if p.caMethod == 'kmeans':
            clusterer = sklearn.cluster.KMeans(
                n_clusters=p.caClusters, n_init=p.caRestarts)
            labels = clusterer.fit_predict(inp)
        elif p.caMethod == 'mbkmeans':
            clusterer = sklearn.cluster.MiniBatchKMeans(
                n_clusters=p.caClusters, n_init=p.caRestarts)
            labels = clusterer.fit_predict(inp)
        elif p.caMethod == 'strongest':
            labels = inp.argmax(1)
            p.caClusters = labels.max() + 1
        else:
            raise ValueError('Unknown clustering method %s' % p.caMethod)
        # elif p.caMethod == 'meanshift':
        #     clusterer = sklearn.cluster.MeanShift(n_jobs=-1)
        # elif p.caMethod == 'spectral':
        #     clusterer = sklearn.cluster.SpectralClustering(
        #         n_clusters=p.caClusters, n_init=p.caRestarts, n_jobs=-1)
        # labels = clusterer.fit_predict(inp)
        # if p.caMethod == 'meanshift':
        #     p.caClusters = labels.max() + 1
        self.data.set_clustering_settings(p, roi)
        self.data.set_clustering_labels(labels, relabel=True)
        self.plot_cluster.set_roi_and_clusters(
            self.data.clustering_roi, self.data.clustering_labels)
        self.caPopulateLists()

    def caAddAnnotation(self, checked=False, name='New annotation',
                        edit=True):
        # self.caSelectAnnotation()
        # self.selectedAnnotation = None
        item = QListWidgetItem(name, parent=self.listWidgetClusterAnnotations)
        item.setFlags(item.flags() | Qt.ItemIsEditable)
        item.octavvs_id = name
        self.listWidgetClusterAnnotations.addItem(item)
        if edit:
            self.listWidgetClusterAnnotations.setCurrentItem(item)
            self.listWidgetClusterAnnotations.editItem(item)

    def caClustersToList(self, clusters, listw):
        inlist = set()
        for r in reversed(range(listw.count())):
            c = listw.item(r).octavvs_id
            if c not in clusters:
                listw.takeItem(r)
            else:
                inlist.add(c)
        for c in clusters - inlist:
            item = QListWidgetItem('Cluster %d' % c)
            cc = np.asarray(self.plot_cluster.cluster_color(c)[:3]) * 255
            item.setBackground(QtGui.QBrush(QtGui.QColor(*cc)))
            item.octavvs_id = c
            listw.addItem(item)

    def caClearList(self, listw):
        while listw.count():
            listw.takeItem(0)

    def caPopulateLists(self):
        for l in [self.listWidgetClusterUsed, self.listWidgetClusterUnused ]:
            self.caClearList(l)
        awlist = self.listWidgetClusterAnnotations
        awlist.setCurrentItem(None)
        self.listWidgetClusterUsed.setEnabled(False)
        anns = set()
        for r in reversed(range(awlist.count())):
            name = awlist.item(r).text()
            # Remove duplicates. Should maybe also remove if not in self.data
            if name in anns:
                awlist.takeItem(r)
            else:
                anns.add(name)
        if self.data.clustering_annotations:
            for name in self.data.clustering_annotations:
                if name not in anns:
                    self.caAddAnnotation(name=name, edit=False)
        if self.data.clustering_labels is not None:
            self.caClustersToList(self.data.get_unannotated_clusters(),
                                  self.listWidgetClusterUnused)

    def caUpdateAnnotation(self):
        "Save selected clusters to self.data"
        sel = self.listWidgetClusterAnnotations.selectedItems()
        sel = sel[0].text() if sel else None
        if sel:
            clusters = set()
            for r in range(self.listWidgetClusterUsed.count()):
                clusters.add(self.listWidgetClusterUsed.item(r).octavvs_id)
            self.data.set_annotation_clusters(sel, clusters)
        else:
            assert not self.listWidgetClusterUsed.isEnabled()

    def caSelectAnnotation(self):
        "Update the currently selected annotation"
        sel = self.listWidgetClusterAnnotations.selectedItems()
        sel = sel[0].text() if sel else None
        if sel:
            used = self.data.get_annotation_clusters(sel)
            self.caClustersToList(used, self.listWidgetClusterUsed)
        else:
            self.caClearList(self.listWidgetClusterUsed)
        self.listWidgetClusterUsed.setEnabled(sel is not None)

    def caStoreAnnotations(self):
        "Save all to self.data"
        self.caUpdateAnnotation()
        awlist = self.listWidgetClusterAnnotations
        for r in range(awlist.count()):
            name = awlist.item(r).text()
            self.data.set_annotation_clusters(name)

    def caDelAnnotation(self):
        "Delete annotation and make items unused"
        sel = self.listWidgetClusterAnnotations.selectedItems()
        if not sel:
            return
        self.data.del_annotation(sel[0].text())
        while self.listWidgetClusterUsed.count():
            item = self.listWidgetClusterUsed.takeItem(0)
            self.listWidgetClusterUnused.addItem(item)
        self.listWidgetClusterAnnotations.takeItem(
            self.listWidgetClusterAnnotations.row(sel[0]))

    def caEditedAnnotation(self, item):
        "Enact annotation name change"
        nnew = item.text()
        clusters = self.data.get_annotation_clusters(nnew)
        if hasattr(item, 'octavvs_id'):
            nold = item.octavvs_id
            clusters.update(self.data.get_annotation_clusters(nold))
            self.data.del_annotation(nold)
        item.octavvs_id = nnew
        self.data.set_annotation_clusters(nnew, clusters)
        self.caUpdateAnnotation()

    def caClickedUsed(self, item):
        if not self.listWidgetClusterAnnotations.selectedItems():
            return
        self.listWidgetClusterUsed.takeItem(
            self.listWidgetClusterUsed.row(item))
        self.listWidgetClusterUnused.addItem(item)
        self.caUpdateAnnotation()

    def caClickedUnused(self, item):
        if not self.listWidgetClusterAnnotations.selectedItems():
            return
        self.listWidgetClusterUnused.takeItem(
            self.listWidgetClusterUnused.row(item))
        self.listWidgetClusterUsed.addItem(item)
        self.caUpdateAnnotation()

    def caSelectCluster(self, num):
        if not self.listWidgetClusterAnnotations.selectedItems():
            return
        for r in range(self.listWidgetClusterUnused.count()):
            item = self.listWidgetClusterUnused.item(r)
            if item.octavvs_id == num:
                item = self.listWidgetClusterUnused.takeItem(
                    self.listWidgetClusterUnused.row(item))
                self.listWidgetClusterUsed.addItem(item)
                self.caUpdateAnnotation()
                break

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

    def caExport(self, average=None, batch=False):
        if not self.data.curFile:
            return
        if batch:
            dirname = self.getDirectoryName(
                "Batch export annotated spectra",
                settingname='exportDir')
            for i in range(len(self.data.filenames)):
                self.fileLoader.spinBoxFileNumber.setValue(i + 1)
                filename = os.path.join(
                    dirname, os.path.splitext(
                        os.path.basename(self.data.curFile))[0]+'.csv')
                try:
                    self.data.save_annotated_spectra(
                        filename, filetype='csv', average=average)
                except Exception as e:
                    self.showDetailedErrorMessage(
                            "Error saving settings to "+filename+": "+repr(e),
                            traceback.format_exc())
                    return
        else:
            filename = os.path.splitext(self.data.curFile)[0]+'.csv'
            filename = self.getSaveFileName(
                    "Export annotated spectra",
                    filter="Comma-separated values (*.csv);;All files (*)",
                    settingname='exportDir',
                    suffix='csv',
                    defaultfilename=filename)
            if not filename:
                return
            try:
                self.data.save_annotated_spectra(
                    filename, filetype='csv', average=average)
            except Exception as e:
                self.showDetailedErrorMessage(
                        "Error saving settings to "+filename+": "+repr(e),
                        traceback.format_exc())


    # Parameter handling
    useRoiNames = ['ignore', 'ifdef', 'require']
    dcInitialValuesNames = ['simplisma', 'kmeans', 'clustersubtract']
    dcAlgorithmNames = ['mcr-als-anderson', 'mcr-als']
    caInputNames = ['raw-roi', 'raw', 'decomposition']
    caMethodNames = ['kmeans', 'mbkmeans', 'strongest']
    caNormalizationNames = ['none', 'mean1', 'mean0', 'mean0var1']
    # caMethodNames = ['kmeans', 'meanshift', 'spectral']

    # Loading and saving parameters
    def getParameters(self):
        """Copy from UI to some kind of parameters object."""
        p = Parameters()
        self.fileLoader.saveParameters(p)
        self.imageVisualizer.saveParameters(p)

        # p.dirMode = self.comboBoxDirectory.currentIndex()
        # p.directory = self.lineEditDirectory.text()
        # p.autosave = self.checkBoxRdcAutosave.isChecked()

        p.dcAlgorithm = self.dcAlgorithmNames[
            self.comboBoxAlgorithm.currentIndex()]
        p.dcImpute = self.checkBoxImpute.isChecked()
        p.dcDerivative = self.comboBoxDerivative.currentIndex()
        p.dcDerivativeWindow = self.spinBoxDerivativeWindow.value()
        p.dcDerivativePoly = self.spinBoxDerivativePoly.value()
        p.dcComponents = self.spinBoxComponents.value()
        p.dcStartingPoint = self.comboBoxStartingPoint.currentIndex()
        p.dcInitialValues = self.dcInitialValuesNames[
            self.comboBoxInitialValues.currentIndex()]
        p.dcInitialValuesSkew = self.lineEditInitSkew.value()
        p.dcInitialValuesPower = self.spinBoxInitPower.value()
        p.dcInitialValuesNoise = self.lineEditInitNoise.value()
        p.dcRoi = self.useRoiNames[self.comboBoxUseRoi.currentIndex()]
        p.dcContrast = self.checkBoxContrast.isChecked()
        p.dcContrastConcentrations = self.comboBoxContrast.currentIndex()
        p.dcContrastWeight = self.lineEditContrast.value()
        p.dcIterations = self.spinBoxIterations.value()
        p.dcTolerance = self.lineEditTolerance.value()

        # For batch job, not for load/save of settings
        p.dcDirectory = self.lineEditDcDirectory.text()

        p.caInput = self.caInputNames[self.comboBoxClusterInput.currentIndex()]
        p.caNormalization = self.caNormalizationNames[
            self.comboBoxClusterNormalization.currentIndex()]
        p.caMethod = self.caMethodNames[
            self.comboBoxClusterMethod.currentIndex()]
        p.caClusters = self.spinBoxClusterClusters.value()
        p.caRestarts = self.spinBoxClusterRestarts.value()
        p.caAnnotations = []
        for i in range(self.listWidgetClusterAnnotations.count()):
            item = self.listWidgetClusterAnnotations.item(i)
            p.caAnnotations.append(item.text())

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
        """Copy from some kind of parameters object to UI."""
        self.fileLoader.loadParameters(p)
        self.imageVisualizer.loadParameters(p)

        # self.lineEditDirectory.setText(p.directory)
        # self.comboBoxDirectory.setCurrentIndex(p.dirMode)
        # self.checkBoxRdcAutosave.setChecked(p.autosave)

        self.comboBoxAlgorithm.setCurrentIndex(
            self.dcAlgorithmNames.index(p.dcAlgorithm))
        self.checkBoxImpute.setChecked(p.dcImpute)
        self.comboBoxDerivative.setCurrentIndex(p.dcDerivative)
        self.spinBoxDerivativeWindow.setValue(p.dcDerivativeWindow)
        self.spinBoxDerivativePoly.setValue(p.dcDerivativePoly)
        self.spinBoxComponents.setValue(p.dcComponents)
        self.comboBoxStartingPoint.setCurrentIndex(p.dcStartingPoint)
        self.comboBoxInitialValues.setCurrentIndex(
            self.dcInitialValuesNames.index(p.dcInitialValues))
        self.lineEditInitSkew.setValue(p.dcInitialValuesSkew)
        self.spinBoxInitPower.setValue(p.dcInitialValuesPower)
        self.lineEditInitNoise.setValue(p.dcInitialValuesNoise)
        self.comboBoxUseRoi.setCurrentIndex(self.useRoiNames.index(p.dcRoi))
        self.checkBoxContrast.setChecked(p.dcContrast)
        self.comboBoxContrast.setCurrentIndex(p.dcContrastConcentrations)
        self.lineEditContrast.setValue(p.dcContrastWeight)
        self.spinBoxIterations.setValue(p.dcIterations)
        self.lineEditTolerance.setValue(p.dcTolerance)

        self.comboBoxClusterInput.setCurrentIndex(
            self.caInputNames.index(p.caInput))
        self.comboBoxClusterNormalization.setCurrentIndex(
             self.caNormalizationNames.index(p.caNormalization))
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
                self.caAddAnnotation(name=a, edit=False)


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


