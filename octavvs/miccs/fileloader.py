#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 11 19:59:34 2020

@author: carl
"""

import os
import os.path
import fnmatch
import traceback
from pkg_resources import resource_filename

from PyQt5.QtWidgets import QWidget, QInputDialog
from PyQt5.QtWidgets import QMessageBox, QFileDialog
from PyQt5.QtCore import QThread, pyqtSignal, pyqtSlot, Qt
from PyQt5 import uic

from . import NoRepeatStyle

FileLoaderUi = uic.loadUiType(resource_filename(__name__, "fileloader.ui"))[0]

class FileLoaderWidget(QWidget, FileLoaderUi):
    """
    Widget with a bunch of graphical components related to loading a list of files
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setupUi(self)

        self.spinBoxFileNumber.setStyle(NoRepeatStyle())
        self.comboBoxSingMult.currentIndexChanged.connect(
                lambda ix: self.lineEditLoadFilter.setEnabled(ix > 0))

    def setEnabled(self, onoff):
        self.pushButtonLoad.setEnabled(onoff)
        self.pushButtonAdd.setEnabled(onoff)
        self.spinBoxFileNumber.setEnabled(onoff)
        self.pushButtonShowFiles.setEnabled(onoff)
        self.lineEditSaveExt.setEnabled(onoff)

    def setSuffix(self, suffix):
        self.lineEditSaveExt.setText(suffix)

    def saveParameters(self, p):
        "Copy from UI to some kind of parameters object"
        p.fileFilter = self.lineEditLoadFilter.text()
        p.saveExt = self.lineEditSaveExt.text()

    def loadParameters(self, p):
        "Copy to UI from some kind of parameters object"
        self.lineEditKeyword.setText(p.fileFilter)
        self.lineEditSaveExt.setText(p.saveExt)



class FileLoader():
    """
    Mixin class that adds file loading UI stuff to an OctavvsMainWindow.

    Things assumed to exist in self:
        fileLoader - FileLoader (from UI)
        settings - QSettings (from OctavvsApplication)
        data - SpectralData
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.fileLoader.pushButtonLoad.clicked.connect(self.loadFolder)
        self.fileLoader.pushButtonAdd.clicked.connect(self.addFolder)
        self.fileLoader.spinBoxFileNumber.valueChanged.connect(self.selectFile)
        self.fileLoader.pushButtonShowFiles.clicked.connect(self.showFileList)


    def addFolder(self):
        self.loadFolder(add=True)

    def loadFolder(self, add=False):
        mode = self.fileLoader.comboBoxSingMult.currentIndex()
        if mode > 0:
            recurse = mode == 2
            pattern = self.fileLoader.lineEditLoadFilter.text()

            foldername = self.getDirectoryName(
                    "Load spectra from directory (filter: %s)" % (pattern),
                    settingname='spectraDir')
            if not foldername:
                return

            filenames = []
            for root, dirs, files in os.walk(foldername):
                self.fileLoader.lineEditFilename.setText(root)
                if '/' in pattern:
                    files = [ os.path.join(root, f) for f in files ]
                    files = fnmatch.filter(files, pattern)
                    filenames += files
                else:
                    files = fnmatch.filter(files, pattern)
                    filenames += [ os.path.join(root, f) for f in files ]
#                self.lineEditTotal.setText(str(len(filenames)))
                if not recurse:
                    break;
#            self.lineEditTotal.setText('' if self.data.filenames is None else
#                                       str(len(self.data.filenames)))
            if not filenames:
                self.errorMsg.showMessage('No matching files found in directory "' +
                                          foldername + '"')
                return
            filenames.sort()
        else:
            filenames = self.getLoadFileNames("Open hyperspectral image",
                        filter="Matrix files (*.mat *.txt *.csv *.0 *.1 *.2 *.3);;All files (*)",
                        settingname='spectraDir')
            if not filenames:
                return
            foldername = os.path.dirname(filenames[0])

        if self.updateFileList(filenames, False, add=add):
            self.data.foldername = foldername
            self.settings.setValue('whitelightDir', foldername)

    def loadFile(self, file):
        "Load a file or return (error message, traceback) from trying"
        try:
            self.data.readMatrix(file)
            return
        except (RuntimeError, FileNotFoundError) as e:
            return (str(e), None)
        except Exception as e:
            return (repr(e), traceback.format_exc())

    def updateFileListInfo(self, filenames):
        "Helper function for updateFileList"
        self.data.filenames = filenames
        self.fileLoader.lineEditTotal.setText(str(len(filenames)))
        self.fileLoader.spinBoxFileNumber.setMaximum(len(filenames))
        self.fileLoader.spinBoxFileNumber.setEnabled(True)

    def updateFileList(self, filenames, avoidreload, add=False):
        """
        Updates the list of files and loads a file if possible
        Parameters:
            filenames: list of files from dialog/user
            avoidreload: if possible, select the current file in the list without reload
        Returns:
            True if a file was loaded, False if the user cancelled
        """
        if add:
            existing = set(self.data.filenames)
            filenames = self.data.filenames + [f for f in filenames if f not in existing]

        if avoidreload:
            # Is the current file already in our list?
            try:
                ix = filenames.index(self.data.curFile)
            except ValueError:
                pass
            else:
                self.updateFileListInfo(filenames)
                self.fileLoader.spinBoxFileNumber.setValue(ix+1)
                return

        skipall = False
        while len(filenames):
            err = self.loadFile(filenames[0])
            if err is None:
                self.updateFileListInfo(filenames)
                self.updateFile(0)
                return True

            if not skipall:
                q = self.loadErrorBox(filenames[0], err)
                q.addButton('Abort', QMessageBox.RejectRole)
                if len(filenames) > 1:
                    q.addButton('Skip file', QMessageBox.AcceptRole)
                    q.addButton('Skip all with errors', QMessageBox.AcceptRole)
                ret = q.exec()
                if ret == 0:
                    break
                elif ret == 2:
                    skipall = True
            filenames = filenames[1:]
        return False

    def showFileList(self):
        s, ok = QInputDialog.getMultiLineText(
                self, 'Input files',
                'Editable list of input files',
                '\n'.join(self.data.filenames))
        if ok:
            fn = s.splitlines()
            if fn:
                self.updateFileList(fn, True)

    def selectFile(self):
        skipall = False
        while True:
            num = self.fileLoader.spinBoxFileNumber.value() - 1
            if (num < 0 or num >= len(self.data.filenames) or
                self.data.filenames[num] == self.data.curFile):
                return
            err = self.loadFile(self.data.filenames[num])
            if not err:
                self.updateFile(num)
                break
            elif not skipall:
                q = self.loadErrorBox(self.data.filenames[num], err)
                q.addButton('Skip file', QMessageBox.AcceptRole)
                q.addButton('Skip all with errors', QMessageBox.AcceptRole)
                ret = q.exec()
                if ret:
                    skipall = True

            del self.data.filenames[num]
            self.updateFileListInfo(self.data.filenames)


    def updateFile(self, num):
        file = self.data.filenames[num]
        self.data.curFile = file
        self.fileLoader.spinBoxFileNumber.setValue(num+1)
        self.fileLoader.lineEditFilename.setText(file)

