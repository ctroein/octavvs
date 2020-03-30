#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 21:07:55 2020

@author: carl
"""

import os
import sys
import traceback
import multiprocessing

from PyQt5.QtWidgets import QMainWindow, QApplication, QFileDialog
from PyQt5.QtWidgets import QErrorMessage, QInputDialog, QDialog, QMessageBox
from PyQt5.QtCore import Qt, QSettings
from PyQt5.Qt import qApp

from .exceptiondialog import ExceptionDialog
from .copyfigure import add_clipboard_to_figures




class OctavvsMainWindow(QMainWindow):

    fileOptions = QFileDialog.Options() | QFileDialog.DontUseNativeDialog

    octavvs_version = 'v0.0.31'

    def __init__(self, parent=None):
        super().__init__(parent)
        self.settings = QSettings('MICCS', 'OCTAVVS ' + self.program_name())
        qApp.installEventFilter(self)
        self.setupUi(self)

        self.errorMsg = QErrorMessage(self)
        self.errorMsg.setWindowModality(Qt.WindowModal)

    def post_setup(self):
        "Called by children after setting up UI but before loading data etc"
        self.setWindowTitle('OCTAVVS %s %s' % (self.program_name(), self.octavvs_version))
        ExceptionDialog.install(self)

    @classmethod
    def program_name(cls):
        "Return the name of the program that this main window represents"
        return cls.__name__

    def showDetailedErrorMessage(self, err, details):
        "Show an error dialog with details"
        q = QMessageBox(self)
        q.setIcon(QMessageBox.Critical)
        q.setWindowTitle("Error")
        q.setText(err)
        q.setTextFormat(Qt.PlainText)
        q.setDetailedText(details)
        q.addButton('OK', QMessageBox.AcceptRole)
        return q.exec()

    def loadErrorBox(self, file, err):
        "Prepare an error dialog for failure to load a file"
        q = QMessageBox(self)
        q.setIcon(QMessageBox.Warning)
        q.setWindowTitle("Error loading file")
        q.setText("Failed to load '"+file+"':\n"+err[0])
        q.setTextFormat(Qt.PlainText)
        q.setDetailedText(err[1])
        return q

    def getLoadSaveFileName(self, title, filter=None, settingname=None,
                            savesuffix=None, multiple=False):
        "Show a file dialog and select one or more files"
        setting = self.settings.value(settingname, None) if settingname is not None else None
        directory = setting if type(setting) is str else None
        dialog = QFileDialog(parent=self, caption=title,
                             directory=directory, filter=filter)
#        if setting and type(setting) is not str:
#            dialog.restoreState(setting)
        dialog.setOption(QFileDialog.DontUseNativeDialog, True)
        if savesuffix is not None:
            dialog.setAcceptMode(QFileDialog.AcceptSave)
            dialog.setDefaultSuffix(savesuffix)
        elif multiple:
            dialog.setFileMode(QFileDialog.ExistingFiles)
        else:
            dialog.setFileMode(QFileDialog.ExistingFile)
        dialog.exec()
        files = dialog.selectedFiles()
        if not dialog.result() or not files:
            return None
        self.settings.setValue(settingname, os.path.dirname(files[0]))
        return files if multiple else files[0]

    def getSaveFileName(self, title, suffix='', filter=None, settingname=None):
        "Show a file dialog and select an output file"
        return self.getLoadSaveFileName(title=title, filter=filter,
                                        settingname=settingname, savesuffix=suffix)

    def getLoadFileName(self, title, filter=None, settingname=None):
        "Show a file dialog and select an input file"
        return self.getLoadSaveFileName(title=title, filter=filter,
                                        settingname=settingname)

    def getLoadFileNames(self, title, filter=None, settingname=None):
        "Show a file dialog and select an input file"
        return self.getLoadSaveFileName(title=title, filter=filter,
                                        settingname=settingname, multiple=True)

    def getImageFileName(self, title, settingname=None):
        "Show a file dialog and select a single image file"
        return self.getLoadSaveFileName(
                title=title,
                filter="Image files (*.jpg *.jpeg *.png *.tif *.tiff *.bmp *.gif);;All files (*)",
                settingname=settingname)

    def getDirectoryName(self, title, settingname=None, savesetting=True):
        "Show a file dialog and select a directory"
        directory = self.settings.value(settingname, None) if settingname is not None else None
        dialog = QFileDialog(parent=self, caption=title,
                             directory=directory)

        dialog.setAcceptMode(QFileDialog.AcceptSave)
        dialog.setFileMode(QFileDialog.Directory)
        dialog.setOption(QFileDialog.DontUseNativeDialog, True)
        dialog.setOption(QFileDialog.ShowDirsOnly, True)
        dialog.exec()
        dirs = dialog.selectedFiles()
        if not dialog.result() or not dirs:
            return None
        if savesetting:
            self.settings.setValue(settingname, dirs[0])
        return dirs[0]

    def updateWavenumberRange(self):
        "For derived classes, when the wn range is updated"
        pass

    def updateDimensions(self, wh):
        "For derived classes, when image width/height is updated"
        pass


    @classmethod
    def run_octavvs_application(windowclass, parser=None, parameters=[], isChild=False):
        res = 1
        try:
            progver = 'OCTAVVS %s %s' % (windowclass.program_name(), OctavvsMainWindow.octavvs_version)
            windowparams = {}
            if parser is not None:
                parser.add_argument('--version', action='version', version=progver)
                parser.add_argument('--mpmethod')
                args = parser.parse_args()
                windowparams = { k: args.__dict__[k] for k in parameters }
                if args.mpmethod and multiprocessing.get_start_method(allow_none=True) is None:
                    multiprocessing.set_start_method(args.mpmethod)

            app = QApplication.instance()
            if not app:
                app = QApplication(sys.argv)
            add_clipboard_to_figures()
            window = windowclass(**windowparams)
            window.show()
            if not isChild:
                qApp.lastWindowClosed.connect(qApp.quit);
                res = app.exec_()

        except Exception:
            traceback.print_exc()
            print('Press some key to quit')
            input()
        if not isChild :
            sys.exit(res)

