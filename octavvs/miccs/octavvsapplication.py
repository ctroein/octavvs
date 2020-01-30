#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 21:07:55 2020

@author: carl
"""

import sys
import traceback

from PyQt5.QtWidgets import QMainWindow
from PyQt5.QtWidgets import QApplication, QFileDialog, QErrorMessage, QInputDialog, QDialog, QMessageBox
from PyQt5.QtCore import Qt, QSettings


class OctavvsMainWindow(QMainWindow):

    fileOptions = QFileDialog.Options() | QFileDialog.DontUseNativeDialog


    def __init__(self, name, parent=None, files=None):
        super().__init__(parent)
        self.programName = name
        self.settings = QSettings('MICCS', 'OCTAVVS ' + name)


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

    def getSaveFile(self, title, filter, directory, suffix):
        saveDialog = QFileDialog(parent=self, caption=title, directory=directory, filter=filter)
        saveDialog.setAcceptMode(QFileDialog.AcceptSave)
        saveDialog.setDefaultSuffix(suffix)
        saveDialog.exec()
        file = saveDialog.selectedFiles()
        return file[0] if len(file) == 1 else None


def run_octavvs_application(class_, **kwargs):
    try:
        app = QApplication.instance()
        if not app:
            app = QApplication(sys.argv)
        window = class_(**kwargs)
        window.show()
        res = app.exec_()
    except Exception:
        traceback.print_exc()
        print('Press some key to quit')
        input()
    sys.exit(res)

