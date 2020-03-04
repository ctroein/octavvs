#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 14:45:19 2019

@author: carl
"""

import sys
import traceback
from PyQt5.QtWidgets import QMessageBox
from PyQt5.QtCore import Qt

class ExceptionDialog:
    def install(qwin):
        "installs a graphical exception handler that shows a message and doesn't terminate"
        def ehook(exctype, value, tb):
            err = 'Unhandled exception: ' + repr(value)
            details = ''.join(traceback.format_exception(exctype, value, tb))
            print(err, '\n', details)
    
            q = QMessageBox(qwin)
            q.setIcon(QMessageBox.Critical)
            q.setWindowTitle("Error")
            q.setText(err)
            q.setTextFormat(Qt.PlainText)
            q.setDetailedText(str(details))
            q.addButton('OK', QMessageBox.AcceptRole)
            return q.exec()
    
        oldhook = sys.excepthook
        sys.excepthook = ehook
        return oldhook

