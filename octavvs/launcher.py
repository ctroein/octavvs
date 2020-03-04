#!/usr/bin/python
# -*- coding: utf-8 -*-
"""
Author: Stéphan Pissot
A minimalistic launcher to start the tools
"""
import sys
from PyQt5.Qt import qApp
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QLabel
from octavvs import preprocessing, clustering, mcr_als

def start_preprocessing():
    preprocessing.MyMainWindow.run_octavvs_application(isChild=True)
def start_clustering():
    clustering.MyMainWindow.run_octavvs_application(isChild=True)
def start_mcrals():
    mcr_als.MyMainWindow.run_octavvs_application(isChild=True)

def main():
   app = QApplication(sys.argv)
   widget = QWidget()

   label=QLabel(widget)
   label.setText('OCTAVVS')
   label.move(10,10)

   button1 = QPushButton(widget)
   button1.setText("Preprocessing")
   button1.move(10,42)
   button1.clicked.connect(start_preprocessing)

   button2 = QPushButton(widget)
   button2.setText("Clustering")
   button2.move(10,72)
   button2.clicked.connect(start_clustering)

   button3 = QPushButton(widget)
   button3.setText("MCR-ALS")
   button3.move(10,102)
   button3.clicked.connect(start_mcrals)

   widget.setWindowTitle("OCTAVVS Launcher")
   widget.show()
   qApp.lastWindowClosed.connect(qApp.quit);
   sys.exit(app.exec_())

#if __name__ == '__main__':
#    main()
