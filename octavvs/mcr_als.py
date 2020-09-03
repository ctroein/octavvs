# -*- coding: utf-8 -*-
"""
@author: syahr
"""
import gc
#import sys
import csv
import glob
import os
from sklearn.cluster import MiniBatchKMeans
#import traceback
from os.path import basename, dirname
from datetime import datetime
from pkg_resources import resource_filename
import argparse
from scipy.optimize import nnls as nnls

from PyQt5.QtWidgets import QFileDialog, QMessageBox,  QDialog
from PyQt5.Qt import QMainWindow,qApp
from PyQt5.QtCore import QThread, pyqtSignal, pyqtSlot #QCoreApplication, QObject, QRunnable, QThreadPool
from PyQt5 import uic

import matplotlib
matplotlib.use('QT5Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error as msea
import scipy as sc
from skimage.draw import polygon
import pickle

# from .pymcr_new.regressors import OLS, NNLS
# from .pymcr_new.constraints import ConstraintNonneg, ConstraintNorm
from .mcr import ftir_function as ff
from octavvs.algorithms import correction as mc
from octavvs.ui import (FileLoader, ImageVisualizer, OctavvsMainWindow, NoRepeatStyle, uitools)

Ui_MainWindow = uic.loadUiType(resource_filename(__name__, "mcr/mcr_final_loc.ui"))[0]
Ui_MainWindow2 = uic.loadUiType(resource_filename(__name__, "mcr/mcr_roi_sub.ui"))[0]
Ui_DialogAbout = uic.loadUiType(resource_filename(__name__, "mcr/about.ui"),
                                from_imports=True, import_from='octavvs')[0]

class DialogAbout(QDialog, Ui_DialogAbout):
    def __init__(self, parent=None):
        QDialog.__init__(self, parent)
        qApp.installEventFilter(self)
        self.setupUi(self)

class Second(QMainWindow, Ui_MainWindow2):
    def __init__(self, parent=None):
        super(Second, self).__init__(parent=None)
        qApp.installEventFilter(self)
        self.setupUi(self)
        self.pushButtonClose.clicked.connect(self.close)
        self.pushButtonReset.clicked.connect(self.Plot_ROI.resetAll)
        self.pushButtonRemove.clicked.connect(self.Plot_ROI.removeOne)
        self.comboBox_roi.currentIndexChanged.connect(self.ImageProjection)
        self.mainwin = parent

    @pyqtSlot()
    def ImageProjection(self):
        img = self.mainwin.plot_whitelight.img
        self.comboBox_roi.setEnabled(img is not None)
        if img is None:
            self.comboBox_roi.setCurrentIndex(0)

        self.Plot_ROI.setSize(self.mainwin.projection.shape)
        if self.comboBox_roi.currentIndex() == 0:
            self.Plot_ROI.setImage(self.mainwin.projection, self.mainwin.comboBoxCmaps.currentText())
        else:
            self.Plot_ROI.setImage(img)

    @pyqtSlot(str)
    def setCmap(self, cmap):
        self.Plot_ROI.setCmap(cmap)

    @pyqtSlot()
    def resetAll(self):
        self.Plot_ROI.resetAll()
        self.hide()


class MyMainWindow(OctavvsMainWindow, Ui_MainWindow):

    projectionUpdated = pyqtSignal()
    loadedFile = pyqtSignal()

    @classmethod
    def program_name(cls):
        "Return the name of the program that this main window represents"
        return 'MCR-ALS'

    def __init__(self, parent=None):
        super().__init__(parent)
        self.default_dir =  os.path.dirname(__file__) + "\\mcr\\"+'dat.initi'
        self.lineEditSuffix.setText('_purest')
        self.pushButtonLocal.setEnabled(False)
        self.pushButtonLoad.clicked.connect(self.Load)
        self.lock_all(False)
        self.progressBar.hide()
        self.pushButtonStop.hide()
        self.comboBoxMethod.currentIndexChanged.connect(self.ImageProjection)
        self.horizontalSliderWavenumber.valueChanged.connect(self.Wavenumbercal)
        self.comboBoxCmaps.currentTextChanged.connect(self.plot_visual.setCmap)

        self.comboBoxCmaps.currentTextChanged.connect(self.ImageProjection)
        self.loadedFile.connect(self.plot_visual.clearMarkings)

        self.pushButtonSVD.clicked.connect(self.SVDprocess)
        self.comboBoxInitial.currentIndexChanged.connect(self.InitialCondition)
        self.pushButtonPurestCal.clicked.connect(self.run)
        self.checkBoxSaveInit.toggled.connect(self.SaveInit)
        self.checkBoxSavePurest.toggled.connect(self.SavePurest)
        self.pushButtonExpandSpectra.clicked.connect(self.ExpandSpec)
        self.pushButtonExpandProjection.clicked.connect(self.ExpandProj)
        self.pushButtonExpandSVD.clicked.connect(self.ExpandSVD)
        self.pushButtonExpandInitSpect.clicked.connect(self.ExpandInitSpect)
        self.pushButtonExpandPurConc.clicked.connect(self.ExpandPurConc)
        self.pushButtonExpandPurSp.clicked.connect(self.ExpandPurSp)
        self.pushButtonWhitelight.clicked.connect(self.WhiteRead)
        self.pushButtonStop.clicked.connect(self.killer)
        self.lineEditHeight.editingFinished.connect(self.ValidationX)
        self.lineEditWidth.editingFinished.connect(self.ValidationY)

        self.pushButtonLocal.clicked.connect(self.roi)

        self.projection = None
        self.roiDialog = Second(self)
        self.roiDialog.Plot_ROI.captured.connect(self.setRoi)
        self.roiDialog.Plot_ROI.captured.connect(self.plot_visual.setRoi)
        self.projectionUpdated.connect(self.roiDialog.ImageProjection)
        self.projectionUpdated.connect(self.ExpandProjU)

        self.comboBoxCmaps.currentTextChanged.connect(self.roiDialog.setCmap)
        self.loadedFile.connect(self.roiDialog.resetAll)

        self.checkBoxInvert.toggled.connect(self.Invert)
        self.About = DialogAbout()

        self.actionAbout.triggered.connect(self.About.show)
        self.roiDialog.actionAbout.triggered.connect(self.About.show)
        self.comboBoxImp.currentIndexChanged.connect(self.VisSpectra)
        self.spinBoxWlength.valueChanged.connect(self.VisSpectra)
        self.spinBoxPoly.valueChanged.connect(self.VisSpectra)

        

        self.spinBoxWlength.hide()
        self.spinBoxPoly.hide()
        self.labelJ.hide()
        self.labelWl.hide()
        self.labelPl.hide()



        self.post_setup()




    def closeEvent(self, event):
        msgBox = QMessageBox()
        msgBox.setIcon(QMessageBox.Question)
        msgBox.setText("Warning")
        msgBox.setInformativeText('Are you sure to close the window ?')
        msgBox.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        msgBox.setDefaultButton(QMessageBox.No)
        reply = msgBox.exec_()
        if reply == QMessageBox.Yes:
            plt.close('all')
            self.roiDialog.close()
            self.About.close()
            if hasattr(self, 'calpures'):
                self.calpures.stop()
#            self.killer_renew()
#            qApp.quit()
        else:
            event.ignore()


    def roi(self):
        self.comboBoxInitial.setCurrentIndex(0)
        self.roiDialog.show()

    def Load(self):
        try:
            infile = open(self.default_dir,'rb')
            dire = pickle.load(infile)
            infile.close()
        except:
            dire = os.path.dirname(__file__) 
        
        if self.comboBoxSingMult.currentIndex() == 1:
            self.pushButtonLocal.setEnabled(False)
            options = QFileDialog.Options()
            options |= QFileDialog.DontUseNativeDialog
            name,_ = QFileDialog.getOpenFileName(self,"Open Matrix File", dire,"Matrix File (*.mat)", options=options)
            self.foldername = dirname(name)
            if self.foldername:
                outfile = open(self.default_dir,'wb')
                pickle.dump(self.foldername,outfile)
                outfile.close()
                self.progressBar.show()
                self.search_whole_folder(self.foldername)
            else:
                msg = QMessageBox()
                msg.setIcon(QMessageBox.Critical)
                msg.setText('No file is loaded')
                msg.setInformativeText("Please select a file")
                msg.setWindowTitle("Warning")
                msg.setStandardButtons(QMessageBox.Ok )
                msg.exec_()

        elif self.comboBoxSingMult.currentIndex() == 0:
            self.pushButtonLocal.setEnabled(True)
            self.progressBar.hide()
            options = QFileDialog.Options()
            options |= QFileDialog.DontUseNativeDialog
            filename, _ = QFileDialog.getOpenFileName(self,"Open Matrix File", dire,"Matrix File (*.mat)", options=options)
            if filename:
                outfile = open(self.default_dir,'wb')
                pickle.dump(dirname(filename),outfile)
                outfile.close()
                
                self.allnames = [filename]
                self.coord = []
                self.clear_prev()
                self.lineEditTotal.setText(str(1))
                self.initialization(filename)
                self.lineEditFileNumber.setText(str(1))

        elif self.comboBoxSingMult.currentIndex() == 2:
            options = QFileDialog.Options()
            options |= QFileDialog.DontUseNativeDialog
            fileNames, _ = QFileDialog.getOpenFileNames(self,"Open Matrix File", dire,"Matrix File (*.mat)", options=options)
            if fileNames:
                outfile = open(self.default_dir,'wb')
                pickle.dump(dirname(fileNames[0]),outfile)
                outfile.close()

                count = 0
                name =  {}
                foldername = dirname(fileNames[0])
                for file in fileNames:
                    name[count] = file
                    count += 1
                # count = 0

                w = csv.writer(open(foldername+"//Fileall.csv", "w"))
                for key, val in sorted(name.items(), key=lambda item: item[1]):
#                    for key, val in sorted(name.items()):
                    w.writerow([key, val])

                self.allnames = name
                self.nfiles = count
                self.lineEditTotal.setText(str(count))
                self.lineEditFileNumber.setText(str(1))
                self.initialization(name[0])
                self.SVDprocess()


    def initialization(self,fileName):

        self.lineEditFilename.setText(basename(fileName))
        self.labelDirectory.setText(dirname(fileName))

        try:
            self.lock_all(True)
            self.sx, self.sy, self.pin,self.wavenumber, self.spo = ff.readmat(fileName)
            self.spo = mc.nonnegative(self.spo)


            self.lineEditLength.setText(str(len(self.wavenumber)))
            self.labelMinwn.setText(str("%.2f" % np.min(self.wavenumber)))
            self.labelMaxwn.setText(str("%.2f" % np.max(self.wavenumber)))
            self.lineEditWavenumber.setText(str("%.2f" % np.min(self.wavenumber)))

            kmeans = MiniBatchKMeans(n_clusters=8, random_state=0).fit(self.spo.T)
            self.raw = kmeans.cluster_centers_
            self.raw = self.raw.T


            self.VisSpectra()
            self.Select_spectra()


        except:
            # This exception handling must be made much more specific. What exception, what lines?
            raise
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Critical)
            msg.setText('The .mat file is not FTIR File')
            msg.setInformativeText("Please select another file !")
            msg.setWindowTitle("Warning")
            msg.setStandardButtons(QMessageBox.Ok )
            msg.exec_()

        self.plot_whitelight.load(fileName.replace(fileName.split('.0')[-1],'.jpg'))
#        self.plot_whitelight.load(os.path.splitext(fileName)[0]+'.jpg')
        self.loadedFile.emit()



    def Select_spectra(self):
        win = int(self.spinBoxWlength.value())
        pol = int(self.spinBoxPoly.value())
        if self.comboBoxImp.currentIndex() == 0:
            self.sp = mc.nonnegative(self.spo)
            self.p = self.sp.reshape(np.shape(self.pin),order='C')
        elif self.comboBoxImp.currentIndex() == 1:
            self.sp = sc.signal.savgol_filter(self.spo.T,win, polyorder = pol,deriv=1)
            self.sp = self.sp.T
            self.sp = mc.nonnegative(self.sp)
            self.p = self.sp.reshape(np.shape(self.pin),order='C')

        elif self.comboBoxImp.currentIndex() == 2:
            self.sp = sc.signal.savgol_filter(self.spo.T,win, polyorder = pol,deriv=2)
            self.sp = self.sp.T
            self.sp = mc.nonnegative(self.sp)
            self.p = self.sp.reshape(np.shape(self.pin),order='C')

        try:
            x = int(self.lineEditHeight.text())
            y = int(self.lineEditWidth.text())
            z = int(self.lineEditLength.text())
            self.p = np.reshape(self.p,(z,x,y))
        except ValueError:
            self.lineEditWidth.setText(str(self.sx))
            self.lineEditHeight.setText(str(self.sy))

        self.ImageProjection()




    def VisSpectra(self):
        self.Select_spectra()
        win = int(self.spinBoxWlength.value())
        pol = int(self.spinBoxPoly.value())

        if self.comboBoxImp.currentIndex() == 0:
            self.yvis = self.raw.copy()
            self.spinBoxWlength.hide()
            self.spinBoxPoly.hide()
            self.labelJ.hide()
            self.labelWl.hide()
            self.labelPl.hide()


        elif self.comboBoxImp.currentIndex() == 1:
            self.yvis= sc.signal.savgol_filter(self.raw.T,win, polyorder = pol,deriv=1)
            self.yvis= self.yvis.T
            self.spinBoxWlength.show()
            self.spinBoxPoly.show()
            self.labelJ.show()
            self.labelWl.show()
            self.labelPl.show()


        else:
            self.yvis= sc.signal.savgol_filter(self.raw.T,win, polyorder = pol,deriv=2)
            self.yvis= self.yvis.T

            self.spinBoxWlength.show()
            self.spinBoxPoly.show()
            self.labelJ.show()
            self.labelWl.show()
            self.labelPl.show()


        self.yvis = mc.nonnegative(self.yvis)
        self.plot_specta.canvas.ax.clear()
        self.plot_specta.canvas.ax.plot(self.wavenumber,self.yvis)
        if self.checkBoxInvert.isChecked():
            self.plot_specta.Invert()
        self.plot_specta.canvas.fig.tight_layout()
        self.plot_specta.canvas.draw()

        self.ExpandSpecU()

    def Invert(self):
        self.ExpandSpecU()
        self.ExpandPurSpU()
        # self.ExpandInitSpectU()
        
        self.plot_specta.Invert()
        self.plot_specta.canvas.fig.tight_layout()
        self.plot_specta.canvas.draw()

        self.plotPurestSpectra.Invert()
        self.plotPurestSpectra.canvas.fig.tight_layout()
        self.plotPurestSpectra.canvas.draw()

        if self.comboBoxMethod.currentIndex() == 2:
            self.Wavenumbercal()

        if hasattr(self,'nr') :
            if self.comboBoxInitial.currentIndex() == 0:
                self.InitialVis()



    def ValidationX(self):
        x = int(self.lineEditHeight.text())
        y = int(self.lineEditWidth.text())
        z = int(self.lineEditLength.text())
        xy = int(self.sx*self.sy)

        if x == 0 or y == 0:
            self.lineEditHeight.setText(str(self.sy))
            self.lineEditWidth.setText(str(self.sx))
            x = self.sx
            y = self.sy

        elif int(x*y) != xy:
            excess = np.mod(xy,x)
            if excess == 0 :
                y=xy/x
                y = int(y)
                self.lineEditWidth.setText(str(y))
            else:
                self.lineEditHeight.setText(str(self.sy))
                self.lineEditWidth.setText(str(self.sx))
                x = self.sx
                y = self.sy
        else:
            self.lineEditHeight.setText(str(x))
            self.lineEditWidth.setText(str(y))

        self.p = np.reshape(self.p,(z,x,y))
        self.ImageProjection()

    def ValidationY(self):
        x = int(self.lineEditHeight.text())
        y = int(self.lineEditWidth.text())
        z = int(self.lineEditLength.text())
        xy = int(self.sx*self.sy)

        if x == 0 or y == 0:
            self.lineEditHeight.setText(str(self.sy))
            self.lineEditWidth.setText(str(self.sx))
            x = self.sx
            y = self.sy

        elif int(x*y) != xy:
            excess = np.mod(xy,y)
            if excess == 0:
                x=xy/y
                x = int(x)
                self.lineEditHeight.setText(str(x))
            else:
                self.lineEditHeight.setText(str(self.sy))
                self.lineEditWidth.setText(str(self.sx))
                x = self.sx
                y = self.sy
        else:
            self.lineEditHeight.setText(str(x))
            self.lineEditWidth.setText(str(y))

        self.p = np.reshape(self.p,(z,x,y))
        self.ImageProjection()




    def search_whole_folder(self, foldername):
        count = 0
        name =  {}
        a = [x[0] for x in os.walk(foldername)]
        for i in a:
            os.chdir(i)
            for file in glob.glob('*.mat'):
                name[count] = str(i+'/'+file)
                count += 1


        if count != 0:
            w = csv.writer(open(foldername+"//Fileall.csv", "w"))
            for key, val in sorted(name.items(), key=lambda item: item[1]):
#            for key, val in sorted(name.items()):
                w.writerow([key, val])

            self.allnames = name
            self.nfiles = count
            self.lineEditTotal.setText(str(count))
            self.lineEditFileNumber.setText(str(1))
            self.initialization(name[0])
            self.SVDprocess()
        else:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Critical)
            msg.setText('The .mat file does not exist in this directory')
            msg.setInformativeText("Please select another directory !")
            msg.setWindowTitle("Warning")
            msg.setStandardButtons(QMessageBox.Ok )
            msg.exec_()

    def WhiteRead(self):
       options = QFileDialog.Options()
       options |= QFileDialog.DontUseNativeDialog
       white, _ = QFileDialog.getOpenFileName(self,"Open White Light Image", "","images(*.jpg *.png)", options=options)
       if white:
           self.plot_whitelight.load(white)
           self.projectionUpdated.emit()

    def ImageProjection(self):

        if  self.comboBoxMethod.currentIndex() == 0:
            self.horizontalSliderWavenumber.setEnabled(False)
            self.projection = ff.proarea(self.p,self.wavenumber)
        if  self.comboBoxMethod.currentIndex() == 1:
            self.lineEditWavenumber.setEnabled(False)
            self.horizontalSliderWavenumber.setEnabled(False)
            self.projection = ff.promip(self.p)
        if  self.comboBoxMethod.currentIndex() == 2:
            self.lineEditWavenumber.setEnabled(True)
            self.horizontalSliderWavenumber.setEnabled(True)
            self.wavenumv = float(self.lineEditWavenumber.text())
            self.projection = ff.prowavenum(self.p,self.wavenumber,self.wavenumv)

        self.plot_visual.setImage(self.projection, self.comboBoxCmaps.currentText())

#        self.plot_visual.canvas.ax.clear()
#        self.plot_visual.canvas.ax.imshow(self.projection,str(self.comboBoxCmaps.currentText()))
#        self.plot_visual.canvas.fig.tight_layout()
#        self.plot_visual.canvas.draw()

        self.projectionUpdated.emit()


    def ExpandProj(self):
        nr = self.spinBoxSVDComp.value()
        fig = plt.figure("Image Projection",tight_layout={'pad':.5})
        fig.clear()
        ax = fig.gca()
        ax.imshow(self.projection,str(self.comboBoxCmaps.currentText()))
        plt.colorbar(ax.imshow(self.projection,str(self.comboBoxCmaps.currentText())))
        if len(self.splot ) != 1 :
            for j in range(0,nr):
                ax.plot(self.pos[j,0],self.pos[j,1],marker='p', color = 'black')
        fig.show()


    @pyqtSlot()
    def ExpandProjU(self):
        nr = self.spinBoxSVDComp.value()
        if plt.fignum_exists("Image Projection"):
            fig = plt.figure("Image Projection",tight_layout={'pad':.5})
            fig.clear()
            ax = fig.gca()
            ax.imshow(self.projection,str(self.comboBoxCmaps.currentText()))
            plt.colorbar(ax.imshow(self.projection,str(self.comboBoxCmaps.currentText())))
            if len(self.splot ) != 1 :
                for j in range(0,nr):
                    ax.plot(self.pos[j,0],self.pos[j,1],marker='p', color = 'black')
            fig.canvas.draw_idle()
        else:
            pass



    def ExpandSVD(self):
        if len(self.splot ) != 1 :
            # plt.close("SVD Plot")
            fig = plt.figure("SVD Plot",tight_layout={'pad':.5})
            fig.clear()
            ax = fig.gca() 
            ax.plot(self.xplot,self.splot,'-o')
            fig.show()
        else:
            pass


    def ExpandSVDU(self):
        if plt.fignum_exists("SVD Plot"):
            fig = plt.figure("SVD Plot",tight_layout={'pad':.5})
            ax = fig.gca()
            ax.clear()
            ax.plot(self.xplot,self.splot,'-o')
            fig.canvas.draw_idle()
        else:
            pass


    def ExpandInitSpect(self):
        # plt.close("Initial")
        fig = plt.figure("Initial",tight_layout={'pad':.5})
        ax = fig.gca() 
        if len(self.insp) != 1:
            if self.comboBoxInitial.currentIndex() == 0:
                ax.plot(self.wavenumber,self.insp.T)
                if self.checkBoxInvert.isChecked():
                    ax.invert_xaxis()
            else:
                ax.plot(np.arange(self.sx*self.sy),self.incon.T)
            fig.show()


    def ExpandInitSpectU(self,x,y):
        if plt.fignum_exists("Initial"):
            fig = plt.figure("Initial",tight_layout={'pad':.5})
            ax = fig.gca()
            ax.clear()
            ax.plot(x,y)
            if self.checkBoxInvert.isChecked():
                ax.invert_xaxis()
            fig.canvas.draw_idle()
        else:
            pass



    def Wavenumbercal(self):
        nnow1 = ((np.max(self.wavenumber) - np.min(self.wavenumber))
        *float(self.horizontalSliderWavenumber.value())/10000.0 + np.min(self.wavenumber))
        nnow1 = "%.2f" % nnow1
        self.lineEditWavenumber.setText(str(nnow1))
        self.wavenumv = float(self.lineEditWavenumber.text())
        self.projection = ff.prowavenum(self.p,self.wavenumber,self.wavenumv)

        self.plot_specta.canvas.ax.clear()
        self.plot_specta.canvas.ax.plot(self.wavenumber,self.yvis)
        self.plot_specta.canvas.ax.axvline(x=self.wavenumv)
        if self.checkBoxInvert.isChecked():
            self.plot_specta.Invert()
        self.plot_specta.canvas.fig.tight_layout()
        self.plot_specta.canvas.draw()

        self.plot_visual.setImage(self.projection, self.comboBoxCmaps.currentText())
        self.projectionUpdated.emit()
        self.ExpandSpecU()


    def ExpandSpec(self):
        # plt.close("Spectra")
        fig = plt.figure("Spectra",tight_layout={'pad':.5})
        ax = fig.gca()
        ax.clear()
        ax.plot(self.wavenumber,self.yvis)
        if self.comboBoxMethod.currentIndex() == 2:
                ax.axvline(x=self.wavenumv)
        if self.checkBoxInvert.isChecked():
            ax.invert_xaxis()

        ax.set_xlabel("Wavenumber(1/cm)",fontsize=14)
        ax.set_ylabel("Absorption(arb. units)",fontsize=14)
        ax.tick_params(axis='both',direction='in', length=8, width=1)
        ax.tick_params(axis='both',which='major',labelsize=14)
        fig.show()


    def ExpandSpecU(self):
        if plt.fignum_exists("Spectra"):
            fig = plt.figure("Spectra",tight_layout={'pad':.5})
            ax = fig.gca()
            ax.clear()
            ax.plot(self.wavenumber,self.yvis)
            if self.comboBoxMethod.currentIndex() == 2:
                ax.axvline(x=self.wavenumv)
            if self.checkBoxInvert.isChecked():
                ax.invert_xaxis()
            fig.canvas.draw_idle()
        else:
            pass



    def ExpandPurConc(self):
        if len(self.copt)  != 1:
            # plt.close("Purest Concentrations")
            fig = plt.figure("Purest Concentrations",tight_layout={'pad':.5})
            ax = fig.subplots()
            ax.plot(np.arange(len(self.copt)),self.copt)
            fig.show()

    def ExpandPurConcU(self,copt):
        if plt.fignum_exists("Purest Concentrations"):
            fig = plt.figure("Purest Concentrations",tight_layout={'pad':.5})
            ax = fig.gca()
            ax.clear()
            ax.plot(np.arange(len(copt)),copt)
            fig.canvas.draw_idle()
        else:
            pass


    def ExpandPurSp(self):
        if len(self.sopt) != 1:
            # plt.close("Purest Spectra")
            fig = plt.figure("Purest Spectra",tight_layout={'pad':.5})
            ax = fig.subplots()
            if self.checkBoxInvert.isChecked():
                ax.invert_xaxis()
            ax.plot(self.wavenumber,self.sopt)
            fig.show()

    def ExpandPurSpU(self):
        if plt.fignum_exists("Purest Spectra") and len(self.sopt) == len(self.wavenumber):
            fig = plt.figure("Purest Spectra",tight_layout={'pad':.5})
            ax = fig.gca()
            ax.clear()
            ax.plot(self.wavenumber, self.sopt)
            if self.checkBoxInvert.isChecked():
                ax.invert_xaxis()
            fig.canvas.draw_idle()
        else:
            pass

    def SVDprocess(self):
        win = int(self.spinBoxWlength.value())
        pol = int(self.spinBoxPoly.value())

        self.nr = self.spinBoxSVDComp.value()
        if self.nr < 20:
            nplot = self.nr+5
        else:
            nplot = self.nr

        if self.comboBoxImp.currentIndex() == 1:
            self.sp = sc.signal.savgol_filter(self.spo.T,win, polyorder = pol,deriv=1)
            self.sp = self.sp.T

        elif self.comboBoxImp.currentIndex() == 2:
            self.sp= sc.signal.savgol_filter(self.spo.T,win, polyorder = pol,deriv=2)
            self.sp= self.sp.T
        else:
            self.sp = self.spo.copy()




        if not self.coord:
            self.sp = mc.nonnegative(self.sp)
            self.u, self.s, self.v = np.linalg.svd(self.sp)

            self.xplot = np.arange(nplot)
            self.splot =self.s[0:nplot]
            self.SVDPlot()
        else:
            nx, ny = int(self.lineEditWidth.text()),int(self.lineEditHeight.text())

            self.roi = np.zeros((ny, nx))
            vertex_col_coords,vertex_row_coords = np.array(self.coord).T
            fill_row_coords, fill_col_coords = polygon(
                    vertex_row_coords, vertex_col_coords, self.roi.shape)
            self.roi[fill_row_coords, fill_col_coords] = 1
            self.rem = self.roi * self.projection
            img_d = np.reshape(self.rem,(nx*ny,1))
            self.ind =  np.where(img_d > 0)[0]

            sp_new = mc.nonnegative(self.sp[:,self.ind])
            self.u, self.s, self.v = np.linalg.svd(sp_new)

            nplot = min(nplot, len(self.s))
            self.xplot = np.arange(nplot)
            self.splot =self.s[0:nplot]

            self.plotSVD.canvas.ax.clear()
            self.plotSVD.canvas.ax.plot(self.xplot,self.splot,'-o')
            self.plotSVD.canvas.draw()
            self.ExpandSVDU()

            per = float(self.lineEditNoisePercent.text())
            self.f = per*0.01


            if self.comboBoxInitial.currentIndex() == 0:
                self.incon = [0,0]
                self.labelInitial.setText("Initial Spectra*")
                self.insp, points = ff.initi_simplisma(sp_new,self.nr,self.f)
                self.pos = np.zeros((self.nr,2))

                for i in range(0,self.nr):
                    self.pos[i,0] = self.ind[points[i]] % nx
                    self.pos[i,1] = self.ind[points[i]] // nx
                self.plot_visual.addPoints(self.pos)
                self.InitialVis()


    def InitialVis(self):
        self.plotInitSpec.canvas.ax.clear()
        self.plotInitSpec.canvas.ax.plot(self.wavenumber,self.insp.T)
        if self.checkBoxInvert.isChecked():
            self.plotInitSpec.Invert()
        self.plotInitSpec.canvas.fig.tight_layout()
        self.plotInitSpec.canvas.draw()
        self.ExpandInitSpectU(self.wavenumber,self.insp.T)




    def SVDPlot(self):
        self.plotPurestConc.canvas.ax.clear()
        self.plotPurestConc.canvas.draw()
        self.plotPurestSpectra.canvas.ax.clear()
        self.plotPurestSpectra.canvas.draw()

        self.plotSVD.canvas.ax.clear()
        self.plotSVD.canvas.ax.plot(self.xplot,self.splot,'-o')
        self.plotSVD.canvas.draw()
#        if self.comboBoxPurest.currentIndex() == 0:
        self.ExpandSVDU()
        self.InitialCondition()


    def InitialCondition(self):
        nr = self.spinBoxSVDComp.value()
        per = float(self.lineEditNoisePercent.text())
        self.f = per*0.01

        if self.comboBoxInitial.currentIndex() == 0:
            self.incon = [0,0]
            self.labelInitial.setText("Initial Spectra*")
            self.insp, points = ff.initi_simplisma(self.sp,nr,self.f)
            self.pos = np.array([points % self.projection.shape[0], points // self.projection.shape[1]]).T
            self.plot_visual.setImage(self.projection, self.comboBoxCmaps.currentText())
            self.plot_visual.addPoints(self.pos)
            self.InitialVis()
            self.ExpandProjU()
            if self.checkBoxSaveInit.isChecked():
                self.save_data_init(self.insp.T)


        else:
            self.insp = [0,0]
            self.labelInitial.setText("Initial Concentration*")
            self.incon, __ = ff.initi_simplisma(self.sp.T,nr,self.f)
            self.plotInitSpec.canvas.ax.clear()
            self.plot_visual.clearMarkings()
            self.plotInitSpec.canvas.ax.plot(np.arange(self.sx*self.sy),self.incon.T)
            self.plotInitSpec.canvas.fig.tight_layout()
            self.plotInitSpec.canvas.draw()
            self.ExpandInitSpectU(np.arange(self.sx*self.sy),self.incon.T)
            self.ImageProjection()
            if self.checkBoxSaveInit.isChecked():
                self.save_data_init(self.incon.T)


    def lockmcr(self):
        self.comboBoxInitial.setEnabled(True)
        self.pushButtonExpandInitSpect.setEnabled(True)
        self.comboBoxRegressor.setEnabled(True)
        self.lineEditNoisePercent.setEnabled(True)
        self.checkBoxSaveInit.setEnabled(True)
        self.lineEditPurIter.setText('700')
        # self.SVDprocess()


    def run(self):

        self.plotPurestSpectra.canvas.ax.clear()
        self.plotPurestSpectra.canvas.draw()

        self.plotPurestConc.canvas.ax.clear()
        self.plotPurestConc.canvas.draw()


        if self.checkBoxSavePurest.isChecked():
             if self.comboBoxSingMult.currentIndex() == 0:
                 self.runsingle()
             else:
                 self.runall()

        else:
            msgBox = QMessageBox()
            msgBox.setIcon(QMessageBox.Question)
            msgBox.setText("Warning")
            msgBox.setInformativeText('The Purest Spectra Will not be saved')
            msgBox.setStandardButtons(QMessageBox.Yes| QMessageBox.No)
            msgBox.setDefaultButton(QMessageBox.No)
            reply = msgBox.exec_()
            if reply == QMessageBox.Yes:
                if self.comboBoxSingMult.currentIndex() == 0:
                    self.runsingle()
                else:
                    self.runall()
            else:
                pass


    def runall(self):
        nr = self.spinBoxSVDComp.value()
        f = 0.01*float(self.lineEditNoisePercent.text())
        max_iter = int(self.lineEditPurIter.text())
        stopping_error = float(self.lineEditTol.text())
        init = self.comboBoxInitial.currentIndex()
        win = int(self.spinBoxWlength.value())
        pol = int(self.spinBoxPoly.value())
        implement = int(self.comboBoxImp.currentIndex())


        self.progressBar.setEnabled(True)
        self.progressBar.setMaximum(self.nfiles+1)
        self.lineEditStatus.setText('Multiple files')
        self.calpures = Multiple_Calculation(self.allnames, nr, f,max_iter,
                                             stopping_error, init,win, pol,implement)


        self.calpures.purest.connect(self.finished_single)
        self.calpures.DataInit.connect(self.finished_mcr_all)
        self.calpures.start()
        self.pushButtonStop.show()
        self.pushButtonPurestCal.setEnabled(False)

    def finished_mcr_all(self,count, filename):
        self.progressBar.setValue(count)
        self.lineEditFileNumber.setText(str(count))
        self.lineEditFilename.setText(basename(filename))
        self.initialization(filename)
        self.SVDprocess()


    @pyqtSlot(list)
    def setRoi(self, roi):
        self.coord = roi

    def runsingle(self):
        if not self.coord:
            self.nfiles = 1
            self.runall()
        else:
            self.runsingle_roi()

    # def runsingle_noroi(self):
    #     max_iter = int(self.lineEditPurIter.text())
    #     tol_percent = float(self.lineEditTol.text())
    #     f = 0.01*float(self.lineEditNoisePercent.text())
    #     self.SVDprocess()
    #     nr = self.spinBoxSVDComp.value()
    #     init = self.comboBoxInitial.currentIndex()
    #     self.InitialCondition()
    #     self.calpures = single_report(self.sp,nr,f, max_iter,tol_percent,init)

    #     self.calpures = Multiple_Calculation(self.allnames, nr, f,max_iter,
    #                                          stopping_error, init,win, pol,implement)




    #     self.calpures.purest.connect(self.finished_single)
    #     self.calpures.start()
    #     self.pushButtonStop.show()
    #     self.pushButtonPurestCal.setEnabled(False)



    def killer(self):
        msgBox = QMessageBox()
        msgBox.setIcon(QMessageBox.Question)
        msgBox.setText("Warning")
        msgBox.setInformativeText('Are you sure to terminate the calculation ?')
        msgBox.setStandardButtons(QMessageBox.Yes| QMessageBox.No)
        msgBox.setDefaultButton(QMessageBox.No)
        reply = msgBox.exec_()
        if reply == QMessageBox.Yes:
            self.pushButtonStop.hide()
            self.pushButtonPurestCal.setEnabled(True)
            if hasattr(self, 'calpures'):
                self.calpures.stop()
            self.killer_renew()
        else:
            pass

    def runsingle_roi(self):
        self.comboBoxInitial.setCurrentIndex(0)
        nx, ny = int(self.lineEditWidth.text()),int(self.lineEditHeight.text())
        max_iter = int(self.lineEditPurIter.text())
        tol_percent = float(self.lineEditTol.text())

        self.roi = np.zeros((ny, nx))
        vertex_col_coords, vertex_row_coords = np.array(self.coord).T
        fill_row_coords, fill_col_coords = polygon(
                vertex_row_coords, vertex_col_coords, self.roi.shape)
        self.roi[fill_row_coords, fill_col_coords] = 1
        self.rem = self.roi * self.projection

        img_d = np.reshape(self.rem,(int(nx*ny),1))
        self.ind =  np.where(img_d > 0)[0]

        nr = self.spinBoxSVDComp.value()

        sp_new = mc.nonnegative(self.sp[:,self.ind])
        self.u, self.s, self.v = np.linalg.svd(sp_new)
        if nr < 20:
            nplot = nr+5
        else:
            nplot = nr

        self.xplot = np.arange(nplot)
        self.splot =self.s[0:nplot]

        self.plotPurestConc.canvas.ax.clear()
        self.plotPurestConc.canvas.draw()
        self.plotPurestSpectra.canvas.ax.clear()
        self.plotPurestSpectra.canvas.draw()

        self.plotSVD.canvas.ax.clear()
        self.plotSVD.canvas.ax.plot(self.xplot,self.splot,'-o')
        self.plotSVD.canvas.draw()
        self.ExpandSVDU()

        per = float(self.lineEditNoisePercent.text())
        self.f = per*0.01


        if self.comboBoxInitial.currentIndex() == 0:
            self.incon = [0,0]
            self.labelInitial.setText("Initial Spectra*")
            self.insp, points = ff.initi_simplisma(sp_new,nr,self.f)
            self.InitialVis()
            # self.plotInitSpec.canvas.ax.clear()
            # self.plotInitSpec.canvas.ax.plot(self.wavenumber,self.insp.T)
            # if self.checkBoxInvert.isChecked():
            #     self.plotInitSpec.Invert()
            # self.plotInitSpec.canvas.fig.tight_layout()
            # self.plotInitSpec.canvas.draw()
            if self.checkBoxSaveInit.isChecked():
                self.save_data_init(self.insp.T)
#            self.pos = np.array([points % self.projection.shape[0], points // self.projection.shape[1]]).T
            self.pos = np.zeros((nr,2))

            for i in range(0,nr):
                self.pos[i,0] = self.ind[points[i]] % nx
                self.pos[i,1] = self.ind[points[i]] // nx
                self.plot_visual.addPoints(self.pos)


        nrow, ncol = np.shape(sp_new)

        s = sc.linalg.diagsvd(self.s,nrow, ncol)
        u = self.u[:,0:nr]
        s = s[0:nr,0:nr]
        v = self.v[0:nr,:]
        self.dn = u @ s @ v


        nrow, ncol = np.shape(self.dn)
        dauxt = np.zeros((ncol,nrow))
        aux=self.dn.T

        for i in range(0,ncol):
            dauxt[i,:]=aux[i,:]/np.sqrt(np.sum(aux[i,:]*aux[i,:]))


        f = 0.01*float(self.lineEditNoisePercent.text())
        init = self.comboBoxInitial.currentIndex()



        self.calpures = single_report(sp_new, nr, f, max_iter,tol_percent, init)
        self.calpures.purest.connect(self.finished_single_roi)
        self.calpures.start()
        self.pushButtonStop.show()
        self.pushButtonPurestCal.setEnabled(False)

    def finished_single_roi(self, itern,error,status,copt,sopt):
        self.copt = copt
        self.sopt = sopt

        self.ExpandPurConcU(copt)
        self.ExpandPurSpU()



        nr = self.spinBoxSVDComp.value()
        nx, ny = int(self.lineEditWidth.text()),int(self.lineEditHeight.text())
        bea = np.zeros((int(nx*ny),nr))
        for i in range(0,nr):
            bea[self.ind,i] = copt[:,i]


        self.plotPurestSpectra.canvas.ax.clear()
        self.plotPurestSpectra.canvas.ax.plot(self.wavenumber,sopt)
        if self.checkBoxInvert.isChecked():
            self.plotPurestSpectra.Invert()
        self.plotPurestSpectra.canvas.fig.tight_layout()
        self.plotPurestSpectra.canvas.draw()

        self.plotPurestConc.canvas.ax.clear()
        self.plotPurestConc.canvas.ax.plot(np.arange(len(bea)),bea)
        self.plotPurestConc.canvas.fig.tight_layout()
        self.plotPurestConc.canvas.draw()


        self.lineEdit_Niter.setText(str(itern))
        self.lineEdit_Error.setText(str(round(error,5)))
        self.lineEditStatus.setText(status)


        if (status == 'Max iterations reached') or (status == 'converged'):
            self.pushButtonPurestCal.setEnabled(True)
            self.pushButtonStop.hide()
            if self.checkBoxSavePurest.isChecked():
                self.save_data(bea,sopt)



    def save_data(self, copt, sopt):
        win = int(self.spinBoxWlength.value())
        pol = int(self.spinBoxPoly.value())
        met = int(self.comboBoxImp.currentIndex())
        meta = np.zeros((3,len(sopt.T)))
        meta[0,0] = met
        meta[1,0] = win
        meta[2,0] = pol


        # if self.checkBoxSavePurest.isChecked():
        auxi = np.concatenate((meta,sopt,copt), axis = 0)
        namef = self.lineEditFilename.text()
        namef = namef.replace('.mat','')
        np.savetxt(self.folpurest+'/'+namef+self.lineEditSuffix.text()+'.csv', auxi, delimiter=',')



    def SaveInit(self):
        if self.checkBoxSaveInit.isChecked():
            options = QFileDialog.Options()
            options |= QFileDialog.DontUseNativeDialog
            self.folinit = QFileDialog.getExistingDirectory(self,"Open the input data")
        else:
            pass


    def save_data_init(self, data):
        if self.checkBoxSaveInit.isChecked():
            # auxi = np.concatenate((sopti,copti), axis = 0)
            namef = self.lineEditFilename.text()
            namef = namef.replace('.mat','')
            if self.comboBoxInitial.currentIndex() == 0:
                suf= '_init_spec'
            else:
                suf = '_init_con'
            np.savetxt(self.folinit+'/'+namef+suf+'.csv', data, delimiter=',')


    def SavePurest(self):
        if self.checkBoxSavePurest.isChecked():
            options = QFileDialog.Options()
            options |= QFileDialog.DontUseNativeDialog
            self.folpurest = QFileDialog.getExistingDirectory(self,"Open the input data")
        else:
            pass

    def finished_single(self, itern,error,status,copt,sopt):
        self.copt = copt
        self.sopt = sopt

        self.ExpandPurConcU(copt)
        self.ExpandPurSpU()



        self.plotPurestSpectra.canvas.ax.clear()
        self.plotPurestSpectra.canvas.ax.plot(self.wavenumber,sopt)
        if self.checkBoxInvert.isChecked():
            self.plotPurestSpectra.Invert()
        self.plotPurestSpectra.canvas.fig.tight_layout()
        self.plotPurestSpectra.canvas.draw()

        self.plotPurestConc.canvas.ax.clear()
        self.plotPurestConc.canvas.ax.plot(np.arange(len(copt)),copt)
        self.plotPurestConc.canvas.fig.tight_layout()
        self.plotPurestConc.canvas.draw()

        self.lineEdit_Niter.setText(str(itern))
        self.lineEdit_Error.setText(str(round(error,5)))
        self.lineEditStatus.setText(status)

        if (status == 'Max iterations reached') or (status == 'converged'):
            if self.checkBoxSavePurest.isChecked():
                self.save_data(copt,sopt)
            if self.comboBoxSingMult.currentIndex() == 1 and (int(self.lineEditFileNumber.text())==
                                                             int(self.lineEditTotal.text())) :
                self.pushButtonPurestCal.setEnabled(True)
                self.progressBar.setValue(self.nfiles+1)
                self.lineEditStatus.setText('DONE')
                self.pushButtonStop.hide()
            elif self.comboBoxSingMult.currentIndex() == 0:
                self.pushButtonPurestCal.setEnabled(True)
                # self.progressBar.setValue(self.nfiles+1)
                self.lineEditStatus.setText('DONE')
                self.pushButtonStop.hide()



    def lock_all(self,Stat):
        self.pushButtonExpandSpectra.setEnabled(Stat)
        self.comboBoxMethod.setEnabled(Stat)
        self.pushButtonExpandProjection.setEnabled(Stat)
        self.pushButtonExpandSVD.setEnabled(Stat)
        self.pushButtonExpandInitSpect.setEnabled(Stat)
        self.pushButtonSVD.setEnabled(Stat)
        self.spinBoxSVDComp.setEnabled(Stat)
        self.lineEditPurIter.setEnabled(Stat)
        self.lineEditTol.setEnabled(Stat)
        self.lineEditNoisePercent.setEnabled(Stat)
        self.lineEditStatus.setEnabled(Stat)
        self.checkBoxSaveInit.setEnabled(Stat)
        self.checkBoxSavePurest.setEnabled(Stat)
        # self.pushButtonPurestCal.setEnabled(Stat)
        self.pushButtonExpandPurSp.setEnabled(Stat)
        self.pushButtonExpandPurConc.setEnabled(Stat)
        self.comboBoxCmaps.setEnabled(Stat)
        self.comboBoxInitial.setEnabled(Stat)
        self.comboBoxRegressor.setEnabled(Stat)
        self.lineEditWavenumber.setEnabled(Stat)
        self.lineEditKeyword.setEnabled(Stat)
        self.lineEditSuffix.setEnabled(Stat)
        self.lineEditFilename.setEnabled(Stat)
        self.lineEditFileNumber.setEnabled(Stat)
        self.lineEditTotal.setEnabled(Stat)
        self.lineEditLength.setEnabled(Stat)
        self.lineEditWidth.setEnabled(Stat)
        self.lineEditHeight.setEnabled(Stat)
        self.lineEdit_Niter.setEnabled(Stat)
        self.lineEdit_Error.setEnabled(Stat)


    def clear_prev(self):
        self.plotSVD.canvas.ax.clear()
        self.plotSVD.canvas.draw()
        self.plotInitSpec.canvas.ax.clear()
        self.plotInitSpec.canvas.draw()
        self.plotPurestConc.canvas.ax.clear()
        self.plotPurestConc.canvas.draw()
        self.plotPurestSpectra.canvas.ax.clear()
        self.plotPurestSpectra.canvas.draw()
        self.plot_whitelight.load(None)
        self.lineEditStatus.setText('')

        self.insp = [0]
        self.copt = [0]
        self.sopt = [0]
        self.projection = 0
        self.xplot = [0]
        self.splot = [0]
        self.u = 0
        self.s = 0
        self.v = 0
        self.dn = 0
        self.sx = 0
        self.sy = 0
        self.p = 0
        self.wavenumber = 0


    def killer_renew(self):
        self.plotPurestConc.canvas.ax.clear()
        self.plotPurestConc.canvas.draw()
        self.plotPurestSpectra.canvas.ax.clear()
        self.plotPurestSpectra.canvas.draw()
        self.lineEditStatus.setText('STOP')


#----------------------------------------------------------------------
class Multiple_Calculation(QThread):

    DataInit = pyqtSignal(int, str)
    purest = pyqtSignal(np.int,np.float64,str,np.ndarray, np.ndarray)

    QThread.setTerminationEnabled()
    def __init__(self, filenames, nr, f, max_iter, stopping_error, init,win, pol,implement, parent=None):
        QThread.__init__(self, parent)
        self.foldername = dirname(filenames[0])
        self.nr = nr
        self.max_iter = max_iter
        self.stopping_error = stopping_error
        self.init = init
        self.f = f
        self.max_iter = max_iter
        self.rem = False
        self.win = win
        self.pol = pol
        self.implement = implement
        self.filenames = filenames
        if self.init == 0:
            inguess = 'Spectra'
        else:
            inguess = 'Concentration'

        print(self.foldername)
#----------------------------------------------------------------------------------------
        now = datetime.now()
        date_time = now.strftime("%m/%d/%Y, %H:%M:%S")
        newformat = now.strftime('%m-%d-%Y_%H-%M-%S')
        self.logfile = open(self.foldername+"//logfile_"+newformat+".txt", "w")
        now = datetime.now()
        date_time = now.strftime("%m/%d/%Y, %H:%M:%S")
        self.logfile.write("%s\n" % ("Octavvs Project"))
        self.logfile.write("%s\n" % ("Multivariate Curve Resolution-Alternating Least Square"))
        self.logfile.write("%s\n" % ("Logging started at "+date_time))
        self.logfile.write("%s\n" % ("   Folder: "+self.foldername))
        self.logfile.write("%s\n" % ("   Number of components: "+str(self.nr)))
        self.logfile.write("%s\n" % ("   Noise in SIMPLISMA: "+str(self.f)))
        self.logfile.write("%s\n" % ("   Tolerance: "+str(self.stopping_error)))
        self.logfile.write("%s\n" % ("   Initial Guess: "+ inguess))
        self.logfile.write("%s\n" % ("-------------------------------------------------------"))
#----------------------------------------------------------------------------------------
    def stop(self):
        self.rem = True
        try:
            self.logfile.write("%s\n" % ("calculation was terminated"))
            self.logfile.write("%s\n" % ("-------------------------------------------------------"))
            self.logfile.close()
        except:
            pass

    def logreport(self,filenumber,filename,niter,status,maxfile):
        now = datetime.now()
        date_time = now.strftime("%m/%d/%Y, %H:%M:%S")
        self.logfile.write("%s %s %s %s\n" % (int(filenumber+1),date_time, basename(filename), status+" at "+str(niter)))


        if (filenumber+1) == maxfile:
            self.logfile.close()


    def finished_one(self, filenumber,filename, niter, status, maxfile):
        if status == 'converged' or status =='Max iterations reached':
            self.logreport(filenumber,filename,niter,status,maxfile)


    def solve_lineq(self,A,B):
        if B.ndim ==2:
            N = B.shape[-1]
        else:
            N = 0

        X_ = np.zeros((A.shape[-1], N))
        residual_ = np.zeros((N))

        if N == 0:
            X_, residual_ = nnls(A, B)
        else:
            for num in range(N):
                X_[:, num],residual_[num] = nnls(A, B[:, num])
        return X_

    def normalize(self,dn):
        nrow, ncol = dn.shape
        dauxt = np.zeros((ncol,nrow))
        aux=dn.T
        for i in range(0,ncol):
            dauxt[i,:]=aux[i,:]/np.sqrt(np.sum(aux[i,:]*aux[i,:]))
        return dauxt


    def constraint_norm(self,A):
        A = A.copy()
        A /= A.sum(axis=0)[None, :]
        return A*(A > 0)

    def run(self):
        self.maxfile = len(self.filenames)

        self.count = 0
        for i in range(0,self.maxfile):
            if self.rem:
                status = 'STOP'
                # self.purest.emit(iter,per,status, Cf.T,Sf.T)
                break
            else:

                self.filename = self.filenames[i]
                self.count = self.count + 1
                self.sx, self.sy, self.p ,self.wavenumber, self.spo = ff.readmat(self.filename)
            # self,xplot,splot,sx,sy, p, wavenumber, sp, count
                if self.implement == 1:
                    self.sp = sc.signal.savgol_filter(self.spo.T,self.win, polyorder = self.pol,deriv=1)
                    self.sp = self.sp.T

                elif self.implement == 2:
                    self.sp= sc.signal.savgol_filter(self.spo.T,self.win, polyorder = self.pol,deriv=2)
                    self.sp= self.sp.T
                else:
                    self.sp = self.spo.copy()

                self.sp = mc.nonnegative(self.sp)


                if self.init == 0:
                    insp, points = ff.initi_simplisma(self.sp,self.nr,self.f)
                    C_ = None
                    ST_ = insp
                elif self.init == 1:
                    insp, points = ff.initi_simplisma(self.sp.T,self.nr,self.f)
                    C_ = insp
                    ST_ = None

                u,s,v = np.linalg.svd(self.sp)

                self.DataInit.emit(self.count,self.filename)

                nrow, ncol = np.shape(self.sp)



                s = sc.linalg.diagsvd(s,nrow, ncol)

                u = u[:,0:self.nr]
                s = s[0:self.nr,0:self.nr]
                v = v[0:self.nr,:]
                dn = u @ s @ v


                dauxt = self.normalize(dn)

                for num in range(self.max_iter):
                    iter = num + 1
                    if self.rem:
                        status = 'STOP'
                # self.purest.emit(iter,per,status, Cf.T,Sf.T)
                        break
                    else:

                        if ST_ is not None:
                            Ctemp = self.solve_lineq(ST_.T,dauxt.T)
                            #Constraint
                            # Ctemp = constraint_norm(Ctemp)

                            Dcal = np.dot(Ctemp.T,ST_)
                            error = msea(dauxt,Dcal)

                            if iter ==1:
                                error0 = error.copy()
                                C_ = Ctemp

                            per = 100*abs(error-error0)/error
                            # print(iter,'and',per)
                            if per == 0:
                                Cf = C_.copy()
                                Sf = ST_.copy()
                                ST_temp = ST_.copy()
                                status = 'Iterating'
                                self.purest.emit(iter,per,status, Cf.T,Sf.T)

                            elif  per < self.stopping_error:
                                Cf = C_.copy()
                                Sf = ST_.copy()
                                # Save_data(Cf,Sf)
                                status = 'converged'
                                self.purest.emit(iter,per,status, Cf.T,Sf.T)
                                self.finished_one(i,self.filename,iter,status,self.maxfile)
                                break
                            else:
                                error0 = error.copy()
                                C_ = Ctemp.copy()
                                Sf = ST_.copy()
                                status = 'Iterating'
                                self.purest.emit(iter,per,status, C_.T,Sf.T)

                        if C_ is not None:
                            ST_temp = self.solve_lineq(C_.T,dauxt)
                            #Constraint
                            # ST_temp = constraint_norm(ST_temp)

                            Dcal = np.dot(C_.T,ST_temp)
                            error = msea(dauxt,Dcal)

                            if iter ==1:
                                error0 = error.copy()
                                # C_ = Ctemp
                                ST_ = ST_temp

                            per = 100*abs(error-error0)/error
                            # print(iter,'and',per)
                            if per == 0:
                                Cf = C_.copy()
                                Sf = ST_.copy()
                                status = 'Iterating'
                                self.purest.emit(iter,per,status, Cf.T,Sf.T)

                            elif per < self.stopping_error:
                                Cf = C_.copy()
                                Sf = ST_.copy()
                                # Save_data(Cf,Sf)
                                status = 'converged'
                                self.purest.emit(iter,per,status, Cf.T,Sf.T)
                                self.finished_one(i,self.filename,iter,status,self.maxfile)
                                break

                            else:
                                error0 = error.copy()
                                ST_ = ST_temp.copy()
                                status = 'Iterating'
                                self.purest.emit(iter,per,status, C_.T,ST_.T)

                        if iter == self.max_iter:
                            status = 'Max iterations reached'
                            self.purest.emit(iter,per,status, C_.T,ST_.T)
                            self.finished_one(i,self.filename,iter,status,self.maxfile)
                            break



 #----------------------------------------------------------------------


class single_report(QThread):
    purest = pyqtSignal(np.int,np.float64,str,np.ndarray, np.ndarray)
    def __init__(self, sp, nr, f, niter, stopping_error, init, parent=None
                ):
        QThread.__init__(self, parent)
        """
        Multivariate Curve Resolution - Alternating Least Square
        """
        self.sp = sp
        self.nr = nr
        self.f = f
        self.niter = niter
        self.stopping_error = stopping_error
        self.init = init
        self.rem = False

    def stop(self):
        self.rem = True


    def solve_lineq(self,A,B):
        if B.ndim ==2:
            N = B.shape[-1]
        else:
            N = 0

        X_ = np.zeros((A.shape[-1], N))
        residual_ = np.zeros((N))

        # nnls is Ax = b; thus, need to iterate along
        if N == 0:
            X_, residual_ = nnls(A, B)
        else:
            for num in range(N):
                X_[:, num],residual_[num] = nnls(A, B[:, num])
        return X_

    def normalize(self,dn):
        nrow, ncol = dn.shape
        dauxt = np.zeros((ncol,nrow))
        aux=dn.T
        for i in range(0,ncol):
            dauxt[i,:]=aux[i,:]/np.sqrt(np.sum(aux[i,:]*aux[i,:]))
        return dauxt


    def constraint_norm(self,A):
        A = A.copy()
        A /= A.sum(axis=0)[None, :]
        return A*(A > 0)


    def run(self):
        if self.init == 0:
            insp, points = ff.initi_simplisma(self.sp,self.nr,self.f)
            C_ = None
            ST_ = insp
        elif self.init == 1:
            insp, points = ff.initi_simplisma(self.sp.T,self.nr,self.f)
            C_ = insp
            ST_ = None

        u,s,v = np.linalg.svd(self.sp)
        nrow, ncol = np.shape(self.sp)
        # nr =
        s = sc.linalg.diagsvd(s,nrow, ncol)
        u = u[:,0:self.nr]
        s = s[0:self.nr,0:self.nr]
        v = v[0:self.nr,:]
        dn = u @ s @ v

        dauxt = self.normalize(dn)

        for num in range(self.niter):
            iter = num + 1
            if self.rem:
                status = 'STOP'
                # self.purest.emit(iter,per,status, Cf.T,Sf.T)
                break
            else:

                if ST_ is not None:
                    Ctemp = self.solve_lineq(ST_.T,dauxt.T)
                    #Constraint
                    # Ctemp = constraint_norm(Ctemp)

                    Dcal = np.dot(Ctemp.T,ST_)
                    error = msea(dauxt,Dcal)

                    if iter ==1:
                        error0 = error.copy()
                        C_ = Ctemp

                    per = 100*abs(error-error0)/error
                    print(iter,'and',per)
                    if per == 0:
                        Cf = C_.copy()
                        Sf = ST_.copy()
                        ST_temp = ST_.copy()
                        status = 'Iterating'
                        self.purest.emit(iter,per,status, Cf.T,Sf.T)
                    elif  per < self.stopping_error:
                        Cf = C_.copy()
                        Sf = ST_.copy()
                    # Save_data(Cf,Sf)
                        status = 'converged'
                        self.purest.emit(iter,per,status, Cf.T,Sf.T)
                        break
                    else:
                        error0 = error.copy()
                        C_ = Ctemp.copy()
                        Sf = ST_.copy()
                        # status = 'Iterating'
                        # self.purest.emit(iter,per,status, C_.T,Sf.T)

                if C_ is not None:
                    ST_temp = self.solve_lineq(C_.T,dauxt)
                    #Constraint
                    # ST_temp = constraint_norm(ST_temp)

                    Dcal = np.dot(C_.T,ST_temp)
                    error = msea(dauxt,Dcal)

                    if iter ==1:
                        error0 = error.copy()
                        # C_ = Ctemp
                        ST_ = ST_temp

                    per = 100*abs(error-error0)/error
                    print(iter,'and',per)
                    if per == 0:
                        Cf = C_.copy()
                        Sf = ST_.copy()
                        status = 'Iterating'
                        self.purest.emit(iter,per,status, Cf.T,Sf.T)

                    elif per < self.stopping_error:
                        Cf = C_.copy()
                        Sf = ST_.copy()
                    # Save_data(Cf,Sf)
                        status = 'converged'
                        self.purest.emit(iter,per,status, Cf.T,Sf.T)
                        break

                    else:
                        error0 = error.copy()
                        ST_ = ST_temp.copy()
                        status = 'Iterating'
                        self.purest.emit(iter,per,status, C_.T,ST_.T)

                if iter == self.niter:
                    status = 'Max iterations reached'
                    self.purest.emit(iter,per,status, C_.T,ST_.T)
                    break
            # Save_data(Cf,Sf)


def main():
    parser = argparse.ArgumentParser(
            description='Graphical application for MCR-ALS analysis of hyperspectral data.')
    MyMainWindow.run_octavvs_application(parser=parser)
