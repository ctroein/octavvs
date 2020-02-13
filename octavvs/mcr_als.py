# -*- coding: utf-8 -*-
"""
@author: syahr
"""
import gc
import sys
import csv
import glob
import os
import pandas as pd
import traceback
from os.path import basename, dirname
from datetime import datetime
from pkg_resources import resource_filename
import argparse


from PyQt5.QtWidgets import QApplication, QFileDialog, QMessageBox
from PyQt5.Qt import QMainWindow,qApp
from PyQt5.QtCore import QThread, pyqtSignal, pyqtSlot #QCoreApplication, QObject, QRunnable, QThreadPool
from PyQt5 import uic

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error as msea
import scipy as sc
import skimage
from skimage.draw import polygon

from .pymcr_new.regressors import OLS, NNLS
from .pymcr_new.constraints import ConstraintNonneg, ConstraintNorm
from .mcr import ftir_function as ff
from .miccs import correction as mc
from .miccs import ExceptionDialog

Ui_MainWindow = uic.loadUiType(resource_filename(__name__, "mcr/mcr_final_loc.ui"))[0]
Ui_MainWindow2 = uic.loadUiType(resource_filename(__name__, "mcr/mcr_roi_sub.ui"))[0]


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


class MyMainWindow(QMainWindow, Ui_MainWindow):

    projectionUpdated = pyqtSignal()
    loadedFile = pyqtSignal()

    def __init__(self,parent=None):
        super(MyMainWindow, self).__init__(parent)
        qApp.installEventFilter(self)
        self.setupUi(self)

        self.lineEditSuffix.setText('_purest')
        self.pushButtonLocal.setEnabled(False)
        self.pushButtonLoad.clicked.connect(self.Load)
        self.lock_all(False)
        self.progressBar.hide()
        self.pushButtonStop.hide()
        self.comboBoxMethod.currentIndexChanged.connect(self.ImageProjection)
        self.horizontalSliderWavenumber.valueChanged.connect(self.Wavenumbercal)
        self.comboBoxCmaps.currentTextChanged.connect(self.plot_visual.setCmap)
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
        self.comboBoxCmaps.currentTextChanged.connect(self.roiDialog.setCmap)
        self.loadedFile.connect(self.roiDialog.resetAll)

        ExceptionDialog.install(self)

    def closeEvent(self, event):
        self.roiDialog.close()
        msgBox = QMessageBox()
        msgBox.setIcon(QMessageBox.Question)
        msgBox.setText("Warning")
        msgBox.setInformativeText('Are you sure to close the window ?')
        msgBox.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        msgBox.setDefaultButton(QMessageBox.No)
        reply = msgBox.exec_()
        if reply == QMessageBox.Yes:
            plt.close('all')
            if hasattr(self, 'calpures'):
                self.calpures.stop()
#            self.killer_renew()
            qApp.quit()
        else:
            event.ignore()


    def roi(self):
        self.comboBoxInitial.setCurrentIndex(0)
        self.roiDialog.show()

    def Load(self):
        if self.comboBoxSingMult.currentIndex() == 1:
            self.pushButtonLocal.setEnabled(False)
            options = QFileDialog.Options()
            options |= QFileDialog.DontUseNativeDialog
            self.foldername = QFileDialog.getExistingDirectory(self,"Open the input data")
            if self.foldername:
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
            fileName, _ = QFileDialog.getOpenFileName(self,"Open Matrix File", "","Matrix File (*.mat)", options=options)
            if fileName:
                self.coord = []
                self.clear_prev()
                self.lineEditTotal.setText(str(1))
                self.initialization(fileName)
                self.lineEditFileNumber.setText(str(1))

    def initialization(self,fileName):

        self.lineEditFilename.setText(basename(fileName))
        self.labelDirectory.setText(dirname(fileName))

        try:
            self.lock_all(True)
            self.sx, self.sy, self.p ,self.wavenumber, self.sp = ff.readmat(fileName)
            self.sp = mc.nonnegative(self.sp)

            self.lineEditLength.setText(str(len(self.wavenumber)))
            self.labelMinwn.setText(str("%.2f" % np.min(self.wavenumber)))
            self.labelMaxwn.setText(str("%.2f" % np.max(self.wavenumber)))
            self.lineEditWavenumber.setText(str("%.2f" % np.min(self.wavenumber)))

            self.index = np.random.randint(0,int(self.sx*self.sy),(20))

            self.plot_specta.canvas.ax.clear()
            self.plot_specta.canvas.ax.plot(self.wavenumber,self.sp[:,self.index])
            self.plot_specta.canvas.fig.tight_layout()
            self.plot_specta.canvas.draw()

            self.ExpandSpecU(self.wavenumber,self.sp)

            try:
                x = int(self.lineEditHeight.text())
                y = int(self.lineEditWidth.text())
                z = int(self.lineEditLength.text())
                self.p = np.reshape(self.p,(z,x,y))
            except ValueError:
                self.lineEditWidth.setText(str(self.sx))
                self.lineEditHeight.setText(str(self.sy))

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
        self.ImageProjection()


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
        plt.close("Image Projection")
        plt.figure("Image Projection")
        plt.imshow(self.projection,str(self.comboBoxCmaps.currentText()))
        if len(self.splot ) != 1 :
            for j in range(0,nr):
                plt.plot(self.pos[j,0],self.pos[j,1],marker='p', color = 'black')
        plt.show()

    def ExpandProjU(self,nr):
        if plt.fignum_exists("Image Projection"):
            fig = plt.figure("Image Projection")
            ax = fig.gca()
            ax.imshow(self.projection,str(self.comboBoxCmaps.currentText()))
            if len(self.splot ) != 1 :
                for j in range(0,nr):
                    ax.plot(self.pos[j,0],self.pos[j,1],marker='p', color = 'black')
            fig.canvas.draw_idle()
        else:
            pass



    def ExpandSVD(self):
        if len(self.splot ) != 1 :
            plt.close("SVD Plot")
            plt.figure("SVD Plot")
            plt.plot(self.xplot,self.splot,'-o')
            plt.show()
        else:
            pass


    def ExpandSVDU(self,x,s):
        if plt.fignum_exists("SVD Plot"):
            fig = plt.figure("SVD Plot")
            ax = fig.gca()
            ax.clear()
            ax.plot(x,s,'-o')
            fig.canvas.draw_idle()
        else:
            pass


    def ExpandInitSpect(self):
        plt.close("Initial")
        plt.figure("Initial")

        if len(self.insp) != 1:
            if self.comboBoxInitial.currentIndex() == 0:
                plt.plot(self.wavenumber,self.insp.T)
            else:
                plt.plot(np.arange(self.sx*self.sy),self.incon.T)
            plt.show("Initial")


    def ExpandInitSpectU(self,x,y):
        if plt.fignum_exists("Initial"):
            fig = plt.figure("Initial")
            ax = fig.gca()
            ax.clear()
            ax.plot(x,y)
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
        self.plot_specta.canvas.ax.plot(self.wavenumber,self.sp[:,self.index])
        self.plot_specta.canvas.ax.axvline(x=self.wavenumv)
        self.plot_specta.canvas.fig.tight_layout()
        self.plot_specta.canvas.draw()

        self.plot_visual.setImage(self.projection, self.comboBoxCmaps.currentText())
        self.projectionUpdated.emit()



    def ExpandSpec(self):
        plt.close("Spectra")
        plt.figure("Spectra")
        plt.plot(self.wavenumber,self.sp[:,self.index])
        plt.xlabel("Wavenumber(1/cm)",fontsize=24)
        plt.ylabel("Absorption(arb. units)",fontsize=24)
        plt.tick_params(axis='both',direction='in', length=8, width=1)
        plt.tick_params(axis='both',which='major',labelsize=24)
        plt.show()

    def ExpandSpecU(self,wn, sp):
        if plt.fignum_exists("Spectra"):
            fig = plt.figure("Spectra")
            ax = fig.gca()
            ax.clear()
            ax.plot(wn,sp[:,self.index])
            fig.canvas.draw_idle()
        else:
            pass



    def ExpandPurConc(self):
        if len(self.copt)  != 1:
            plt.close("Purest Concentrations")
            plt.figure("Purest Concentrations")
            plt.plot(np.arange(len(self.copt)),self.copt)
            plt.show("Purest Concentrations")

    def ExpandPurConcU(self,copt):
        if plt.fignum_exists("Purest Concentrations"):
            fig = plt.figure("Purest Concentrations")
            ax = fig.gca()
            ax.clear()
            ax.plot(np.arange(len(copt)),copt)
            fig.canvas.draw_idle()
        else:
            pass


    def ExpandPurSp(self):
        if len(self.sopt) != 1:
            plt.close("Purest Spectra")
            plt.figure("Purest Spectra")
            plt.plot(self.wavenumber,self.sopt)
            plt.show("Purest Spectra")

    def ExpandPurSpU(self, sopt):
        if plt.fignum_exists("Purest Spectra") and len(sopt) == len(self.wavenumber):
            fig = plt.figure("Purest Spectra")
            ax = fig.gca()
            ax.clear()
            ax.plot(self.wavenumber, sopt)
            fig.canvas.draw_idle()
        else:
            pass



    def SVDprocess(self):
        self.nr = self.spinBoxSVDComp.value()
        if self.nr < 20:
            nplot = self.nr+5
        else:
            nplot = self.nr


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
            self.ExpandSVDU(self.xplot,self.splot)

            per = float(self.lineEditNoisePercent.text())
            self.f = per*0.01


            if self.comboBoxInitial.currentIndex() == 0:
                self.incon = [0,0]
                self.labelInitial.setText("Initial Spectra*")
                self.insp, points = ff.initi_simplisma(sp_new,self.nr,self.f)
                self.plotInitSpec.canvas.ax.clear()
                self.plotInitSpec.canvas.ax.plot(self.wavenumber,self.insp.T)
                self.plotInitSpec.canvas.fig.tight_layout()
                self.plotInitSpec.canvas.draw()


                self.pos = np.zeros((self.nr,2))

                for i in range(0,self.nr):
                    self.pos[i,0] = self.ind[points[i]] % nx
                    self.pos[i,1] = self.ind[points[i]] // nx
                self.plot_visual.addPoints(self.pos)

    def SVDPlot(self):

        self.plotPurestConc.canvas.ax.clear()
        self.plotPurestConc.canvas.draw()
        self.plotPurestSpectra.canvas.ax.clear()
        self.plotPurestSpectra.canvas.draw()

        self.plotSVD.canvas.ax.clear()
        self.plotSVD.canvas.ax.plot(self.xplot,self.splot,'-o')
        self.plotSVD.canvas.draw()
#        if self.comboBoxPurest.currentIndex() == 0:
        self.ExpandSVDU(self.xplot,self.splot)
        self.InitialCondition()


    def InitialCondition(self):
        nr = self.spinBoxSVDComp.value()
        per = float(self.lineEditNoisePercent.text())
        self.f = per*0.01

        if self.comboBoxInitial.currentIndex() == 0:
            self.incon = [0,0]
            self.labelInitial.setText("Initial Spectra*")
            self.insp, points = ff.initi_simplisma(self.sp,nr,self.f)
            self.plotInitSpec.canvas.ax.clear()
            self.plotInitSpec.canvas.ax.plot(self.wavenumber,self.insp.T)
            self.plotInitSpec.canvas.fig.tight_layout()
            self.plotInitSpec.canvas.draw()
            self.ExpandInitSpectU(self.wavenumber,self.insp.T)

            self.pos = np.array([points % self.projection.shape[0], points // self.projection.shape[1]]).T

            self.plot_visual.setImage(self.projection, self.comboBoxCmaps.currentText())
#            self.plot_visual.canvas.ax.clear()
#            self.plot_visual.canvas.ax.imshow(self.projection,str(self.comboBoxCmaps.currentText()))
#            self.plot_visual.canvas.fig.tight_layout()
            self.plot_visual.addPoints(self.pos)
#            for j in range(0,nr):
#                self.plot_visual.canvas.ax.plot(self.pos[j,0],self.pos[j,1],marker='p', color = 'black')
#            self.plot_visual.canvas.draw()
            self.ExpandProjU(nr)
        else:
            self.insp = [0,0]
            self.labelInitial.setText("Initial Concentration*")
            self.incon, __ = ff.initi_simplisma(self.sp.T,nr,self.f)
            self.plotInitSpec.canvas.ax.clear()
            self.plotInitSpec.canvas.ax.plot(np.arange(self.sx*self.sy),self.incon.T)
            self.plotInitSpec.canvas.fig.tight_layout()
            self.plotInitSpec.canvas.draw()
            self.ExpandInitSpectU(np.arange(self.sx*self.sy),self.incon.T)
            self.ImageProjection()


    def lockmcr(self):
        self.comboBoxInitial.setEnabled(True)
        self.pushButtonExpandInitSpect.setEnabled(True)
        self.comboBoxRegressor.setEnabled(True)
        self.lineEditNoisePercent.setEnabled(True)
        self.checkBoxSaveInit.setEnabled(True)
        self.lineEditPurIter.setText('700')
        self.lineEditTol.setText('2e-12')
        self.SVDprocess()


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
        f = float(self.lineEditNoisePercent.text())
        f = f*0.01
        max_iter = int(self.lineEditPurIter.text())
        tol_percent = float(self.lineEditTol.text())
        tol_error = float(self.lineEditTol.text())


        init = self.comboBoxInitial.currentIndex()
        met = 'NNLS'

        self.progressBar.setEnabled(True)
        self.lineEditStatus.setText('Multiple files')
        self.calpures = Multiple_Calculation(init,f,nr,self.foldername, verbose=True, c_regr=met, st_regr=met, c_fit_kwargs={},
                 st_fit_kwargs={}, c_constraints=[ConstraintNonneg(), ConstraintNorm()],
                 st_constraints=[ConstraintNonneg()],
                 max_iter=max_iter, tol_percent = tol_percent,
                 tol_increase=0.0, tol_n_increase=1, tol_err_change=tol_error)

        self.calpures.purest.connect(self.finished_mcr_all)
        self.calpures.start()
        self.pushButtonStop.show()
        self.pushButtonPurestCal.setEnabled(False)

    def finished_mcr_all(self,itera,name,itern,error,status,copt,sop):
        self.copt = copt
        self.sopt = sop

        self.ExpandPurSpU(sop)
        self.ExpandPurConcU(copt)

        if itern == 2:
            self.initialization(name)
            self.lineEditFileNumber.setText(str(itera+1))
#            if self.comboBoxInitial.currentIndex() == 0:
            self.SVDprocess()

        self.plotPurestSpectra.canvas.ax.clear()
        self.plotPurestSpectra.canvas.ax.plot(self.wavenumber,sop)
        self.plotPurestSpectra.canvas.fig.tight_layout()
        self.plotPurestSpectra.canvas.draw()

        self.plotPurestConc.canvas.ax.clear()
        self.plotPurestConc.canvas.ax.plot(np.arange(len(copt)),copt)
        self.plotPurestConc.canvas.fig.tight_layout()
        self.plotPurestConc.canvas.draw()

        self.lineEdit_Niter.setText(str(itern))
        self.lineEdit_Error.setText(str(round(error,5)))
        self.lineEditStatus.setText(status)

        self.progressBar.setMaximum(self.nfiles+1)
        self.progressBar.setValue(itera+1)

        if (status == 'Max iterations reached') or (status == 'converged'):
#            self.pushButtonPurestCal.setEnabled(True)

            self.save_data(copt,sop)
            if (itera+1 == self.nfiles):
                self.progressBar.setValue(self.nfiles+1)
                self.pushButtonPurestCal.setEnabled(True)


    @pyqtSlot(list)
    def setRoi(self, roi):
        self.coord = roi

    def runsingle(self):
        if not self.coord:
            self.runsingle_noroi()
        else:
            self.runsingle_roi()

    def runsingle_noroi(self):
        max_iter = int(self.lineEditPurIter.text())
        tol_percent = float(self.lineEditTol.text())
#        self.lineEditStatus.setText('-----Iterating-----')
        self.SVDprocess()
        tol_error = float(self.lineEditTol.text())

        nr = self.spinBoxSVDComp.value()
        nrow, ncol = np.shape(self.sp)
        s = sc.linalg.diagsvd(self.s,nrow, ncol)
        u = self.u[:,0:nr]
        s = s[0:nr,0:nr]
        v = self.v[0:nr,:]
        self.dn = u @ s @ v

        init = self.comboBoxInitial.currentIndex()
        self.regres = self.comboBoxRegressor.currentIndex()

        if self.regres == 0:
            met= 'NNLS'
        else:
            met= 'OLS'

        nrow, ncol = np.shape(self.dn)
        dauxt = np.zeros((ncol,nrow))
        aux=self.dn.T
        tol_percent = float(self.lineEditTol.text())

        for i in range(0,ncol):
            dauxt[i,:]=aux[i,:]/np.sqrt(np.sum(aux[i,:]*aux[i,:]))


        if init == 0:
            C = None
            ST = self.insp
        else:
            C = self.incon.T
            ST = None

        self.calpures = single_report(dauxt, C=C, ST=ST, verbose=True, c_regr=met, st_regr=met, c_fit_kwargs={},
                 st_fit_kwargs={}, c_constraints=[ConstraintNonneg(), ConstraintNorm()],
                 st_constraints=[ConstraintNonneg()],
                 max_iter=max_iter, tol_percent = tol_percent,
                 tol_increase=0.0, tol_n_increase=1, tol_err_change=tol_error)
        self.calpures.purest.connect(self.finished_single)
        self.calpures.start()
        self.pushButtonStop.show()
        self.pushButtonPurestCal.setEnabled(False)

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
#        mask = np.ones((nx,ny))
        max_iter = int(self.lineEditPurIter.text())
        tol_error = float(self.lineEditTol.text())

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
        self.ExpandSVDU(self.xplot,self.splot)

        per = float(self.lineEditNoisePercent.text())
        self.f = per*0.01


        if self.comboBoxInitial.currentIndex() == 0:
            self.incon = [0,0]
            self.labelInitial.setText("Initial Spectra*")
            self.insp, points = ff.initi_simplisma(sp_new,nr,self.f)
            self.plotInitSpec.canvas.ax.clear()
            self.plotInitSpec.canvas.ax.plot(self.wavenumber,self.insp.T)
            self.plotInitSpec.canvas.fig.tight_layout()
            self.plotInitSpec.canvas.draw()

#            self.pos = np.array([points % self.projection.shape[0], points // self.projection.shape[1]]).T
            self.pos = np.zeros((nr,2))

            for i in range(0,self.nr):
                self.pos[i,0] = self.ind[points[i]] % nx
                self.pos[i,1] = self.ind[points[i]] // nx
                self.plot_visual.addPoints(self.pos)
#            self.plot_visual.canvas.ax.plot(self.pos[i,0],self.pos[i,1],marker='p', color = 'black')
#            self.plot_visual.canvas.draw_idle()

#            self.plot_visual.canvas.ax.clear()
#            self.plot_visual.canvas.ax.imshow(self.projection,str(self.comboBoxCmaps.currentText()))
#            self.plot_visual.canvas.fig.tight_layout()
#            self.plot_visual.canvas.ax.plot(xs,ys,'red')


        nrow, ncol = np.shape(sp_new)

        s = sc.linalg.diagsvd(self.s,nrow, ncol)
        u = self.u[:,0:nr]
        s = s[0:nr,0:nr]
        v = self.v[0:nr,:]
        self.dn = u @ s @ v

        self.regres = self.comboBoxRegressor.currentIndex()
        if self.regres == 0:
            met= 'NNLS'
        else:
            met= 'OLS'

        nrow, ncol = np.shape(self.dn)
        dauxt = np.zeros((ncol,nrow))
        aux=self.dn.T
        tol_percent = float(self.lineEditTol.text())

        for i in range(0,ncol):
            dauxt[i,:]=aux[i,:]/np.sqrt(np.sum(aux[i,:]*aux[i,:]))


        self.calpures = single_report(dauxt, C=None, ST=self.insp, verbose=True, c_regr=met, st_regr=met, c_fit_kwargs={},
                 st_fit_kwargs={}, c_constraints=[ConstraintNonneg(), ConstraintNorm()],
                 st_constraints=[ConstraintNonneg()],
                 max_iter=max_iter, tol_percent = tol_percent,
                 tol_increase=0.0, tol_n_increase=1, tol_err_change=tol_error)
        self.calpures.purest.connect(self.finished_single_roi)
        self.calpures.start()
        self.pushButtonStop.show()
        self.pushButtonPurestCal.setEnabled(False)

    def finished_single_roi(self, itern,error,status,copt,sopt):
        self.copt = copt
        self.sopt = sopt

        self.ExpandPurConcU(copt)
        self.ExpandPurSpU(sopt)



        nr = self.spinBoxSVDComp.value()
        nx, ny = int(self.lineEditWidth.text()),int(self.lineEditHeight.text())
        bea = np.zeros((int(nx*ny),nr))
        for i in range(0,nr):
            bea[self.ind,i] = copt[:,i]


        self.plotPurestSpectra.canvas.ax.clear()
        self.plotPurestSpectra.canvas.ax.plot(self.wavenumber,sopt)
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
            self.save_data(bea,sopt)



    def save_data(self, copt, sopt):
        if self.checkBoxSavePurest.isChecked():
            auxi = np.concatenate((sopt,copt), axis = 0)
            namef = self.lineEditFilename.text()
            namef = namef.replace('.mat','')
            np.savetxt(self.folpurest+'/'+namef+self.lineEditSuffix.text()+'.csv', auxi, delimiter=',')
#            QApplication.primaryScreen().grabWindow(self.winId()).save(self.folpurest+'/'+namef+'_SC'+'.png')


    def SaveInit(self):
        if self.checkBoxSaveInit.isChecked():
            options = QFileDialog.Options()
            options |= QFileDialog.DontUseNativeDialog
            self.folinit = QFileDialog.getExistingDirectory(self,"Open the input data")
        else:
            pass


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
        self.ExpandPurSpU(sopt)



        self.plotPurestSpectra.canvas.ax.clear()
        self.plotPurestSpectra.canvas.ax.plot(self.wavenumber,sopt)
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
            self.pushButtonPurestCal.setEnabled(True)
            self.save_data(copt,sopt)
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
        self.pushButtonPurestCal.setEnabled(Stat)
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
"""
The code of these classes were taken modified from
National Institute of Standards and Technology (NIST), reference
(1) C. H. Camp Jr., “pyMCR: A Python Library for MultivariateCurve
    Resolution Analysis with Alternating Regression (MCR-AR)”, 124, 1-10 (2019)

"""



class Multiple_Calculation(QThread):
#self.purest.emit(itera,name,self.ST,self.C,self.n_iter,err_differ,status, self.C_,self.ST_.T)

    purest = pyqtSignal(np.int,str,np.int,np.float64,str,np.ndarray, np.ndarray)
    QThread.setTerminationEnabled()
    def __init__(self,init, sim_error,nr,foldername, verbose=False, c_regr=OLS(), st_regr=OLS(), c_fit_kwargs={},
                 st_fit_kwargs={}, c_constraints=[ConstraintNonneg()],
                 st_constraints=[ConstraintNonneg()],
                 max_iter=50, tol_percent=0.1,
                 tol_increase=0.0, tol_n_increase=10, tol_err_change=None, parent=None
                ):
        QThread.__init__(self, parent)
        """
        Multivariate Curve Resolution - Alternating Regression
        """
        self.init = init
        self.nr = nr
        self.f = sim_error
        self.fold = foldername
        self.tol_percent = tol_percent
        self.C = None
        self.ST = None
        self.dn = None
        self.max_iter = max_iter
        self.tol_increase = tol_increase
        self.tol_n_increase = tol_n_increase
        self.tol_err_change = tol_err_change

#        self.err_fcn = err_fcn
        self.err = []

        self.c_constraints = c_constraints
        self.st_constraints = st_constraints

        self.c_regressor = self._check_regr(c_regr)
        self.st_regressor = self._check_regr(st_regr)
        self.c_fit_kwargs = c_fit_kwargs
        self.st_fit_kwargs = st_fit_kwargs

        self.C_ = None
        self.ST_ = None

        self.C_opt_ = None
        self.ST_opt_ = None
        self.n_iter_opt = None

        self.n_iter = None
        self.n_increase = None
        self.max_iter_reached = False

        # Saving every C or S^T matrix at each iteration
        # Could create huge memory usage
        self._saveall_st = False
        self._saveall_c = False
        self._saved_st = []
        self._saved_c = []
        self.verbose = verbose
        self.rem = False

        if self.init == 0:
            inguess = "Spectra"
        else:
            inguess ="Concentrations"
#----------------------------------------------------------------------------------------
        self.logfile = open(self.fold+"//logfile.txt", "w")
        now = datetime.now()
        date_time = now.strftime("%m/%d/%Y, %H:%M:%S")
        self.logfile.write("%s\n" % ("Multivariate Curve Resolution-Alternating Least Square"))
        self.logfile.write("%s\n" % ("Logging started at "+date_time))
        self.logfile.write("%s\n" % ("   Folder: "+self.fold))
        self.logfile.write("%s\n" % ("   Number of components: "+str(self.nr)))
        self.logfile.write("%s\n" % ("   Noise in SIMPLISMA: "+str(self.f)))
        self.logfile.write("%s\n" % ("   Tolerance: "+str(self.tol_percent)))
        self.logfile.write("%s\n" % ("   Initial Guess: "+ inguess))
        self.logfile.write("%s\n" % ("-------------------------------------------------------"))
#----------------------------------------------------------------------------------------

    def logreport(self,filenumber,filename,niter,status,maxfile):
        now = datetime.now()
        date_time = now.strftime("%m/%d/%Y, %H:%M:%S")
        self.logfile.write("%s %s %s %s\n" % (int(filenumber+1),date_time, filename, status+" at "+str(niter)))

        if (filenumber+1) == maxfile:
            self.logfile.close()


    def _check_regr(self, mth):
        """
            Check regressor method. If accetible strings, instantiate and return
            object. If instantiated class, make sure it has a fit attribute.
        """
        if isinstance(mth, str):
            if mth.upper() == 'OLS':
                return OLS()
            elif mth.upper() == 'NNLS':
                return NNLS()
            else:
                raise ValueError('{} is unknown. Use NNLS or OLS.'.format(mth))
        elif hasattr(mth, 'fit'):
            return mth
        else:
            raise ValueError('Input class {} does not have a \'fit\' method'.format(mth))


    @property
    def D_(self):
        """ D matrix with current C and S^T matrices """
        return np.dot(self.C_, self.ST_)

    @property
    def D_opt_(self):
        """ D matrix with optimal C and S^T matrices """
        return np.dot(self.C_opt_, self.ST_opt_)

    def _ismin_err(self, val):
        """ Is the current error the minimum """
        if len(self.err) == 0:
            return True
        else:
            return ([val > x for x in self.err].count(True) == 0)

    def stop(self):
        try:
            self.logfile.write("%s\n" % ("The calculation is terminated"))
            self.logfile.close()
        except:
            pass
        self.rem = True



    def run(self):
        df = pd.read_csv(os.path.join(self.fold, "Fileall.csv"),header=None)
        value = len(df)
        err_differ = 0.
#        sc.io.savemat(os.path.join(self.fold, '00.start'), {'nada': [[0]] } )
        for itera in range(0,value):
            filename = df.iloc[itera,1]
            name = filename#basename(filename)
            sx, sy, p ,wavenumber, sp = ff.readmat(filename)

            if self.rem:
                status = 'STOP'
                self.purest.emit(itera,name,self.n_iter,err_differ,status, self.C_opt_,self.ST_opt_.T)
                break
            else:

                self.err = []
                sp = mc.nonnegative(sp)
                if self.init== 0:
                    self.ST , points = ff.initi_simplisma(sp,self.nr,self.f)
                    self.C = None
                else:
                    con, points = ff.initi_simplisma(sp.T,self.nr,self.f)
                    self.C = con.T
                    self.ST = None


                u, s, v = np.linalg.svd(sp)

                nrow, ncol = np.shape(sp)

                s = sc.linalg.diagsvd(s,nrow, ncol)
                u = u[:,0:self.nr]
                s = s[0:self.nr,0:self.nr]
                v = v[0:self.nr,:]
                dn = u @ s @ v

                nrow, ncol = np.shape(dn)
                dauxt = np.zeros((ncol,nrow))
                aux=dn.T

                for i in range(0,ncol):
                    dauxt[i,:]=aux[i,:]/np.sqrt(np.sum(aux[i,:]*aux[i,:]))

                D = dauxt

                if (self.C is None) & (self.ST is None):
                    raise TypeError('C or ST estimate must be provided')
                elif (self.C is not None) & (self.ST is not None):
                    raise TypeError('Only C or ST estimate must be provided, only one')
                else:
                    self.C_ = self.C
                    self.ST_ = self.ST

                self.n_increase = 0


                for num in range(self.max_iter):
                    self.n_iter = num + 1
                    if self.rem:
                        status = 'STOP'
                        self.purest.emit(itera,name,self.n_iter,err_differ,status, self.C_opt_,self.ST_opt_.T)
                        break
                    else:

                        if self.ST_ is not None:
                            # Debugging feature -- saves every S^T matrix in a list
                            # Can create huge memory usage
                            if self._saveall_st:
                                self._saved_st.append(self.ST_)

                            self.c_regressor.fit(self.ST_.T, D.T, **self.c_fit_kwargs)
                            C_temp = self.c_regressor.coef_

                            # Apply c-constraints
                            for constr in self.c_constraints:
                                C_temp = constr.transform(C_temp)

                            D_calc = np.dot(C_temp, self.ST_)
                            err_temp = msea(D, D_calc)
            #                err_temp = self.err_fcn(C_temp, self.ST_, D, D_calc)


                            if self._ismin_err(err_temp):
                                self.C_opt_ = 1*C_temp
                                self.ST_opt_ = 1*self.ST_
                                self.n_iter_opt = num + 1

                            # Calculate error fcn and check for tolerance increase
                            if self.err != 0:
                                self.err.append(1*err_temp)
                                self.C_ = 1*C_temp
                            elif (err_temp <= self.err[-1]*(1+self.tol_increase)):
                                self.err.append(1*err_temp)
                                self.C_ = 1*C_temp
                            else:
                                print('Mean squared residual increased above tol_increase {:.4e}. Exiting'.format(err_temp))
                                break

                            # Check if err went up
                            if len(self.err) > 1:
                                if self.err[-1] > self.err[-2]:  # Error increased
                                    self.n_increase += 1
                                else:
                                    self.n_increase *= 0

                            # Break if too many error-increases in a row
                            if self.n_increase > self.tol_n_increase:
                                print('Maximum error increases reached ({}). Exiting.'.format(self.tol_n_increase))
                                break



                        if self.C_ is not None:

                            # Debugging feature -- saves every C matrix in a list
                            # Can create huge memory usage
                            if self._saveall_c:
                                self._saved_c.append(self.C_)

                            self.st_regressor.fit(self.C_, D, **self.st_fit_kwargs)
                            ST_temp = self.st_regressor.coef_.T


                            # Apply ST-constraints
                            for constr in self.st_constraints:
                                ST_temp = constr.transform(ST_temp)

                            D_calc = np.dot(self.C_, ST_temp)
                            err_temp = msea(D, D_calc)

                            # Calculate error fcn and check for tolerance increase
                            if self._ismin_err(err_temp):
                                self.ST_opt_ = 1*ST_temp
                                self.C_opt_ = 1*self.C_
                                self.n_iter_opt = num + 1

                            if len(self.err) == 0:
                                self.err.append(1*err_temp)
                                self.ST_ = 1*ST_temp
                            elif (err_temp <= self.err[-1]*(1+self.tol_increase)):
                                self.err.append(1*err_temp)
                                self.ST_ = 1*ST_temp
                            else:
                                print('Mean squared residual increased above tol_increase {:.4e}. Exiting'.format(err_temp))
        #                        break

                            # Check if err went up
                            if len(self.err) > 1:
                                if self.err[-1] > self.err[-2]:  # Error increased
                                    self.n_increase += 1
                                else:
                                    self.n_increase *= 0

                            # Break if too many error-increases in a row
                            if self.n_increase > self.tol_n_increase:
                                print('Maximum error increases reached ({}). Exiting.'.format(self.tol_n_increase))
                                break

                        if self.n_iter >= self.max_iter:
                            print('Max iterations reached ({}).'.format(num+1))
                            status = 'Max iterations reached'
        #                    err_differ=0.0
        #                    self.purest.emit(itera,name,self.ST,camp,self.n_iter,err_differ,status, self.C_opt_,self.ST_opt_.T)
                            self.purest.emit(itera,name,self.n_iter,err_differ,status, self.C_opt_,self.ST_opt_.T)
                            self.max_iter_reached = True
                            self.logreport(itera,basename(filename),self.n_iter,status,value)
                            break

                        self.n_iter = num + 1


                        if ((self.tol_err_change is not None) & (len(self.err) > 2)):
            #

                            err_differ = np.abs(self.err[-1] - self.err[-3])/self.err[-1]*100
                            status = 'iterating'
                            self.purest.emit(itera,name,self.n_iter,err_differ,status, self.C_,self.ST_opt_.T)
                            if err_differ < np.abs(self.tol_percent):
                                print('Change in err below tol_err_change ({:.4e}). Exiting.'.format(err_differ))
                                status = 'converged'
                                self.purest.emit(itera,name,self.n_iter,err_differ,status, self.C_opt_,self.ST_opt_.T)
                                self.logreport(itera,basename(filename),self.n_iter,status,value)
                                break

            gc.collect()

 #----------------------------------------------------------------------


class single_report(QThread):

    purest = pyqtSignal(np.int,np.float64,str,np.ndarray, np.ndarray)
    def __init__(self,D, C=None, ST=None, verbose=False, c_regr=OLS(), st_regr=OLS(), c_fit_kwargs={},
                 st_fit_kwargs={}, c_constraints=[ConstraintNonneg()],
                 st_constraints=[ConstraintNonneg()],
                 max_iter=50, tol_percent=0.1,
                 tol_increase=0.0, tol_n_increase=10, tol_err_change=None, parent=None
                ):
        QThread.__init__(self, parent)
        """
        Multivariate Curve Resolution - Alternating Regression
        """

        self.tol_percent = tol_percent
        self.C = C
        self.ST = ST
        self.dn = D
        self.max_iter = max_iter
        self.tol_increase = tol_increase
        self.tol_n_increase = tol_n_increase
        self.tol_err_change = tol_err_change

#        self.err_fcn = err_fcn
        self.err = []

        self.c_constraints = c_constraints
        self.st_constraints = st_constraints

        self.c_regressor = self._check_regr(c_regr)
        self.st_regressor = self._check_regr(st_regr)
        self.c_fit_kwargs = c_fit_kwargs
        self.st_fit_kwargs = st_fit_kwargs

        self.C_ = None
        self.ST_ = None

        self.C_opt_ = None
        self.ST_opt_ = None
        self.n_iter_opt = None

        self.n_iter = None
        self.n_increase = None
        self.max_iter_reached = False

        # Saving every C or S^T matrix at each iteration
        # Could create huge memory usage
        self._saveall_st = False
        self._saveall_c = False
        self._saved_st = []
        self._saved_c = []
        self.verbose = verbose
        self.rem = False

    def _check_regr(self, mth):
        """
            Check regressor method. If accetible strings, instantiate and return
            object. If instantiated class, make sure it has a fit attribute.
        """
        if isinstance(mth, str):
            if mth.upper() == 'OLS':
                return OLS()
            elif mth.upper() == 'NNLS':
                return NNLS()
            else:
                raise ValueError('{} is unknown. Use NNLS or OLS.'.format(mth))
        elif hasattr(mth, 'fit'):
            return mth
        else:
            raise ValueError('Input class {} does not have a \'fit\' method'.format(mth))


    @property
    def D_(self):
        """ D matrix with current C and S^T matrices """
        return np.dot(self.C_, self.ST_)

    @property
    def D_opt_(self):
        """ D matrix with optimal C and S^T matrices """
        return np.dot(self.C_opt_, self.ST_opt_)

    def _ismin_err(self, val):
        """ Is the current error the minimum """
        if len(self.err) == 0:
            return True
        else:
            return ([val > x for x in self.err].count(True) == 0)


    def stop(self):
        self.rem = True


    def run(self):
        """
        Perform MCR-ALS. D = CS^T. Solve for C and S^T iteratively.

        Parameters
        ----------
        Dn: ndarray
            Dn --> Dexperiment

        D : ndarray
            D matrix --> DPCA

        C : ndarray
            Initial C matrix estimate. Only provide initial C OR S^T.

        ST : ndarray
            Initial S^T matrix estimate. Only provide initial C OR S^T.

        verbose : bool
            Display iteration and per-least squares err results.
        """

        # Ensure only C or ST provided
        D = self.dn

        if (self.C is None) & (self.ST is None):
            raise TypeError('C or ST estimate must be provided')
        elif (self.C is not None) & (self.ST is not None):
            raise TypeError('Only C or ST estimate must be provided, only one')
        else:
            self.C_ = self.C
            self.ST_ = self.ST
        self.n_increase = 0
        err_differ = 0.

        for num in range(self.max_iter):

            if self.rem:
                status = 'STOP'
                self.purest.emit(self.n_iter,err_differ,status, self.C_opt_,self.ST_opt_.T)
                break
            else:
                self.n_iter = num + 1
                if self.ST_ is not None:
                    # Debugging feature -- saves every S^T matrix in a list
                    # Can create huge memory usage
                    if self._saveall_st:
                        self._saved_st.append(self.ST_)

                    self.c_regressor.fit(self.ST_.T, D.T, **self.c_fit_kwargs)
                    C_temp = self.c_regressor.coef_

                    # Apply c-constraints
                    for constr in self.c_constraints:
                        C_temp = constr.transform(C_temp)

                    D_calc = np.dot(C_temp, self.ST_)
                    err_temp = msea(D, D_calc)
    #                err_temp = self.err_fcn(C_temp, self.ST_, D, D_calc)


                    if self._ismin_err(err_temp):
                        self.C_opt_ = 1*C_temp
                        self.ST_opt_ = 1*self.ST_
                        self.n_iter_opt = num + 1

                    # Calculate error fcn and check for tolerance increase
                    if self.err != 0:
                        self.err.append(1*err_temp)
                        self.C_ = 1*C_temp
                    elif (err_temp <= self.err[-1]*(1+self.tol_increase)):
                        self.err.append(1*err_temp)
                        self.C_ = 1*C_temp
                    else:
                        print('Mean squared residual increased above tol_increase {:.4e}. Exiting'.format(err_temp))
                        status = 'Mean squared residual increased above tol_increase'
                        self.purest.emit(self.n_iter,err_differ,status, self.C_opt_,self.ST_opt_.T)
                        break

                    # Check if err went up
                    if len(self.err) > 1:
                        if self.err[-1] > self.err[-2]:  # Error increased
                            self.n_increase += 1
                        else:
                            self.n_increase *= 0

                    # Break if too many error-increases in a row
                    if self.n_increase > self.tol_n_increase:
                        print('Maximum error increases reached ({}). Exiting.'.format(self.tol_n_increase))
                        status = 'Maximum error increases reached'
                        self.purest.emit(self.n_iter,err_differ,status, self.C_opt_,self.ST_opt_.T)
                        break

                if self.C_ is not None:

                    # Debugging feature -- saves every C matrix in a list
                    # Can create huge memory usage
                    if self._saveall_c:
                        self._saved_c.append(self.C_)

                    self.st_regressor.fit(self.C_, D, **self.st_fit_kwargs)
                    ST_temp = self.st_regressor.coef_.T



                    # Apply ST-constraints
                    for constr in self.st_constraints:
                        ST_temp = constr.transform(ST_temp)
    #
                    D_calc = np.dot(self.C_, ST_temp)


    #                err_temp = self.err_fcn(self.C_, ST_temp, D, D_calc)
                    err_temp = msea(D, D_calc)

                    # Calculate error fcn and check for tolerance increase
                    if self._ismin_err(err_temp):
                        self.ST_opt_ = 1*ST_temp
                        self.C_opt_ = 1*self.C_
                        self.n_iter_opt = num + 1

                    if len(self.err) == 0:
                        self.err.append(1*err_temp)
                        self.ST_ = 1*ST_temp
                    elif (err_temp <= self.err[-1]*(1+self.tol_increase)):
                        self.err.append(1*err_temp)
                        self.ST_ = 1*ST_temp
                    else:
                        print('Mean squared residual increased above tol_increase {:.4e}. Exiting'.format(err_temp))
                        status = 'Mean squared residual increased above tol_increase'
                        self.purest.emit(self.n_iter,err_differ,status, self.C_opt_,self.ST_opt_.T)
                        break

                    # Check if err went up
                    if len(self.err) > 1:
                        if self.err[-1] > self.err[-2]:  # Error increased
                            self.n_increase += 1
                        else:
                            self.n_increase *= 0

                    # Break if too many error-increases in a row
                    if self.n_increase > self.tol_n_increase:
                        print('Maximum error increases reached ({}). Exiting.'.format(self.tol_n_increase))
                        status = 'Maximum error increases'
                        self.purest.emit(self.n_iter,err_differ,status, self.C_opt_,self.ST_opt_.T)
                        break


                if self.n_iter >= self.max_iter:
                    print('Max iterations reached ({}).'.format(num+1))
                    status = 'Max iterations reached'
                    self.purest.emit(self.n_iter,err_differ,status, self.C_opt_,self.ST_opt_.T)
                    self.max_iter_reached = True
                    break

                self.n_iter = num + 1


                if ((self.tol_err_change is not None) & (len(self.err) > 2)):
                    err_differ = np.abs(self.err[-1] - self.err[-3])/self.err[-1]*100
                    status = 'iterating'
                    self.purest.emit(self.n_iter,err_differ,status, self.C_opt_,self.ST_opt_.T)
                    if err_differ < np.abs(self.tol_percent):
                        print('Change in err below tol_err_change ({:.4e}). Exiting.'.format(err_differ))
                        status = 'converged'
                        self.purest.emit(self.n_iter,err_differ,status, self.C_opt_,self.ST_opt_.T)
                        break


def main():
    parser = argparse.ArgumentParser(
            description='Graphical application for MCR-ALS analysis of hyperspectral data.')
    args = parser.parse_args()
    try:
        app = QApplication.instance()
        if not app:
            app = QApplication(sys.argv)
        window = MyMainWindow()
        window.show()
        res = app.exec_()
    except Exception:
        traceback.print_exc()
        print('Press some key to quit')
        input()
    sys.exit(res)

if __name__ == '__main__':
    main()

