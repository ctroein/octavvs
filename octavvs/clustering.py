import sys
import csv
import glob
import os
import pandas as pd
import collections
import traceback
from os.path import basename, dirname
from pkg_resources import resource_filename
import argparse
import io
#from PyQt5.QtCore import *
#from PyQt5.QtGui import QFileDialog
#from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QApplication, QFileDialog, QMessageBox, QDialog
from PyQt5.Qt import QMainWindow, qApp #, QTimer
#from PyQt5.QtCore import (QCoreApplication, QObject, QRunnable, QThread, QThreadPool,pyqtSignal, pyqtSlot)
from PyQt5 import uic
from PyQt5.QtCore import pyqtSignal, pyqtSlot

import numpy as np
from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.preprocessing import StandardScaler
from scipy.signal import savgol_filter
import pickle
import matplotlib
matplotlib.use('QT5Agg')
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QTableWidgetItem, QTableWidgetSelectionRange

from octavvs.mcr import ftir_function as ff
from octavvs.algorithms import correction as mc
from octavvs.ui import (FileLoader, ImageVisualizer, OctavvsMainWindow, NoRepeatStyle, uitools)

Ui_MainWindow = uic.loadUiType(resource_filename(__name__, "mcr/clustering_ui.ui"))[0]
Ui_table = uic.loadUiType(resource_filename(__name__, "mcr/table_ui.ui"),
                          from_imports=True, import_from='octavvs')[0]
Ui_DialogAbout = uic.loadUiType(resource_filename(__name__, "mcr/about.ui"),
                                from_imports=True, import_from='octavvs')[0]

class DialogAbout(QDialog, Ui_DialogAbout):
    def __init__(self, parent=None):
        QDialog.__init__(self, parent)
        qApp.installEventFilter(self)
        self.setupUi(self)


class Table(QMainWindow, Ui_table):
    def __init__(self, parent=None):
        super(Table, self).__init__(parent=None)
        qApp.installEventFilter(self)
        self.setupUi(self)
        self.actionSelect_all.triggered.connect(self.select)
        self.actionDeselect_all.triggered.connect(self.deselect)
        self.actionCopy.triggered.connect(self.copySelection)
        self.actionSave_as.triggered.connect(self.saveas)
        self.actionClose.triggered.connect(self.close)


    @pyqtSlot(np.ndarray, int, list)
    def main_data(self,data, decision,label):
        self.data_tab = data
        self.decision = decision
        self.row  = len(self.data_tab)
        self.col = len(self.data_tab.T)
        self.label = label

        if self.decision == 0:
            name = range(1,self.col)
            name = (list(map(str,name)))

            head = ['Wavenumber']
            head = head + name
            self.tableWidget.setHorizontalHeaderLabels(head)
            self.head = head

        if self.label != ['0']:
            head = ['Wavenumber']
            head = head + self.label
            self.tableWidget.setHorizontalHeaderLabels(head)
            self.head = head

        if self.data_tab is not None:
            self.row  = len(self.data_tab)
            self.col = len(self.data_tab.T)
            self.tableWidget.setRowCount(self.row)
            self.tableWidget.setColumnCount(self.col)

            for i in range(0,self.row):
                for j in range(0,self.col):
                    self.tableWidget.setItem(i, j, QTableWidgetItem(str(self.data_tab[i,j])))

    def select(self):
        self.tableWidget.setRangeSelected(QTableWidgetSelectionRange(0,0, self.row-1, self.col-1), True)

    def deselect(self):
        self.tableWidget.setRangeSelected(QTableWidgetSelectionRange(0,0, self.row-1, self.col-1), False)

    def close(self):
        self.hide()

    def copySelection(self):
        selection = self.tableWidget.selectedIndexes()
        if selection:
            rows = sorted(index.row() for index in selection)
            columns = sorted(index.column() for index in selection)
            rowcount = rows[-1] - rows[0] + 1
            colcount = columns[-1] - columns[0] + 1
            table = [[''] * colcount for _ in range(rowcount)]
            for index in selection:
                row = index.row() - rows[0]
                column = index.column() - columns[0]
                table[row][column] = index.data()
            stream = io.StringIO()
            csv.writer(stream).writerows(table)
            qApp.clipboard().setText(stream.getvalue())

    def saveas(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        filesave, _ = QFileDialog.getSaveFileName(self,"Save the data","","csv file (*.csv)", options=options)
        if filesave:
            __ , ext = os.path.splitext(filesave)
            if  ext == '.csv':
                filesave = filesave
            else:
                filesave = filesave+'.csv'
            df = pd.DataFrame(self.data_tab,columns=[self.head])
            df.to_csv(filesave, index=False)





class MyMainWindow(OctavvsMainWindow, Ui_MainWindow):
    DataUpdated = pyqtSignal(np.ndarray, int, list)
    SearchUp = pyqtSignal(str)
    Onefile = pyqtSignal(str)
    ProjUp = pyqtSignal()
    VisUp = pyqtSignal()
    ClusUp = pyqtSignal()

    @classmethod
    def program_name(cls):
        "Return the name of the program that this main window represents"
        return 'Clustering'

    def __init__(self,parent=None):
        super().__init__(parent)
        
        self.default_dir =  os.path.dirname(__file__) + "\\mcr\\"+'dat.initi'
        self.pushButtonLoadSpec.clicked.connect(self.Load_chose)
        self.lock_un(False)
        self.pushButtonExpandSpectra.clicked.connect(self.ExpandSpectra)
        self.pushButtonExpandProjection.clicked.connect(self.ExpandProjection)
        self.comboBoxCmaps.currentIndexChanged.connect(self.coloring)
        self.comboBoxMethod.currentIndexChanged.connect(self.ImageProjection)
        self.pushButtonLoadConc.clicked.connect(self.LoadPurest)
        self.horizontalSliderWavenumber.valueChanged.connect(self.Wavenumbercal)
        self.pushButtonCluster.clicked.connect(self.ClusteringCal)
        self.comboBoxVisualize.currentIndexChanged.connect(self.Cvisualize)
        self.pushButtonExpandVisual.clicked.connect(self.ExpandVis)
        self.pushButtonExpandAve.clicked.connect(self.ExpandAve)
        self.spinBoxNcluster.valueChanged.connect(self.Nclus_on)
        self.pushButtonExpandCluster.clicked.connect(self.ExpandCluster)
        self.pushButtonReduce.clicked.connect(self.Reduce)
        self.pushButtonSaveSpectra.clicked.connect(self.SaveAverage)
        self.pushButtonLoadWhite.clicked.connect(self.IMG)
        self.pushButtonRefresh.clicked.connect(self.Refresh)
        self.pushButtonNext.clicked.connect(self.Next)
        self.pushButtonPrevious.clicked.connect(self.Previous)
        self.pushButtonSC.clicked.connect(self.SC)
        self.lineEditHeight.returnPressed.connect(self.ValidationX)
        self.lineEditWidth.returnPressed.connect(self.ValidationY)
        self.pushButtonTable.clicked.connect(self.TableShow)
        self.pushButtonTableAve.clicked.connect(self.TableAve)
        self.Tableview = Table(self)

        self.About = DialogAbout()

        self.actionAbout.triggered.connect(self.About.show)
        self.Tableview.actionabout.triggered.connect(self.About.show)

        self.DataUpdated.connect(self.Tableview.main_data)
        self.SearchUp.connect(self.search_whole_folder)
        self.Onefile.connect(self.readfile)
        self.ProjUp.connect(self.ImageProjection)
        self.VisUp.connect(self.Cvisualize)
        self.ClusUp.connect(self.ClusteringCal)

        self.post_setup()

        self.checkBoxinvert.toggled.connect(self.Inverting)
        
        self.comboBoxImp.currentIndexChanged.connect(self.Select_spectra)
        self.spinBoxWlength.valueChanged.connect(self.Select_spectra)
        self.spinBoxPoly.valueChanged.connect(self.Select_spectra)


    def closeEvent(self, event):
        msgBox = QMessageBox()
        msgBox.setIcon(QMessageBox.Question)
        msgBox.setText("Warning")
        msgBox.setInformativeText('Are you sure to close the window ?')
        msgBox.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        msgBox.setDefaultButton(QMessageBox.No)
        reply = msgBox.exec_()
        if reply == QMessageBox.Yes:
            self.Tableview.close()
            self.About.close()
            plt.close('all')
#            qApp.quit()
        else:
            event.ignore()

    def TableShow(self):
        self.VisUp.emit()
        self.Tableview.close()
        # self.VisUp.emit()
        self.Tableview.show()


    def Load_chose(self):
        try:    
            infile = open(self.default_dir,'rb')
            dire = pickle.load(infile)
            infile.close()
        except:
            dire = os.path.dirname(__file__)


        if self.comboBoxSingMul.currentIndex() == 1:
            options = QFileDialog.Options()
            options |= QFileDialog.DontUseNativeDialog
            name, __ = QFileDialog.getOpenFileName(self,"Open Matrix File", dire,"Matrix File (*.mat)", options=options)
            self.foldername = dirname(name)
            if self.foldername:
                outfile = open(self.default_dir,'wb')
                pickle.dump(self.foldername,outfile)
                outfile.close()
                
                self.SearchUp.emit(self.foldername)
                self.pushButtonNext.setEnabled(True)
                self.pushButtonPrevious.setEnabled(True)
            else:
                msg = QMessageBox()
                msg.setIcon(QMessageBox.Critical)
                msg.setText('No file is loaded')
                msg.setInformativeText("Please select a file")
                msg.setWindowTitle("Warning")
                msg.setStandardButtons(QMessageBox.Ok )
                msg.exec_()

        elif self.comboBoxSingMul.currentIndex() == 0:
            self.LoadSpec()
            self.pushButtonNext.setEnabled(False)
            self.pushButtonPrevious.setEnabled(False)

        elif self.comboBoxSingMul.currentIndex() == 2:
             options = QFileDialog.Options()
             options |= QFileDialog.DontUseNativeDialog
             fileNames, _ = QFileDialog.getOpenFileNames(self,"Open Matrix File", dire,"Matrix File (*.mat)", options=options)
             if fileNames:
                 outfile = open(self.default_dir,'wb')
                 pickle.dump(dirname(fileNames[0]),outfile)
                 outfile.close()
                 self.foldername  = dirname(fileNames[0])

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
                     
                 self.lineEditTotal.setText(str(count))
                 self.lineEditFileNum.setText(str(1))   
                 self.pushButtonNext.setEnabled(True)
                 self.pushButtonPrevious.setEnabled(True)
                 self.Onefile.emit(name[0])
             else:
                msg = QMessageBox()
                msg.setIcon(QMessageBox.Critical)
                msg.setText('No file is loaded')
                msg.setInformativeText("Please select a file")
                msg.setWindowTitle("Warning")
                msg.setStandardButtons(QMessageBox.Ok )
                msg.exec_()



    def LoadSpec(self):
        try:    
            infile = open(self.default_dir,'rb')
            dire = pickle.load(infile)
            infile.close()
        except:
            dire = os.path.dirname(__file__)
            
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        self.fileName, __ = QFileDialog.getOpenFileName(self,"Open Matrix File", dire,"Matrix File (*.mat)")#, options=options)
        if self.fileName:
            outfile = open(self.default_dir,'wb')
            pickle.dump(dirname(self.fileName),outfile)
            outfile.close()
            
            self.foldername = dirname(self.fileName)
            self.lineEditFileNum.setText('1')
            self.lineEditTotal.setText('1')
            self.Onefile.emit(self.fileName)

        else:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Critical)
            msg.setText('No file is loaded')
            msg.setInformativeText("Please select a file")
            msg.setWindowTitle("Warning")
            msg.setStandardButtons(QMessageBox.Ok )
            msg.exec_()


    def Next(self):
        if int(self.lineEditFileNum.text()) != int(self.lineEditTotal.text()):
            df = pd.read_csv(self.foldername+"//Fileall.csv",header=None)
            count = int(self.lineEditFileNum.text())
            filename = df.iloc[count,1]
            self.lineEditFileNum.setText(str(count+1))
            # self.readfile(filename)
            self.Onefile.emit(filename)
        else:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Critical)
            msg.setText('No More File')
            msg.setInformativeText("Finish")
            msg.setWindowTitle("Warning")
            msg.setStandardButtons(QMessageBox.Ok )
            msg.exec_()


    def Previous(self):
        if int(self.lineEditFileNum.text()) != 1:
            df = pd.read_csv(self.foldername+"//Fileall.csv",header=None)
            count = int(self.lineEditFileNum.text())
            count -=1
            filename = df.iloc[count-1,1]
            self.lineEditFileNum.setText(str(count))
            self.Onefile.emit(filename)
        else:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Critical)
            msg.setText('This is the first file')
            msg.setInformativeText("Click Next for net")
            msg.setWindowTitle("Warning")
            msg.setStandardButtons(QMessageBox.Ok )
            msg.exec_()


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
        self.ProjUp.emit()
        self.VisUp.emit()
        self.ClusUp.emit()

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
        self.ProjUp.emit()
        self.VisUp.emit()
        self.ClusUp.emit()

    def SC(self):
        name = self.lineEditFilename.text()
        QApplication.primaryScreen().grabWindow(self.winId()).save(self.foldername+'/'+name+'_SC'+'.png')


    @pyqtSlot(str)
    def search_whole_folder(self, foldername):
        count = 0
        name =  {}
        a = [x[0] for x in os.walk(foldername)]
        for i in a:
            os.chdir(i)
            for file in glob.glob('*.mat'):
                name[count] = str(i+'/'+file)
                count += 1

        w = csv.writer(open(foldername+"//Fileall.csv", "w"))
        for key, val in sorted(name.items(), key=lambda item: item[1]):
            w.writerow([key, val])

        self.lineEditTotal.setText(str(count))
        self.lineEditFileNum.setText(str(1))
        self.Onefile.emit(name[0])

    @pyqtSlot(str)
    def readfile(self,fileName):
        self.img = None
        try:
            self.img = plt.imread(fileName.replace(fileName.split('.0')[-1],'.jpg'))
        except:
            pass

        try:
            self.sx,self.sy, self.pin ,self.wavenumber, self.spo = ff.readmat(fileName)
            self.spo = mc.nonnegative(self.spo)
            self.lineEditHeight.setText(str(self.sy))
            self.lineEditWidth.setText(str(self.sx))
        except:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Critical)
            msg.setText('No file is loaded')
            msg.setInformativeText('File can not be opened due to the the structure of data')
            msg.setWindowTitle("Warning")
            msg.setStandardButtons(QMessageBox.Ok )
            msg.exec_()
            self.lineEditFileNum.setText('')
            self.lineEditTotal.setText('')
            pass


        
        # self.index = np.random.randint(0,int(self.sx*self.sy),(20))
        kmeans = MiniBatchKMeans(n_clusters=8, random_state=0).fit(self.spo.T)
        
        self.raw = kmeans.cluster_centers_
        self.raw = self.raw.T

        
        self.Select_spectra()
        
        self.clear_all()
        self.lock_un(True)
        self.lineEditFilename.setText((basename(fileName).replace('.mat','')))
        self.lineEditDirSpectra.setText(fileName)

        


        self.PlotSpectraSample()

        self.labelMinwn.setText(str("%.2f" % np.min(self.wavenumber)))
        self.labelMaxwn.setText(str("%.2f" % np.max(self.wavenumber)))
        self.lineEditLength.setText(str(len(self.wavenumber)))
        self.lineEditWavenumber.setText(str("%.2f" % np.min(self.wavenumber)))

 
        self.ProjUp.emit()

        if self.img is not None:
            self.plot_White.canvas.ax.clear()
            self.plot_White.canvas.ax.imshow(self.img)
            self.plot_White.canvas.fig.tight_layout()
            self.plot_White.canvas.ax.set_axis_off() 
            self.plot_White.canvas.draw()


        filepure = fileName.replace('.mat','')
        filepure = filepure+'_purest.csv'
        if os.path.isfile(filepure) == True:
             msgBox = QMessageBox()
             msgBox.setIcon(QMessageBox.Question)
             msgBox.setText("Information")
             msgBox.setInformativeText('The '+basename(filepure)+' is available are you going to use it as purest spectra ?')
             msgBox.setStandardButtons(QMessageBox.Yes| QMessageBox.No)
             msgBox.setDefaultButton(QMessageBox.No)
             reply = msgBox.exec_()
             if reply == QMessageBox.Yes:
                self.AutoLoad(filepure)
             else:
                self.LoadPurest()
        else:
            self.LoadPurest()

    def Inverting(self):
        if self.comboBoxVisualize.currentIndex() == 0:
            self.PlotSpectraSample()
        else:
            self.VisUp.emit()
            self.ClusUp.emit()

        if hasattr(self,'df_spec'):
            self.ClusUp.emit()
            if self.pushButtonReduce.isEnabled():
                pass
            else:
                self.Reduce()

    def Select_spectra(self):
        win = int(self.spinBoxWlength.value())
        pol = int(self.spinBoxPoly.value())
        if self.comboBoxImp.currentIndex() == 0:
            self.sp = self.spo.copy()
            self.sp = mc.nonnegative(self.sp)
            self.p = self.pin.copy() 
        elif self.comboBoxImp.currentIndex() == 1:
            self.sp = savgol_filter(self.spo.T,win, polyorder = pol,deriv=1)
            self.sp = self.sp.T
            self.sp = mc.nonnegative(self.sp)
            self.p = self.sp.reshape(np.shape(self.pin),order='C')
                
        elif self.comboBoxImp.currentIndex() == 2:
            self.sp = savgol_filter(self.spo.T,win, polyorder = pol,deriv=2)
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

        self.ProjUp.emit()
        self.PlotSpectraSample()
        if hasattr(self,'df_spec'):
            self.ClusUp.emit()




    def PlotSpectraSample(self):
        win = int(self.spinBoxWlength.value())
        pol = int(self.spinBoxPoly.value())
        
        if self.comboBoxImp.currentIndex() == 0:
            self.yvis = self.raw.copy()
            self.spinBoxWlength.hide()
            self.spinBoxPoly.hide()
            
            self.spinBoxWlength.hide()
            self.spinBoxPoly.hide()
            self.labelWl.hide()
            self.labelPl.hide()

            
        elif self.comboBoxImp.currentIndex() == 1:
            self.yvis= savgol_filter(self.raw.T,win, polyorder = pol,deriv=1)
            self.yvis= self.yvis.T
            
            self.spinBoxWlength.show()
            self.spinBoxPoly.show()

            self.spinBoxWlength.show()
            self.spinBoxPoly.show()
            
            self.labelWl.show()
            self.labelPl.show()

            
        else:
            self.yvis= savgol_filter(self.raw.T,win, polyorder = pol,deriv=2)
            self.yvis= self.yvis.T
            self.labelWl.show()
            self.labelPl.show()
            self.spinBoxWlength.show()
            self.spinBoxPoly.show()


        self.DataUpdated.emit(np.column_stack((self.wavenumber,self.yvis)),0,['0'])
        
        self.yvis = mc.nonnegative(self.yvis)
        
        self.plot_specta.canvas.ax.clear()
        self.plot_specta.canvas.ax.plot(self.wavenumber,self.yvis)
        if self.checkBoxinvert.isChecked():
            self.plot_specta.Invert()
        self.plot_specta.canvas.fig.tight_layout()
        self.plot_specta.canvas.draw()
        self.ExpandSpectraU()


    def LoadPurest(self):
        sugest = self.foldername+'//'+self.lineEditFilename.text()+'_purest.csv'
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        filepurest, _ = QFileDialog.getOpenFileName(self,"Open Matrix File",sugest,"Matrix File (*.csv)", options=options)
        if filepurest:
            self.comboBoxVisualize.clear()
            self.comboBoxVisualize.addItem('Spectra and White Light Image')

            self.lineEditDirPurest.setText(filepurest)
            dfpurest = pd.read_csv(filepurest, header= None)
            
            #Old System Before implementing First or Second Derivatives
            if (int(self.lineEditLength.text())+int(self.sx*self.sy)) == len(dfpurest):
                self.compen = 0
                self.comboBoxImp.setCurrentIndex(0)
            
            #Old System after implementing First or Second Derivatives, but error
            #if  nr = 2, but mostly we never use nr = 2, but I keep this to read 
            #old data                
            elif (int(self.lineEditLength.text())+int(self.sx*self.sy)) + 1 == len(dfpurest):
                self.compen = 1
                self.spinBoxWlength.setValue(int(dfpurest.iloc[0,1]))
                self.spinBoxPoly.setValue(int(dfpurest.iloc[0,2]))
                self.comboBoxImp.setCurrentIndex(int(dfpurest.iloc[0,0]))
                self.Select_spectra()


            # New System, there is no error even we use nr = 2
            elif (int(self.lineEditLength.text())+int(self.sx*self.sy)) +3 == len(dfpurest):
                self.compen = 3
                self.spinBoxWlength.setValue(int(dfpurest.iloc[1,0]))
                self.spinBoxPoly.setValue(int(dfpurest.iloc[2,0]))
                self.comboBoxImp.setCurrentIndex(int(dfpurest.iloc[0,0]))
                self.Select_spectra()

                                 
                        
            
            self.df_spec = dfpurest.iloc[self.compen:int(self.lineEditLength.text())+self.compen,:]
            self.df_conc = dfpurest.iloc[self.compen+int(self.lineEditLength.text()):,:]
            self.ClusUp.emit()
            self.Nclus_on()
            for comp1 in range(0,len(self.df_conc.T)):
                self.comboBoxVisualize.addItem("component_"+str(comp1+1))

    def AutoLoad(self,filepurest):       
        self.comboBoxVisualize.clear()
        self.comboBoxVisualize.addItem('Spectra and White Light Image')
        self.lineEditDirPurest.setText(filepurest)
        dfpurest = pd.read_csv(filepurest, header= None)
        
        #Old System Before implementing First or Second Derivatives
        if (int(self.lineEditLength.text())+int(self.sx*self.sy)) == len(dfpurest):
            self.compen = 0
            self.comboBoxImp.setCurrentIndex(0)

        #Old System after implementing First or Second Derivatives, but error
        #if  nr = 2, but mostly we never use nr = 2, but I keep this to read 
        #old data                
        if (int(self.lineEditLength.text())+int(self.sx*self.sy)) + 1 == len(dfpurest):
            self.compen = 1
            self.spinBoxWlength.setValue(int(dfpurest.iloc[0,1]))
            self.spinBoxPoly.setValue(int(dfpurest.iloc[0,2]))
            self.comboBoxImp.setCurrentIndex(int(dfpurest.iloc[0,0]))
            self.Select_spectra()

        # New System, there is no error even we use nr = 2
        if (int(self.lineEditLength.text())+int(self.sx*self.sy)) + 3 == len(dfpurest):
            self.compen = 3
            self.spinBoxWlength.setValue(int(dfpurest.iloc[1,0]))
            self.spinBoxPoly.setValue(int(dfpurest.iloc[2,0]))
            self.comboBoxImp.setCurrentIndex(int(dfpurest.iloc[0,0]))
            self.Select_spectra()
        
        
        
        self.df_spec = dfpurest.iloc[self.compen:int(self.lineEditLength.text())+self.compen,:]
        self.df_conc = dfpurest.iloc[self.compen+int(self.lineEditLength.text()):,:]

        # self.ClusteringCal()
        self.ClusUp.emit()
        self.Nclus_on()
        for comp1 in range(0,len(self.df_conc.T)):
            self.comboBoxVisualize.addItem("component_"+str(comp1+1))

    def IMG(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self,"Open Image File", "",
                          "Image (*.jpg *.jpeg *.bmp *.png .tif *.tiff)", options=options)
        if fileName:
            self.img = plt.imread(fileName)
            self.plot_White.canvas.ax.clear()
            self.plot_White.canvas.ax.imshow(self.img)
            self.plot_White.canvas.fig.tight_layout()
            self.plot_White.canvas.ax.set_axis_off() 
            self.plot_White.canvas.draw()
            # self.Cvisualize()
            self.VisUp.emit()


    def ExpandCluster(self):
        fig = plt.figure("Segmentation Map", tight_layout={'pad':.5})
        fig.clear()
        ax = fig.gca()
        ax.imshow(self.mapping,cmap=self.cmap)
        plt.colorbar(ax.imshow(self.mapping,cmap=self.cmap))
        fig.show()


    def ExpandClusterU(self):
        if plt.fignum_exists("Segmentation Map"):
            fig = plt.figure("Segmentation Map", tight_layout={'pad':.5})
            fig.clear()
            ax = fig.gca()
            ax.imshow(self.mapping,cmap=self.cmap)
            plt.colorbar(ax.imshow(self.mapping,cmap=self.cmap))
            fig.canvas.draw_idle()
        else:
            pass


    @pyqtSlot()
    def ClusteringCal(self):
        nclus = int(self.spinBoxNcluster.value())
        method = int(self.comboBoxMethodCluster.currentIndex())

        if self.comboBoxNormalization.currentIndex() == 0:
            data = self.df_conc
        else:
            data = StandardScaler().fit_transform(self.df_conc)

        if method == 0:
            self.clus = KMeans(n_clusters = nclus, random_state=0).fit(data)
        else:
            self.clus = MiniBatchKMeans(n_clusters = nclus, random_state=0).fit(data)


        self.mapping = np.reshape(self.clus.labels_,(int(self.lineEditHeight.text()),
                                           int(self.lineEditWidth.text())))


        c1 =  str(self.comboBoxC1.currentText())
        c2 =  str(self.comboBoxC2.currentText())
        c3 =  str(self.comboBoxC3.currentText())
        c4 =  str(self.comboBoxC4.currentText())
        c5 =  str(self.comboBoxC5.currentText())
        c6 =  str(self.comboBoxC6.currentText())
        c7 =  str(self.comboBoxC7.currentText())
        c8 =  str(self.comboBoxC8.currentText())

        t1 = self.lineEditA1.text()
        t2 = self.lineEditA2.text()
        t3 = self.lineEditA3.text()
        t4 = self.lineEditA4.text()
        t5 = self.lineEditA5.text()
        t6 = self.lineEditA6.text()
        t7 = self.lineEditA7.text()
        t8 = self.lineEditA8.text()



        if nclus == 1:
            self.clis =[c1]
            self.label = [t1]

        elif nclus == 2:
            self.clis =[c1,c2]
            self.label = [t1,t2]

        elif nclus == 3:
            self.clis =[c1,c2,c3]
            self.label = [t1,t2,t3]

        elif nclus == 4:
            self.clis =[c1,c2,c3,c4]
            self.label = [t1,t2,t3,t4]

        elif nclus == 5:
            self.clis =[c1,c2,c3,c4,c5]
            self.label = [t1,t2,t3,t4,t5]

        elif nclus == 6:
            self.clis =[c1,c2,c3,c4,c5,c6]
            self.label = [t1,t2,t3,t4,t5,t6]

        elif nclus == 7:
            self.clis =[c1,c2,c3,c4,c5,c6,c7]
            self.label = [t1,t2,t3,t4,t5,t6,t7]

        elif nclus == 8:
            self.clis =[c1,c2,c3,c4,c5,c6,c7,c8]
            self.label = [t1,t2,t3,t4,t5,t6,t7,t8]

        if nclus > 8:
            self.cmap = str(self.comboBoxColorBig.currentText())
        else:
            self.cmap = matplotlib.colors.ListedColormap(self.clis)

        self.plotCluster.canvas.ax.clear()
        self.plotCluster.canvas.ax.imshow(self.mapping,cmap=self.cmap)
        self.plotCluster.canvas.fig.tight_layout()
        self.plotCluster.canvas.ax.set_axis_off() 
        self.plotCluster.canvas.draw()


        self.ExpandClusterU()

        self.color=[c1,c2,c3,c4,c5,c6,c7,c8]

        self.plotAverage.canvas.ax.clear()
        self.plotAverage.canvas.ax.set_prop_cycle(color=self.color)

        self.pushButtonReduce.setEnabled(True)

        indori = self.clus.labels_
        self.inde ={}
        self.spmean = np.zeros((nclus,len(self.sp)))
        for jj in range(0,nclus):
            self.inde[jj]=[i for i, e in enumerate(indori) if e == jj]
            self.spmean[jj,:]= np.mean(self.sp[:,self.inde[jj]], axis = 1)


        self.plotAverage.canvas.ax.plot(self.wavenumber, self.spmean.T )
        if self.checkBoxinvert.isChecked():
            self.plotAverage.Invert()


        if nclus <= 8:
            self.plotAverage.canvas.ax.legend(self.label,loc='best')
            self.plotAverage.canvas.fig.tight_layout()
        else:
            self.plotAverage.canvas.fig.tight_layout()

        self.ExpandAveU()   

        self.plotAverage.canvas.draw()
        anotclus=len(set(self.clis))
        self.lineEditAnotClus.setText(str(anotclus))



    def ExpandAve(self):
        fig = plt.figure("Average Spectra", tight_layout={'pad':.5})
        fig.clear()
        ax = fig.gca()
        ax.set_prop_cycle(color=self.color)
        ax.plot(self.wavenumber, self.spmean.T)
        if self.checkBoxinvert.isChecked():
            ax.invert_xaxis()
        ax.legend(self.label,loc='best')
        ax.set_xlabel("Wavenumber(1/cm)",fontsize=14)
        ax.set_ylabel("Absorption(arb. units)",fontsize=14)
        ax.tick_params(axis='both',direction='in', length=8, width=1)
        ax.tick_params(axis='both',which='major',labelsize=14)
        fig.show()

    def ExpandAveU(self):
        if plt.fignum_exists("Average Spectra"):
            fig = plt.figure("Average Spectra", tight_layout={'pad':.5})
            fig.clear()
            ax = fig.gca()
            ax.set_prop_cycle(color=self.color)
            ax.plot(self.wavenumber, self.spmean.T)
            if self.checkBoxinvert.isChecked():
                ax.invert_xaxis()
            ax.legend(self.label,loc='best')
            ax.set_xlabel("Wavenumber(1/cm)",fontsize=14)
            ax.set_ylabel("Absorption(arb. units)",fontsize=14)
            ax.tick_params(axis='both',direction='in', length=8, width=1)
            ax.tick_params(axis='both',which='major',labelsize=14)
            fig.canvas.draw_idle()    
        else:
            pass




    def ExpandSpectra(self):
        # plt.close('Spectra')
        fig = plt.figure('Spectra', tight_layout={'pad':.5})
        fig.clear()
        ax = fig.gca()
        if self.comboBoxVisualize.currentIndex() == 0:
            ax.plot(self.wavenumber,self.yvis)
            if self.comboBoxMethod.currentIndex() == 2:
                ax.axvline(x=self.wavenumv)
        else:
            ax.plot(self.wavenumber, self.datas)

        if self.checkBoxinvert.isChecked():
            ax.invert_xaxis()
        ax.set_xlabel("Wavenumber(1/cm)")#,fontsize=24)
        ax.set_ylabel("Absorption(arb. units)")#,fontsize=24)
        ax.tick_params(axis='both',direction='in', length=8, width=1)
        ax.tick_params(axis='both',which='major')#,labelsize=24)
        fig.show()

    def ExpandSpectraU(self):
        if plt.fignum_exists("Spectra"):
            fig = plt.figure("Spectra")
            ax = fig.gca()
            ax.clear()
            if self.comboBoxVisualize.currentIndex() == 0:
                ax.plot(self.wavenumber,self.yvis)
                if self.comboBoxMethod.currentIndex() == 2:
                    ax.axvline(x=self.wavenumv)
            else:
                ax.plot(self.wavenumber, self.datas)

            if self.checkBoxinvert.isChecked():
                ax.invert_xaxis()
                
            ax.set_xlabel("Wavenumber(1/cm)")#,fontsize=24)
            ax.set_ylabel("Absorption(arb. units)")#,fontsize=24)
            ax.tick_params(axis='both',direction='in', length=8, width=1)
            ax.tick_params(axis='both',which='major')#,labelsize=24)
            fig.canvas.draw_idle()
            
        else:
            pass

    def ExpandProjection(self):
        # plt.close("Image Projection")
        fig = plt.figure("Image Projection", tight_layout={'pad':.5})
        fig.clear()
        ax = fig.gca()
        ax.imshow(self.projection,str(self.comboBoxCmaps.currentText()))
        plt.colorbar(ax.imshow(self.projection,str(self.comboBoxCmaps.currentText())))
        fig.show()

    def ExpandProjU(self):
        if plt.fignum_exists("Image Projection"):
            fig = plt.figure("Image Projection")
            fig.clear()
            ax = fig.gca()
            ax.imshow(self.projection,str(self.comboBoxCmaps.currentText()))
            plt.colorbar(ax.imshow(self.projection,str(self.comboBoxCmaps.currentText())))
            fig.canvas.draw_idle()
        else:
            pass



    def coloring(self):
        self.plot_visual.canvas.ax.clear()
        self.plot_visual.canvas.ax.imshow(self.projection,str(self.comboBoxCmaps.currentText()))
        self.plot_visual.canvas.fig.tight_layout()
        self.plot_visual.canvas.ax.set_axis_off() 
        self.plot_visual.canvas.draw()
        self.VisUp.emit()
        self.ExpandProjU()


    @pyqtSlot()
    def ImageProjection(self):
        if  self.comboBoxMethod.currentIndex() == 0:
            self.lineEditWavenumber.setEnabled(False)
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

        self.plot_visual.canvas.ax.clear()
        self.plot_visual.canvas.ax.imshow(self.projection,str(self.comboBoxCmaps.currentText()))
        self.plot_visual.canvas.fig.tight_layout()
        self.plot_visual.canvas.ax.set_axis_off() 
        self.plot_visual.canvas.draw()

        self.ExpandProjU()
        self.ExpandSpectraU()

    def Wavenumbercal(self):
        nnow1 = ((np.max(self.wavenumber) - np.min(self.wavenumber))
        *float(self.horizontalSliderWavenumber.value())/10000.0 + np.min(self.wavenumber))
        nnow1 = "%.2f" % nnow1
        self.lineEditWavenumber.setText(str(nnow1))
        self.wavenumv = float(self.lineEditWavenumber.text())
        self.projection = ff.prowavenum(self.p,self.wavenumber,self.wavenumv)

        self.PlotSpectraSample()

        self.plot_visual.canvas.ax.clear()
        self.plot_visual.canvas.ax.imshow(self.projection,str(self.comboBoxCmaps.currentText()))
        self.plot_visual.canvas.fig.tight_layout()
        self.plot_visual.canvas.ax.set_axis_off() 
        self.plot_visual.canvas.draw()

        self.ExpandProjU()
        self.ExpandSpectraU()


    @pyqtSlot()
    def Cvisualize (self):
        if self.comboBoxVisualize.currentIndex() == 0:
            if self.img is not None:
                self.plotMultiVisual.canvas.ax.clear()
                self.plotMultiVisual.canvas.ax.imshow(self.img)
                self.plotMultiVisual.canvas.fig.tight_layout()
                self.plotMultiVisual.canvas.ax.set_axis_off() 
                self.plotMultiVisual.canvas.draw()

            self.PlotSpectraSample()
            # self.plot_specta.canvas.ax.clear()
            # self.plot_specta.canvas.ax.plot(self.wavenumber,self.sp[:,self.index])
            # self.plot_specta.canvas.fig.tight_layout()
            # self.plot_specta.canvas.draw()
            self.DataUpdated.emit(np.column_stack((self.wavenumber,self.yvis)),0,['0'])

        if 'component_' in self.comboBoxVisualize.currentText():
            import re
            val = int(re.search(r'\d+', self.comboBoxVisualize.currentText()).group()) - 1
            self.datap = self.df_conc.iloc[:,val].to_numpy()

            self.component = np.reshape(self.datap,(int(self.lineEditHeight.text()),
                                           int(self.lineEditWidth.text())))
            self.plotMultiVisual.canvas.ax.clear()
            self.plotMultiVisual.canvas.ax.imshow(self.component,str(self.comboBoxCmaps.currentText()))
            self.plotMultiVisual.canvas.fig.tight_layout()
            self.plotMultiVisual.canvas.ax.set_axis_off() 
            self.plotMultiVisual.canvas.draw()

            self.datas = self.df_spec.iloc[:,val].to_numpy()
            self.plot_specta.canvas.ax.clear()
            self.plot_specta.canvas.ax.plot(self.wavenumber, self.datas)
            if self.checkBoxinvert.isChecked():
                self.plot_specta.Invert()
            self.plot_specta.canvas.fig.tight_layout()
            self.plot_specta.canvas.draw()


            # print(len(self.datas))
            self.DataUpdated.emit(np.column_stack((self.wavenumber,self.datas)),0,[self.comboBoxVisualize.currentText()])


        self.ExpandSpectraU()
        self.ExpandVisU()


    def ExpandVis(self):
        # plt.close('Image Projection')
        fig = plt.figure('Image Visualization', tight_layout={'pad':.5})
        fig.clear()
        ax = fig.gca()
        if self.comboBoxVisualize.currentIndex() == 0:
            ax.imshow(self.img)
        else:
            ax.imshow(self.component,str(self.comboBoxCmaps.currentText()))
            plt.colorbar(ax.imshow(self.component,str(self.comboBoxCmaps.currentText())))
        fig.show()


    def ExpandVisU(self):
        if plt.fignum_exists('Image Visualization'):
            fig = plt.figure('Image Visualization')
            fig.clear()
            ax = fig.gca()
            if self.comboBoxVisualize.currentIndex() == 0:
                ax.imshow(self.img)
            else:
                ax.imshow(self.component,str(self.comboBoxCmaps.currentText()))
                plt.colorbar(ax.imshow(self.component,str(self.comboBoxCmaps.currentText())))
            fig.canvas.draw_idle()
        else:
            pass



    def Reduce(self):
        anotclus=len(set(self.clis))
        self.lineEditAnotClus.setText(str(anotclus))

        if self.spinBoxNcluster.value() != anotclus:
            sim = [item for item, count in collections.Counter(self.clis).items() if count > 1]
            ind = ff.listindexes(self.clis)
            new = np.zeros((len(self.wavenumber),len(sim)))
            suma = np.zeros((len(sim)))


            olddict = dict(zip(self.clis,self.spmean))
            for ii in range(0,len(sim)):
                for jj in range(len(ind[sim[ii]])):
                    auin = int(ind[sim[ii]][jj])
                    suma[ii] = int(suma[ii]) + len(self.inde[auin])
                    new[:,ii] = new[:,ii] + self.spmean[auin,:]*len(self.inde[auin])
                new[:,ii]/= suma[ii]

            #Color of new and multiple spectra
            nosim = [item for item, count in collections.Counter(self.clis).items() if count > 0]


            count = 0
            new_col = 0
            new_col = sim
            if len(sim) != anotclus:
                single = list(set(nosim)-set(sim))
                dumy = np.zeros((len(self.wavenumber),len(single)))
                for jj in single:
                    dumy[:,count] = olddict[jj]
                    count+=1
                    new_col.append(jj)
                new = np.concatenate((new,dumy), axis = 1)

            else:
                new_label =     [item for item, count in collections.Counter(self.label).items() if count > 0]

            new_label=[]
            for mem in new_col:
                index = self.color.index(mem)
                new_label.append(self.label[index])



            self.spmean = new.T
            self.label = new_label
            self.color = new_col

            self.plotAverage.canvas.ax.clear()
            self.plotAverage.canvas.ax.set_prop_cycle(color=self.color)
            self.plotAverage.canvas.ax.plot(self.wavenumber, self.spmean.T )
            self.plotAverage.canvas.ax.legend(self.label,loc= 'best')
            if self.checkBoxinvert.isChecked():
                self.plotAverage.Invert()
            self.plotAverage.canvas.fig.tight_layout()
            self.plotAverage.canvas.draw()

            self.ExpandAveU()
            self.pushButtonReduce.setEnabled(False)

    def Nclus_on(self):
        nclus = int(self.spinBoxNcluster.value())

        if nclus == 1:
            self.comboBoxC1.setEnabled(True)
            self.comboBoxC2.setEnabled(False)
            self.comboBoxC3.setEnabled(False)
            self.comboBoxC4.setEnabled(False)
            self.comboBoxC5.setEnabled(False)
            self.comboBoxC6.setEnabled(False)
            self.comboBoxC7.setEnabled(False)
            self.comboBoxC8.setEnabled(False)

            self.lineEditA1.setEnabled(True)
            self.lineEditA2.setEnabled(False)
            self.lineEditA3.setEnabled(False)
            self.lineEditA4.setEnabled(False)
            self.lineEditA5.setEnabled(False)
            self.lineEditA6.setEnabled(False)
            self.lineEditA7.setEnabled(False)
            self.lineEditA8.setEnabled(False)

        elif nclus == 2:
            self.comboBoxC1.setEnabled(True)
            self.comboBoxC2.setEnabled(True)
            self.comboBoxC3.setEnabled(False)
            self.comboBoxC4.setEnabled(False)
            self.comboBoxC5.setEnabled(False)
            self.comboBoxC6.setEnabled(False)
            self.comboBoxC7.setEnabled(False)
            self.comboBoxC8.setEnabled(False)

            self.lineEditA1.setEnabled(True)
            self.lineEditA2.setEnabled(True)
            self.lineEditA3.setEnabled(False)
            self.lineEditA4.setEnabled(False)
            self.lineEditA5.setEnabled(False)
            self.lineEditA6.setEnabled(False)
            self.lineEditA7.setEnabled(False)
            self.lineEditA8.setEnabled(False)

        elif nclus == 3:
            self.comboBoxC1.setEnabled(True)
            self.comboBoxC2.setEnabled(True)
            self.comboBoxC3.setEnabled(True)
            self.comboBoxC4.setEnabled(False)
            self.comboBoxC5.setEnabled(False)
            self.comboBoxC6.setEnabled(False)
            self.comboBoxC7.setEnabled(False)
            self.comboBoxC8.setEnabled(False)

            self.lineEditA1.setEnabled(True)
            self.lineEditA2.setEnabled(True)
            self.lineEditA3.setEnabled(True)
            self.lineEditA4.setEnabled(False)
            self.lineEditA5.setEnabled(False)
            self.lineEditA6.setEnabled(False)
            self.lineEditA7.setEnabled(False)
            self.lineEditA8.setEnabled(False)

        elif nclus == 4:
            self.comboBoxC1.setEnabled(True)
            self.comboBoxC2.setEnabled(True)
            self.comboBoxC3.setEnabled(True)
            self.comboBoxC4.setEnabled(True)
            self.comboBoxC5.setEnabled(False)
            self.comboBoxC6.setEnabled(False)
            self.comboBoxC7.setEnabled(False)
            self.comboBoxC8.setEnabled(False)

            self.lineEditA1.setEnabled(True)
            self.lineEditA2.setEnabled(True)
            self.lineEditA3.setEnabled(True)
            self.lineEditA4.setEnabled(True)
            self.lineEditA5.setEnabled(False)
            self.lineEditA6.setEnabled(False)
            self.lineEditA7.setEnabled(False)
            self.lineEditA8.setEnabled(False)

        elif nclus == 5:
            self.comboBoxC1.setEnabled(True)
            self.comboBoxC2.setEnabled(True)
            self.comboBoxC3.setEnabled(True)
            self.comboBoxC4.setEnabled(True)
            self.comboBoxC5.setEnabled(True)
            self.comboBoxC6.setEnabled(False)
            self.comboBoxC7.setEnabled(False)
            self.comboBoxC8.setEnabled(False)

            self.lineEditA1.setEnabled(True)
            self.lineEditA2.setEnabled(True)
            self.lineEditA3.setEnabled(True)
            self.lineEditA4.setEnabled(True)
            self.lineEditA5.setEnabled(True)
            self.lineEditA6.setEnabled(False)
            self.lineEditA7.setEnabled(False)
            self.lineEditA8.setEnabled(False)

        elif nclus == 6:
            self.comboBoxC1.setEnabled(True)
            self.comboBoxC2.setEnabled(True)
            self.comboBoxC3.setEnabled(True)
            self.comboBoxC4.setEnabled(True)
            self.comboBoxC5.setEnabled(True)
            self.comboBoxC6.setEnabled(True)
            self.comboBoxC7.setEnabled(False)
            self.comboBoxC8.setEnabled(False)

            self.lineEditA1.setEnabled(True)
            self.lineEditA2.setEnabled(True)
            self.lineEditA3.setEnabled(True)
            self.lineEditA4.setEnabled(True)
            self.lineEditA5.setEnabled(True)
            self.lineEditA6.setEnabled(True)
            self.lineEditA7.setEnabled(False)
            self.lineEditA8.setEnabled(False)


        elif nclus == 7:
            self.comboBoxC1.setEnabled(True)
            self.comboBoxC2.setEnabled(True)
            self.comboBoxC3.setEnabled(True)
            self.comboBoxC4.setEnabled(True)
            self.comboBoxC5.setEnabled(True)
            self.comboBoxC6.setEnabled(True)
            self.comboBoxC7.setEnabled(True)
            self.comboBoxC8.setEnabled(False)

            self.lineEditA1.setEnabled(True)
            self.lineEditA2.setEnabled(True)
            self.lineEditA3.setEnabled(True)
            self.lineEditA4.setEnabled(True)
            self.lineEditA5.setEnabled(True)
            self.lineEditA6.setEnabled(True)
            self.lineEditA7.setEnabled(True)
            self.lineEditA8.setEnabled(False)


        elif nclus == 8:
            self.comboBoxC1.setEnabled(True)
            self.comboBoxC2.setEnabled(True)
            self.comboBoxC3.setEnabled(True)
            self.comboBoxC4.setEnabled(True)
            self.comboBoxC5.setEnabled(True)
            self.comboBoxC6.setEnabled(True)
            self.comboBoxC7.setEnabled(True)
            self.comboBoxC8.setEnabled(True)

            self.lineEditA1.setEnabled(True)
            self.lineEditA2.setEnabled(True)
            self.lineEditA3.setEnabled(True)
            self.lineEditA4.setEnabled(True)
            self.lineEditA5.setEnabled(True)
            self.lineEditA6.setEnabled(True)
            self.lineEditA7.setEnabled(True)
            self.lineEditA8.setEnabled(True)


        elif nclus > 8:
            self.comboBoxC1.setEnabled(False)
            self.comboBoxC2.setEnabled(False)
            self.comboBoxC3.setEnabled(False)
            self.comboBoxC4.setEnabled(False)
            self.comboBoxC5.setEnabled(False)
            self.comboBoxC6.setEnabled(False)
            self.comboBoxC7.setEnabled(False)
            self.comboBoxC8.setEnabled(False)

            self.lineEditA1.setEnabled(False)
            self.lineEditA2.setEnabled(False)
            self.lineEditA3.setEnabled(False)
            self.lineEditA4.setEnabled(False)
            self.lineEditA5.setEnabled(False)
            self.lineEditA6.setEnabled(False)
            self.lineEditA7.setEnabled(False)
            self.lineEditA8.setEnabled(False)

        if nclus > 8:
            self.comboBoxColorBig.setEnabled(True)
            self.lineEditAnotClus.setEnabled(False)
        else:
            self.comboBoxColorBig.setEnabled(False)
            self.lineEditAnotClus.setEnabled(True)

    def SaveAverage(self):
        suggested = self.lineEditFilename.text()+'_clus.xls'
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        filesave, _ = QFileDialog.getSaveFileName(self,"QFileDialog.getSaveFileName()",suggested,"Excel Files(*.xls)", options=options)
        if filesave:
            __ , ext = os.path.splitext(filesave)
            if  ext == '.xls':
                filesave = filesave
            else:
                filesave = filesave+'.xls'
            df_save = pd.DataFrame({'Wavenumber':self.wavenumber})
            df_spec = pd.DataFrame(self.spmean.T, columns = [self.label])
            df_save = pd.concat([df_save, df_spec], axis=1, sort=False)
            df_save.to_excel(filesave,index=False)

    def TableAve(self):
        self.Tableview.close()
        self.DataUpdated.emit(np.column_stack((self.wavenumber, self.spmean.T)),0,self.color)
        self.Tableview.show()


    def clear_all(self):
        self.lineEditDirSpectra.setText('')
        self.lineEditDirPurest.setText('')
        self.lineEditFilename.setText('')
        self.lineEditLength.setText('')
        self.comboBoxMethod.setCurrentIndex(0)
        self.comboBoxCmaps.setCurrentIndex(0)
        self.horizontalSliderWavenumber.setSliderPosition(0)
        self.lineEditWavenumber.setText('')
        self.Refresh()
        self.comboBoxMethodCluster.setCurrentIndex(0)
        self.spinBoxNcluster.setValue(8)
        self.lineEditAnotClus.setText('')

        self.comboBoxVisualize.setCurrentIndex(0)

        self.comboBoxC1.setEnabled(False)
        self.comboBoxC2.setEnabled(False)
        self.comboBoxC3.setEnabled(False)
        self.comboBoxC4.setEnabled(False)
        self.comboBoxC5.setEnabled(False)
        self.comboBoxC6.setEnabled(False)
        self.comboBoxC7.setEnabled(False)
        self.comboBoxC8.setEnabled(False)

        self.lineEditA1.setEnabled(False)
        self.lineEditA2.setEnabled(False)
        self.lineEditA3.setEnabled(False)
        self.lineEditA4.setEnabled(False)
        self.lineEditA5.setEnabled(False)
        self.lineEditA6.setEnabled(False)
        self.lineEditA7.setEnabled(False)
        self.lineEditA8.setEnabled(False)

        self.plotMultiVisual.canvas.ax.clear()
        self.plotMultiVisual.canvas.draw()
        self.plot_visual.canvas.ax.clear()
        self.plot_visual.canvas.draw()
        self.plot_specta.canvas.ax.clear()
        self.plot_specta.canvas.draw()
        self.plot_White.canvas.ax.clear()
        self.plot_White.canvas.draw()
        self.plotAverage.canvas.ax.clear()
        self.plotAverage.canvas.draw()
        self.plotCluster.canvas.ax.clear()
        self.plotCluster.canvas.draw()

        self.comboBoxVisualize.clear()
        self.comboBoxVisualize.addItem('Spectra and White Light Image')


    def Refresh(self):
        self.comboBoxC1.setCurrentIndex(0)
        self.comboBoxC2.setCurrentIndex(1)
        self.comboBoxC3.setCurrentIndex(2)
        self.comboBoxC4.setCurrentIndex(3)
        self.comboBoxC5.setCurrentIndex(4)
        self.comboBoxC6.setCurrentIndex(5)
        self.comboBoxC7.setCurrentIndex(6)
        self.comboBoxC8.setCurrentIndex(7)

        self.lineEditA1.setText("")
        self.lineEditA2.setText("")
        self.lineEditA3.setText("")
        self.lineEditA4.setText("")
        self.lineEditA5.setText("")
        self.lineEditA6.setText("")
        self.lineEditA7.setText("")
        self.lineEditA8.setText("")



    def lock_un(self,stat):
        self.pushButtonLoadConc.setEnabled(stat)
        self.pushButtonPrevious.setEnabled(stat)
        self.pushButtonNext.setEnabled(stat)
        self.pushButtonExpandSpectra.setEnabled(stat)
        self.comboBoxMethod.setEnabled(stat)
        self.comboBoxCmaps.setEnabled(stat)
        self.pushButtonExpandProjection.setEnabled(stat)
        self.comboBoxVisualize.setEnabled(stat)
        self.pushButtonExpandVisual.setEnabled(stat)
        self.pushButtonExpandAve.setEnabled(stat)
        self.pushButtonSaveSpectra.setEnabled(stat)
        self.pushButtonExpandCluster.setEnabled(stat)
        self.pushButtonRefresh.setEnabled(stat)
        self.comboBoxMethodCluster.setEnabled(stat)
        self.spinBoxNcluster.setEnabled(stat)
        self.lineEditAnotClus.setEnabled(stat)
        self.pushButtonCluster.setEnabled(stat)
        self.pushButtonReduce.setEnabled(stat)
        self.comboBoxNormalization.setEnabled(stat)
        self.pushButtonSC.setEnabled(stat)
        self.pushButtonTable.setEnabled(stat)



def main():
    parser = argparse.ArgumentParser(
            description='Graphical application for clustering of MCR-ALS output.')
    MyMainWindow.run_octavvs_application(parser=parser)
