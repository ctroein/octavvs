#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 22:55:02 2019

@author: carl
"""

import traceback
import os.path
import numpy as np
import scipy.signal, scipy.io
# from scipy.interpolate import interp1d
from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot

from octavvs.io import SpectralData, Parameters
from octavvs.algorithms import decomposition
import pymcr
import time

import sys
import logging
logger = logging.getLogger('pymcr')
logger.setLevel(logging.DEBUG)
stdout_handler = logging.StreamHandler(stream=sys.stdout)
stdout_format = logging.Formatter('%(message)s')
stdout_handler.setFormatter(stdout_format)
logger.addHandler(stdout_handler)

class DecompWorker(QObject):
    """
    Worker thread class for the heavy parts of the decomposition
    """
    # Signals for when finished or failed, and for progress indication
    done = pyqtSignal(np.ndarray, np.ndarray) # conc, spectra
    stopped = pyqtSignal()
    failed = pyqtSignal(str, str)
    progress = pyqtSignal(int, float)
    progressPlot = pyqtSignal(int, np.ndarray, np.ndarray)

    fileLoaded = pyqtSignal(int)
    loadFailed = pyqtSignal(str, str, str)

    batchProgress = pyqtSignal(int, int)
    batchDone = pyqtSignal(bool)

    def __init__(self, parent=None):
        QObject.__init__(self, parent)
        self.halt = False

    @pyqtSlot(int)
    def loadFile(self, data, num):
        "Load file number num in the data object, emitting a signal on failure"
        try:
            file = data.filenames[num]
            if file == data.curFile:
                return True
            data.read_matrix(file)
        except (RuntimeError, FileNotFoundError) as e:
            self.loadFailed.emit(file, str(e), '')
        except Exception as e:
            self.loadFailed.emit(file, str(e), traceback.format_exc())
        else:
            self.fileLoaded.emit(num)
            return True
        return False

    def emitProgress(self, *pargs):
        "Combined progress signal and check for user interruption"
        if self.halt:
            raise InterruptedError('interrupted by user')
        self.progress.emit(*pargs)


    def callDecomp(self, data, params, wn, y):
        """
        Run decomposition stuff
        """

        conc, pureix = decomposition.simplisma(
            y.T, params.dcComponents, params.dcSimplismaNoise)
        self.emitProgress(0, -1.)
        initst = y[pureix,:]
        self.progressPlot.emit(0, conc, initst)

        mcr = pymcr.mcr.McrAR(max_iter=params.dcIterations,
                              tol_err_change=params.dcTolerance,
                              tol_increase=1., tol_n_increase=10,
                              tol_n_above_min=30)
        update_interval = 1
        def half_iter(C, ST, D, Dcalc):
            self.emitProgress(mcr.n_iter, mcr.err[-1])
            t = time.monotonic()
            if t - half_iter.iter_time > update_interval:
                half_iter.iter_time = t
                self.progressPlot.emit(mcr.n_iter, C.T, ST)
        half_iter.iter_time = time.monotonic()
        mcr.fit(y, ST=initst, post_iter_fcn=half_iter)

        # self.emitProgress(0, -1)
        self.done.emit(mcr.C_opt_.T, mcr.ST_opt_)
        return mcr.C_opt_.T, mcr.ST_opt_


    @pyqtSlot(SpectralData, Parameters)
    def decompose(self, data, params):
        """ Run decomposition on all or a subset of the raw data.
        Parameters:
            data: SpectralData object with raw data
            params: Parameters with things set
        """
        try:
            self.callDecomp(data, params, data.wavenumber, data.raw)
        except InterruptedError:
            self.stopped.emit()
        except Exception as e:
            traceback.print_exc()
            self.failed.emit(repr(e), traceback.format_exc())
        self.stopped.emit()


    def saveCorrected(self, outfile, fmt, data, wn, y):
        if fmt == 'Quasar.mat':
            out = {'y': y, 'wavenumber': wn}
            if data.pixelxy is not None:
                map_x = np.array([x for (x,y) in data.pixelxy])
                map_y = np.array([y for (x,y) in data.pixelxy])
            else:
                map_x = np.tile(data.wh[0], data.wh[1])
                map_y = np.repeat(range(data.wh[0]), data.wh[1])
            out['map_x'] = map_x[:, None]
            out['map_y'] = map_y[:, None]
            scipy.io.savemat(outfile, out)
        else:
            ab = np.hstack((wn[:, None], y.T))
            scipy.io.savemat(outfile, {'AB': ab, 'wh': data.wh } )

    @pyqtSlot(SpectralData, Parameters, str, bool)
    def startBatch(self, data, params, folder, preservepath):
        """
        Run the batch processing of all the files listed in 'data'
        Parameters:
            data: SpectralData object with one or more files
            params: Parameters object from the user
            folder: output directory
            preservepath: if True, all processed files whose paths are under
            data.foldername will be placed in the corresponding subdirectory
            of the output directory.
        """
        try:
            for fi in range(len(data.filenames)):
                self.batchProgress.emit(fi, len(data.filenames))
                if not self.loadFile(data, fi):
                    continue

                wn = data.wavenumber

                CT, ST = self.callDecomp(data, params, wn, data.raw)

                # Figure out where to save the file
                filename = data.curFile
                if preservepath and filename.startswith(data.foldername):
                    filename = filename[len(data.foldername):]
                    filename = folder + filename
                    os.makedirs(os.path.dirname(filename), exist_ok=True)
                else:
                    filename = os.path.join(folder, os.path.basename(filename))
                # Add the extension
                filename = os.path.splitext(filename)
                filename = filename[0] + params.saveExt + '.mat'

                # self.saveCorrected(filename, params.saveFormat, data, wn, y)
            self.batchDone.emit(True)
            return

        except InterruptedError:
            self.stopped.emit()
        except Exception as e:
            traceback.print_exc()
            self.failed.emit(repr(e), traceback.format_exc())
        self.batchDone.emit(False)


