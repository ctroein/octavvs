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

from octavvs.io import DecompositionData, Parameters
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
    # done: iteration, conc, spectra, errs
    done = pyqtSignal(int, np.ndarray, np.ndarray, np.ndarray)
    stopped = pyqtSignal()
    failed = pyqtSignal(str, str)
    progress = pyqtSignal(list) # errors for all iterations
    progressPlot = pyqtSignal(int, np.ndarray, np.ndarray)

    fileLoaded = pyqtSignal(int)
    loadFailed = pyqtSignal(str, str, str, bool) # file, msg, details, is_warn

    batchProgress = pyqtSignal(int, int)
    batchDone = pyqtSignal(bool)

    def __init__(self, parent=None):
        QObject.__init__(self, parent)
        self.halt = False

    @pyqtSlot(int)
    def loadFile(self, data, num, rcd_dir=None):
        "Load file number num in the data object, emitting a signal on failure"
        try:
            file = data.filenames[num]
            if file == data.curFile:
                return True
            data.read_matrix(file)
        except (RuntimeError, FileNotFoundError) as e:
            self.loadFailed.emit(file, str(e), '', False)
        except Exception as e:
            self.loadFailed.emit(file, str(e), traceback.format_exc(), False)
        else:
            try:
                data.clear_rcd()
                fn = data.rcd_filename(filedir=rcd_dir)
                if os.path.exists(fn):
                    data.load_rcd(filename=fn)
            except Exception as e:
                self.loadFailed.emit(
                    file, str(e), traceback.format_exc(), True)

            self.fileLoaded.emit(num)
            return True
        return False

    def emitProgress(self, *pargs):
        "Combined progress signal and check for user interruption"
        if self.halt:
            raise InterruptedError('interrupted by user')
        self.progress.emit(*pargs)


    def callDecomp(self, data, params):
        """
        Run decomposition stuff
        """

        y = data.raw
        roi = data.decomposition_roi
        if roi is not None:
            y = y[roi, :]

        conc, pureix = decomposition.simplisma(
            y.T, params.dcComponents, params.dcSimplismaNoise)
        self.emitProgress([])
        initst = y[pureix,:]
        self.progressPlot.emit(0, conc, initst)

        mcr = pymcr.mcr.McrAR(max_iter=params.dcIterations,
                              tol_err_change=params.dcTolerance,
                              tol_increase=1., tol_n_increase=10,
                              tol_n_above_min=30)
        update_interval = 1
        def half_iter(C, ST, D, Dcalc):
            self.emitProgress(mcr.err)
            t = time.monotonic()
            if t - half_iter.iter_time > update_interval:
                half_iter.iter_time = t
                self.progressPlot.emit(mcr.n_iter, C.T, ST)
        half_iter.iter_time = time.monotonic()
        mcr.fit(y, ST=initst, post_iter_fcn=half_iter)

        # errs = np.asarray(mcr.err).reshape((-1, 2))
        self.done.emit(mcr.n_iter, mcr.C_opt_.T, mcr.ST_opt_,
                       np.asarray(mcr.err))
        return True


    @pyqtSlot(DecompositionData, Parameters)
    def decompose(self, data, params):
        """ Run decomposition on all or a subset of the raw data.
        Parameters:
            data: DecompositionData object with raw data
            params: Parameters with things set
        """
        try:
            self.callDecomp(data, params)
        except InterruptedError:
            self.stopped.emit()
        except Exception as e:
            traceback.print_exc()
            self.failed.emit(repr(e), traceback.format_exc())
        self.stopped.emit()



    @pyqtSlot(DecompositionData, Parameters)
    def startBatch(self, data, params):
        """
        Run the batch processing of all the files listed in 'data'
        Parameters:
            data: DecompositionData object with one or more files
            params: Parameters object from the user
        """
        try:
            for fi in range(len(data.filenames)):
                self.batchProgress.emit(fi, len(data.filenames))
                rcd_dir = params.directory if params.dirMode else None
                if not self.loadFile(data, fi, rcd_dir=rcd_dir):
                    continue

                data.set_decomposition_settings(params)
                self.callDecomp(data, params)
                ...
                # need to save the data

            self.batchDone.emit(True)
            return

        except InterruptedError:
            self.stopped.emit()
        except Exception as e:
            traceback.print_exc()
            self.failed.emit(repr(e), traceback.format_exc())
        self.batchDone.emit(False)

