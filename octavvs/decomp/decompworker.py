#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 22:55:02 2019

@author: carl
"""

import traceback
import numpy as np
import scipy.signal, scipy.io
# from scipy.interpolate import interp1d
from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot

from octavvs.io import DecompositionData, Parameters
from octavvs.algorithms import decomposition
import time
import sklearn.cluster


class DecompWorker(QObject):
    """
    Worker thread class for the heavy parts of the decomposition
    """
    # Signals for when finished or failed, and for progress indication
    # done: iteration, spectra, concentrations, errors
    done = pyqtSignal(np.ndarray, np.ndarray, np.ndarray)
    stopped = pyqtSignal()
    failed = pyqtSignal(str, str)
    progress = pyqtSignal(int, float) # iteration, rel. error
    # done: iteration, spectra, concentrations, errors
    progressPlot = pyqtSignal(int, np.ndarray, np.ndarray, list)

    fileLoaded = pyqtSignal(int)
    loadFailed = pyqtSignal(str, str, str) # file, msg, details

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
            self.loadFailed.emit(file, str(e), '', False)
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


    def callDecomp(self, data, params):
        """
        Run decomposition stuff.
        """
        y = data.raw
        if data.decomposition_roi is not None:
            y = y[data.decomposition_roi, :]

        c_first = params.dcStartingPoint != 0
        nonneg = [True, True]

        if params.dcDerivative:
            y = scipy.signal.savgol_filter(
                y, window_length=params.dcDerivativeWindow,
                polyorder=params.dcDerivativePoly,
                deriv=params.dcDerivative, axis=1)
            nonneg[int(c_first)] = False

        base_error = np.linalg.norm(y - y.mean(0))**2
        if c_first:
            y = y.T

        if params.dcInitialValues == 'simplisma':
            simplisma_noise = 0.1
            initst = decomposition.simplisma(
                y.T, params.dcComponents, simplisma_noise)[0]
        elif params.dcInitialValues == 'kmeans':
            km = sklearn.cluster.MiniBatchKMeans(
                n_clusters=params.dcComponents).fit(y)
            initst = km.cluster_centers_
        else:
            raise ValueError('Unknown params.dcInitialValues')

        if c_first and params.dcDerivative:
            # If concentrations-first is combined with derivatives,
            # the initial concentrations need to be made positive.
            initst = np.abs(initst)

        if params.dcAlgorithm == 'mcr-als':
            acceleration = None
        elif params.dcAlgorithm == 'mcr-als-anderson':
            acceleration = 'Anderson'
        else:
            raise ValueError('Unknown params.dcAlgorithm')

        self.emitProgress(0, 0.)
        if c_first:
            data.add_decomposition_data(0, None, initst)
            self.progressPlot.emit(0, np.array(()), initst, [])
        else:
            data.add_decomposition_data(0, initst, None)
            self.progressPlot.emit(0, initst, np.array(()), [])

        update_interval = 3
        def cb_iter(it, errs, spectra, concentrations):
            self.emitProgress(len(errs), errs[-1] / base_error)
            t = time.monotonic()
            if t > cb_iter.iter_next:
                cb_iter.iter_next = t + update_interval
                if c_first:
                    concentrations, spectra = (spectra, concentrations)
                self.progressPlot.emit(it + 1, spectra, concentrations, errs)
        cb_iter.iter_next = time.monotonic()

        cweight = None
        if params.dcContrast:
            cweight = ('B' if params.dcContrastConcentrations == c_first \
                       else 'A', params.dcContrastWeight)

        # tt = time.monotonic()
        spectra, concentrations, errors = decomposition.mcr_als(
            y, initst, maxiters=params.dcIterations,
            nonnegative=nonneg,
            # tol_abs_error=params.dcTolerance * base_error,
            tol_rel_improv=params.dcTolerance * .01,
            tol_ups_after_best=30, callback=cb_iter,
            acceleration=acceleration, normalize='B' if c_first else 'A',
            contrast_weight=cweight)
        if c_first:
            concentrations, spectra = (spectra, concentrations)
        # print('Time:', time.monotonic()-tt)

        errors = np.asarray(errors)
        data.add_decomposition_data(len(errors), spectra, concentrations)
        data.set_decomposition_errors(errors)
        self.done.emit(spectra, concentrations, errors)
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
                # rcd_dir = params.directory if params.dirMode else None
                if not self.loadFile(data, fi):
                    continue

                data.set_decomposition_settings(params)
                self.callDecomp(data, params)
                data.save_rdc()

            self.batchDone.emit(True)
            return

        except InterruptedError:
            self.stopped.emit()
        except Exception as e:
            traceback.print_exc()
            self.failed.emit(repr(e), traceback.format_exc())
        self.batchDone.emit(False)

