#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 22:55:02 2019

@author: carl
"""

import traceback
import os.path
from collections import namedtuple
import numpy as np
import scipy.signal, scipy.io
from scipy.interpolate import interp1d
from PyQt5.QtCore import QObject, pyqtSignal, pyqtSlot

from octavvs.io import SpectralData, Parameters
from octavvs.algorithms import baseline, correction, normalization, ptir

class PrepParameters(Parameters):
    """
    A class representing all the settings that can be made in the
    preprocessing UI and used to start a batch job.
    """

    def setAlgorithm(self, algo: str):
        algs = ['bassan', 'konevskikh', 'rasskazov']
        alg = next((a for a in algs if a in algo.lower()), None)
        if alg is None:
            raise ValueError('Algorithm must be one of '+str(algs))
        self.scAlgorithm = alg

    def load(self, filename):
        super().load(filename)
        self.setAlgorithm(self.scAlgorithm)



class PrepWorker(QObject):
    """
    Worker thread class for the heavy parts of the preprocessing: Scattering
    correction and the multiple-file batch processing.
    """
    # Signals for when processing is finished or failed, and for progress indication
    done = pyqtSignal(np.ndarray, np.ndarray, np.ndarray)
    stopped = pyqtSignal()
    failed = pyqtSignal(str, str)
    progress = pyqtSignal(int, int)
    progressPlot = pyqtSignal(np.ndarray, tuple)

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

    def loadReference(self, data, ref, otherref='', percentile=50):
        """
        Load an RMieSC reference spectrum, at wavenumbers that match the
        currently loaded file.
        """
        if ref == 'Mean':
            return data.raw.mean(0)
        elif ref == 'Percentile':
            return np.percentile(data.raw, percentile, axis=0)
        elif ref == 'Other':
            if otherref == '':
                raise RuntimeError('Specify a reference spectrum file')
            return correction.load_reference(data.wavenumber,
                                             matfilename=otherref)
        else:
            return correction.load_reference(data.wavenumber, what=ref.lower())

    def callACandSC(self, data, params, wn, y):
        """
        Run atmospheric correction and/or CRMieSC and/or mIRage correction
        """
        if params.mcDo:
            self.emitProgress(-5, 100)
            y = ptir.correct_mirage(
                wn, y,
                breaks=params.mcBreaks,
                endpoints=params.mcEndpoints,
                slopefactor=params.mcSlopefactor,
                pca_ncomp=params.mcPCAComponents if params.mcPCA else 0,
                soft_limits=params.mcSoftLimit,
                sl_level=params.mcSoftLimitLevel,
                chipweight=params.mcChipWeight)[0]

        if params.acDo:
            self.emitProgress(-1, 100)
            y = correction.atmospheric(
                wn, y, cut_co2=params.acSpline,
                smooth_win=9 if params.acSmooth else 0,
                atm=params.acReference,
                progressCallback=self.emitProgress)[0]

        if params.scDo:
            self.emitProgress(-2, 100)
            ref = self.loadReference(data, params.scRef,
                                     otherref=params.scOtherRef,
                                     percentile=params.scRefPercentile)
            yold = y
            clust = params.scClusters * (-1 if params.scStable else 1) if \
                params.scClustering else 0
#            print(params.scAmin, params.scAmax, params.scResolution)
            modelparams=dict(
                n_components=params.scPCAMax if \
                    params.scPCADynamic else params.scPCA,
                variancelimit=params.scPCAVariance*.01 if \
                    params.scPCADynamic else 0,
                a=np.linspace(params.scAmin, params.scAmax, params.scResolution),
                d=np.linspace(params.scDmin, params.scDmax, params.scResolution),
                bvals=params.scResolution,
                model=params.scAlgorithm,
                constantcomponent=params.scConstant,
                linearcomponent=params.scLinear)
            y = correction.rmiesc(
                    wn, y, ref,
                    iterations=params.scIters,
                    clusters=clust,
                    modelparams=modelparams,
                    weighted=False,
                    autoiterations=params.scAutoIters,
                    targetrelresiduals=1-params.scMinImprov*.01,
                    zeroregionpenalty=params.scPenalizeLambda if
                    params.scPenalize else None,
                    prefit_reference=params.scPrefitReference,
                    verbose=True,
                    progressCallback=self.emitProgress,
                    progressPlotCallback=self.progressPlot.emit)
            self.done.emit(wn, yold, y)

        return y

    def callSGFandSRandBC(self, params, wn, y):
        """
        Run three more preprocessing steps
        """
        if params.sgfDo:
            self.emitProgress(-3, 100)
            y = scipy.signal.savgol_filter(
                y, params.sgfWindow, params.sgfOrder, axis=1)

        if params.srDo:
            a = len(wn) - wn[::-1].searchsorted(params.srMax, 'right')
            b = len(wn) - wn[::-1].searchsorted(params.srMin, 'left')
            wn = wn[a:b]
            y = y[:, a:b]

        if params.bcDo:
            self.emitProgress(-4, 100)
            if params.bcMethod == 'rubberband':
                y = y - baseline.rubberband(
                    wn, y, progressCallback=self.emitProgress)
            elif params.bcMethod == 'concaverubberband':
                y = y - baseline.concaverubberband(
                    wn, y, iters=params.bcIters,
                    progressCallback=self.emitProgress)
            elif params.bcMethod == 'asls':
                y = y - baseline.asls(
                    y, lam=params.bcLambda, p=params.bcP,
                    progressCallback=self.emitProgress)
            elif params.bcMethod == 'arpls':
                y = y - baseline.arpls(
                    y, lam=params.bcLambdaArpls,
                    progressCallback=self.emitProgress)
            else:
                raise ValueError('unknown baseline correction method '+str(params.bcMethod))

        return wn, y

    @pyqtSlot(SpectralData, dict)
    def rmiesc(self, data, params):
        """ Run RMieSC, possibly preceded by atmospheric correction, on
        all or a subset of the raw data.
        Parameters:
            data: SpectralData object with raw data
            params: dictionary, mostly with things from PrepParameters (see code)
        """
        try:
            params['scDo'] = True
            params = namedtuple('rmiescParams', params.keys())(*params.values())
            if params.selected is not None:
                y = data.raw[params.selected]
            else:
                y = data.raw
            self.callACandSC(data, params, data.wavenumber, y)

        except InterruptedError:
            self.stopped.emit()
        except Exception as e:
            traceback.print_exc()
            self.failed.emit(repr(e), traceback.format_exc())


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
            abdata = {'AB': ab, 'wh': data.wh }
            if data.pixelxy is not None:
                abdata['xy'] = data.pixelxy
            scipy.io.savemat(outfile, abdata)

    @pyqtSlot(SpectralData, PrepParameters, str, bool)
    def bigBatch(self, data, params, folder, preservepath):
        """
        Run the batch processing of all the files listed in 'data'
        Parameters:
            data: SpectralData object with one or more files
            params: PrepParameters object from the user
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
                y = data.raw

                y = self.callACandSC(data, params, wn, y)
                wn, y = self.callSGFandSRandBC(params, wn, y)

                if params.normDo:
                    y = normalization.normalize_spectra(
                        params.normMethod, y, wn, wavenum=params.normWavenum)

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
#                filename = filename[0] + params.saveExt + filename[1]
                filename = filename[0] + params.saveExt + '.mat'

                self.saveCorrected(filename, params.saveFormat, data, wn, y)
            self.batchDone.emit(True)
            return

        except InterruptedError:
            self.stopped.emit()
        except Exception as e:
            traceback.print_exc()
            self.failed.emit(repr(e), traceback.format_exc())
        self.batchDone.emit(False)


    @pyqtSlot(SpectralData, PrepParameters, str)
    def createReference(self, data, params, outfile):
        """
        Create reference spectrum from all the files listed in 'data', using
        a subset of the processing steps.
        Parameters:
            data: SpectralData object with one or more files
            params: PrepParameters object from the user
            outfile: name of output file
        """
        params.scDo = False
        params.srDo = False
        wns = []
        ys = []
        try:
            for fi in range(len(data.filenames)):
                self.batchProgress.emit(fi, len(data.filenames))
                if not self.loadFile(data, fi):
                    continue

                wn = data.wavenumber
                y = data.raw

                if params.scRef == 'Percentile':
                    y = np.percentile(y, params.scRefPercentile, axis=0)[None, :]
                else:
                    y = y.mean(0)[None, :]

                y = self.callACandSC(data, params, wn, y)
                wn, y = self.callSGFandSRandBC(params, wn, y)
                if params.normDo:
                    y = normalization.normalize_spectra(
                        params.normMethod, y, wn,
                        wavenum=params.normWavenum)
                wns.append(wn)
                ys.append(y[0])

            # Do all images have the same wavenumbers?
            if all(np.array_equal(v, wns[0]) for v in wns):
                ys = np.median(ys, axis=0)
            else:
                w1 = min(v.min() for v in wns)
                w2 = max(v.max() for v in wns)
                maxres = max((len(v) - 1) / (v.max() - v.min()) for v in wns)
                wn = np.linspace(w1, w2, num=maxres * (w2 - w1) + 1)
                interpol = interp1d(np.concatenate(wns), np.concatenate(ys))
                ys = interpol(wn)

            ab = np.hstack((wn[:, None], ys[:, None]))
            scipy.io.savemat(outfile, {'AB': ab } )
            self.batchDone.emit(True)
            return

        except InterruptedError:
            self.stopped.emit()
        except Exception as e:
            traceback.print_exc()
            self.failed.emit(repr(e), traceback.format_exc())
        self.batchDone.emit(False)


class ABCWorker(QObject):
    """
    A smaller worker thread class for atmospheric, baseline and mirage
    correction only.
    """
    acDone = pyqtSignal(np.ndarray, np.ndarray, np.ndarray, np.ndarray)
    acFailed = pyqtSignal(str)
    bcDone = pyqtSignal(np.ndarray, np.ndarray, np.ndarray)
    bcFailed = pyqtSignal(str)
    mcDone = pyqtSignal(np.ndarray, np.ndarray, np.ndarray, dict)
    mcFailed = pyqtSignal(str)
    mcOptimizeDone = pyqtSignal(str, np.ndarray)
    mcOptimizeFailed = pyqtSignal(str)

    def __init__(self, parent=None):
        QObject.__init__(self, parent)
        self.haltBC = False
        self.haltMCOptimize = False

    @pyqtSlot(np.ndarray, np.ndarray, dict)
    def ac(self, wn, y, params):
        """
        Run baseline correction, emitting the processed data
        """
        try:
            corr, factors = correction.atmospheric(
                wn, y, cut_co2=params['cut_co2'],
                smooth_win=9 if params['smooth'] else 0,
                atm=params['ref'])
            self.acDone.emit(wn, y, corr, factors)
        except Exception:
            self.acFailed.emit(traceback.format_exc())

    def checkHaltBC(self, a, b):
        if self.haltBC:
            raise InterruptedError('interrupted by user')

    @pyqtSlot(np.ndarray, np.ndarray, str, dict)
    def bc(self, wn, y, method, params):
        try:
            self.checkHaltBC(0, 1)
            if method in {'rubberband', 'concaverubberband'}:
                corr = getattr(baseline, method)(wn, y, **params,
                              progressCallback=self.checkHaltBC)
            else:
                corr = getattr(baseline, method)(y, **params,
                              progressCallback=self.checkHaltBC)
            self.bcDone.emit(wn, y, corr)
        except InterruptedError:
            self.bcFailed.emit('')
        except Exception:
            self.bcFailed.emit(traceback.format_exc())

    @pyqtSlot(np.ndarray, np.ndarray, dict, dict)
    def mc(self, wn, y, params, runinfo):
        try:
            corr, runinfo['mcinfo'] = ptir.correct_mirage(wn, y, **params)
            self.mcDone.emit(wn, y, corr, runinfo)
        except InterruptedError:
            self.mcFailed.emit('')
        except Exception:
            self.mcFailed.emit(traceback.format_exc())

    def checkHaltMCOptimize(self, *args):
        if self.haltMCOptimize:
            raise InterruptedError('interrupted by user')


    @pyqtSlot(np.ndarray, np.ndarray, dict, str)
    def mcOptimize(self, wn, y, params, what):
        try:
            self.checkHaltMCOptimize()
            if what == 'pca':
                best = ptir.optimize_mirage_pca(
                    wn, y, callback=self.checkHaltMCOptimize, **params)
            elif what == 'sl':
                best = ptir.optimize_mirage_sl(
                    wn, y, callback=self.checkHaltMCOptimize, **params)
            self.mcOptimizeDone.emit(what, np.asarray(best[1]))
        except InterruptedError:
            self.mcOptimizeFailed.emit('')
        except Exception:
            self.mcOptimizeFailed.emit(traceback.format_exc())

