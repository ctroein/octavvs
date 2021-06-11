#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  2 23:00:44 2021

@author: carl
"""

import os.path
import numpy as np
import h5py

from .spectraldata import SpectralData

class DecompositionData(SpectralData):
    "SpectralData with addition of ROI/decomposition data"
    def __init__(self):
        super().__init__()
        self.roi = None # bool array or None if nothing is selected

        self.decomposition_settings = None # dict with decomp settings
        self.decomposition_roi = None # ROI actually used for decomp, or None
         # list/1d-array of iterations at which spectra were stored
        self.saved_iterations = None
        self.spectra = None  # array (iteration, component, wavenum)
        self.concentrations = None # array (iteration, component, pixel)

    def get_duplicate_filenames(self):
        """
        Get list of filenames whose basename is not unique.

        Returns
        -------
        List of filenames with full paths
        """
        dups = set()
        seen = {}
        for i, f in enumerate([os.path.splitext(os.path.basename(f))[0]
                               for f in self.filenames]):
            if f in seen:
                dups.add(self.data.filenames[i])
                dups.add(self.data.filenames[seen[f]])
            else:
                seen[f] = i
        return list(dups)

    def rdc_filename(self, filedir=None, filetype='odd'):
        """
        Get the default name of the file where the current file's
        ROI, decomposition and clustering is stored.

        Parameters
        ----------
        filedir : str, optional
            The directory the file will be in, or None to use the input dir.

        Returns
        -------
        str
            Filename.
        """
        filetypes = ['odd']
        if filetype not in filetypes:
            raise ValueError("File type must be one of " + str(filetypes))
        if filedir is None:
            return os.path.splitext(self.curFile)[0] + '.' + filetype
        if not filedir:
            raise ValueError("Directory cannot be ''")
        fn = os.path.splitext(os.path.basename(self.curFile))[0]
        return os.path.join(filedir, fn) + '.' + filetype


    def set_roi(self, roi):
        "Set ROI from array. Returns whether is was really changed."
        if self.roi is not None and np.array_equal(self.roi, roi):
            return False
        if roi.shape != (self.raw.shape[0],):
            raise ValueError('Wrong number of values for ROI (%s)' %
                             str(roi.shape))

        self.roi = roi if roi.any() else None
        return True

    def set_decomposition_settings(self, params):
        """
        Populate decomposition_settings from a Parameters object

        Parameters
        ----------
        params : Parameters
            Only the dc settings will be copied, without the 'dc' prefix.

        Returns
        -------
        None.
        """
        dcs = params.filtered('dc')
        self.decomposition_settings = {k[2:]: v for k, v in dcs.items()}

    def load_rdc(self, filename=None, filedir=None):
        """
        Load ROI/decomp/clust from a named file or a file in the
        input/output directory.
        The default is curFile with extension .odd

        Parameters
        ----------
        filename : str, optional
            A named file, takes precedence
        filedir : str, optional
            Directory where to look for filepart(curFile)

        Returns
        -------
        success : bool
            Whether data was loaded

        """
        if filename is None:
            filename = self.rdc_filename(filedir)
        f = h5py.File(filename, mode='r')
        grp = f['SpectralData/Image_0']
        if 'ROI' in grp:
            roi = grp['ROI']
            if roi.shape != (self.raw.shape[0],):
                raise ValueError('Wrong number of values for ROI (%d)' %
                                 (len(roi)))
            roi = roi[:]
        else:
            roi = None

        dcroi = None
        if 'Decomposition' in grp:
            dc = grp['Decomposition']
            dcsettings = dict(dc.attrs.items())
            pixels = 0
            if 'ROI' in dc:
                dcroi = dc['ROI'][:]
                if roi.shape != (self.raw.shape[0],):
                    raise ValueError('Wrong number of values for '
                                     'Decomposition.ROI (%d)' % (len(dcroi)))
                pixels = dcroi.sum()
            if not pixels:
                pixels = self.raw.shape[0]
            iters = dc['Iterations'][:]
            conc = dc['Concentrations'][:]
            spectra = dc['Spectra'][:]

            if conc.shape != (len(iters), conc.shape[1], pixels):
                raise ValueError('Concentration matrix size mismatch')
            if spectra.shape != (len(iters), conc.shape[1], self.raw.shape[1]):
                raise ValueError('Decomposed spectra matrix size mismatch')
        else:
            dcsettings = None
            iters = None
            conc = None
            spectra = None

        f.close()

        self.roi = roi
        self.decomposition_settings = dcsettings
        self.decomposition_roi = dcroi
        self.saved_iterations = iters
        self.concentrations = conc
        self.spectra = spectra
        return True

    def save_rdc(self, filename=None, filedir=None, save_roi=True,
                 save_decomposition=True, save_clustering=True):
        """
        Save data to a named file or a file in the input/output directory.
        The default is based on curFile.

        Parameters
        ----------
        filename : str, optional
            A named file, takes precedence
        filedir : str, optional
            Directory where to look for filepart(curFile)

        Returns
        -------
        None.

        """
        if filename is None:
            if filedir == '':
                raise ValueError("ROI directory cannot be ''")
            filename = self.roi_filename(filedir)
        f = h5py.File(filename, mode='a')
        grp = f.require_group('SpectralData')
        grp.attrs.create('Creator', 'OCTAVVS')
        grp = grp.require_group('Image_0')

        def replace_data(grp, name, arr):
            if name in grp:
                if arr is not None and grp[name].shape != arr.shape:
                    grp[name][...] = arr
                else:
                    del grp[name]
            if arr is not None and name not in grp:
                grp.create_dataset(name, data=arr)

        if save_roi:
            replace_data(grp, 'ROI', self.roi)

        if save_decomposition:
            dc = grp.require_group('Decomposition')
            if self.decomposition_settings:
                for k, v in self.decomposition_settings.items():
                    dc.attrs[k] = v
            else:
                for k in dc.attrs.keys():
                    del dc.attrs[k]

            replace_data(dc, 'ROI', self.decomposition_roi)
            replace_data(dc, 'Iterations', self.saved_iterations)
            replace_data(dc, 'Concentrations', self.concentrations)
            replace_data(dc, 'Spectra', self.spectra)

        if save_clustering:
            if 'Clustering' in grp:
                del grp['Clustering']

        f.close()

    def save_roi(self, filename=None, filedir=None):
        "See save_rdc"
        self.save_rdc(filename=filename, filedir=filedir, save_roi=True,
                      save_decomposition=False, save_clustering=False)

    def save_decomposition(self, filename=None, filedir=None):
        "See save_rdc"
        self.save_rdc(filename=filename, filedir=filedir, save_roi=False,
                      save_decomposition=True, save_clustering=False)


