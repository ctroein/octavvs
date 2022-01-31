#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  2 23:00:44 2021

@author: carl
"""

import os.path
import numpy as np
import h5py
import csv
import traceback
import time
import scipy.io

from .spectraldata import SpectralData

class DecompositionData(SpectralData):
    "SpectralData with addition of ROI/decomposition data"
    def __init__(self):
        super().__init__()
        # self.rdc_directory = None # None means dirname(curFile)
        self.clear_rdc()

    def clear_rdc(self):
        """
        Clear the ROI/decomposition/clustering data.

        Returns
        -------
        None.
        """
        self.roi = None # bool array or None if nothing is selected
        self.decomposition_settings = None # dict with decomp settings
        self.decomposition_roi = None # ROI actually used for decomp, or None
        # {iteration: {'spectra': array(component, wavenum),
        # 'concentrations': array(component, pixel) } }
        self.decomposition_data = None
        self.decomposition_errors = None # 1d array of error values
        self.clustering_settings = None # dict with clustering settings
        self.clustering_roi = None # ROI actually used for clustering, or None
        self.clustering_labels = None # 1d int array of labels
        self.clustering_annotations = None # {str: [labels]}

    # def set_rdc_directory(self, directory):
    #     self.rdc_directory = directory

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
                dups.add(self.filenames[i])
                dups.add(self.filenames[seen[f]])
            else:
                seen[f] = i
        return list(dups)

    # def rdc_filename(self, filetype='odd'):
    #     """
    #     Get the default name of the file where the current file's
    #     ROI, decomposition and clustering is stored.

    #     Parameters
    #     ----------
    #     filedir : str, optional
    #         The directory the file will be in, or None to use the input dir.

    #     Returns
    #     -------
    #     str
    #         Filename.
    #     """
    #     filetypes = ['odd']
    #     if filetype not in filetypes:
    #         raise ValueError("File type must be one of " + str(filetypes))
    #     if self.rdc_directory is None:
    #         return os.path.splitext(self.curFile)[0] + '.' + filetype
    #     if self.rdc_directory == '':
    #         raise ValueError("Directory cannot be ''")
    #     fn = os.path.splitext(os.path.basename(self.curFile))[0]
    #     return os.path.join(self.rdc_directory, fn) + '.' + filetype

    def read_matrix(self, filename):
        self.clear_rdc()
        super().read_matrix(filename)
        # try:
        #     self.load_rdc()
        # except OSError:
        #     pass
        # except Exception:
        #     traceback.print_exc()
        #     print('Warning: rdc auto-load failed')


    # ROI stuff
    def set_roi(self, roi):
        "Set ROI from array."
        if roi is None:
            self.roi = None
        elif roi.shape != (self.raw.shape[0],):
            raise ValueError('Wrong number of values for ROI (%s)' %
                             str(roi.shape))
        else:
            self.roi = roi if roi.any() else None

    # Decomposition stuff
    def set_decomposition_settings(self, params):
        """
        Populate decomposition_settings from a Parameters object,
        set the decomposition ROI depending on params.dcRoi,
        and clear stored data.

        Parameters
        ----------
        params : Parameters
            Only the dc settings will be copied, without the 'dc' prefix.

        Returns
        -------
        None

        Raises
        -------
        ValueError
            If the dcRoi parameter is inconsistent with current ROI
        """
        dcs = params.filtered('dc')
        roi = None
        if params.dcRoi == 'ignore':
            roi = None
        elif params.dcRoi == 'require':
            if self.roi is None:
                raise ValueError('ROI is required but not defined')
            roi = self.roi.copy()
        elif params.dcRoi == 'ifdef':
            roi = self.roi.copy() if self.roi is not None else None
        else:
            raise ValueError('Bad value for params.dcRoi')

        self.decomposition_settings = {k[2:]: v for k, v in dcs.items()}
        self.decomposition_settings['DateTime'] = \
            time.strftime('%Y-%m-%d %H:%M:%S %Z')
        self.decomposition_roi = roi
        self.decomposition_data = {}

    def set_decomposition_errors(self, errors):
        "Set the history of least squares error values"
        self.decomposition_errors = np.asarray(errors)

    def add_decomposition_data(self, iteration, spectra, concentrations):
        "Set data for a specific iteration; 0=initial"
        pixels = self.decomposition_roi.sum() if self.decomposition_roi \
            is not None else self.raw.shape[0]
        comps = self.decomposition_settings['Components']
        assert iteration <= self.decomposition_settings['Iterations']
        if iteration:
            assert spectra is not None and concentrations is not None
        else:
            assert spectra is not None or concentrations is not None
        if spectra is not None:
            assert spectra.shape == (comps, self.raw.shape[1])
            spectra = spectra.copy()
        if concentrations is not None:
            assert concentrations.shape == (comps, pixels)
            concentrations = concentrations.copy()
        self.decomposition_data[iteration] = {
            'spectra': spectra, 'concentrations': concentrations}

    # Clustering/annotation stuff
    def set_clustering_settings(self, params, roi):
        """
        Populate decomposition_settings from a Parameters object,
        copy the decomposition ROI, and clear stored clustering data.

        Parameters
        ----------
        params : Parameters
            Only the ca settings will be copied, without the 'ca' prefix.

        Returns
        -------
        None
        """
        cas = params.filtered('ca')
        self.clustering_settings = {k[2:]: v for k, v in cas.items()}
        self.clustering_settings['DateTime'] = \
            time.strftime('%Y-%m-%d %H:%M:%S %Z')
        self.clustering_roi = roi.copy() if roi is not None else None
        self.clustering_labels = None
        self.clustering_label_set = None
        self.clustering_annotations = {}

    def set_clustering_labels(self, labels, relabel):
        pixels = self.clustering_roi.sum() if \
            self.clustering_roi is not None else self.raw.shape[0]
        assert labels.shape == (pixels,)
        # Count labels, sort indexes and invert permutation array
        bc = np.bincount(labels)
        mapp = np.empty(bc.size, bc.dtype)
        mapp[np.argsort(bc)[::-1]] = np.arange(bc.size)
        self.clustering_labels = mapp[labels]
        self.clustering_label_set = set(self.clustering_labels)

    def get_annotation_clusters(self, annotation):
        if self.clustering_annotations is None:
            self.clustering_annotations = {}
        if annotation not in self.clustering_annotations:
            self.clustering_annotations[annotation] = set()
        return self.clustering_annotations[annotation].copy()

    def get_unannotated_clusters(self):
        used = set()
        for c in self.clustering_annotations.values():
            used.update(c)
        return self.clustering_label_set - used

    def set_annotation_clusters(self, annotation, clusters=None):
        self.get_annotation_clusters(annotation)
        if clusters is not None:
            cset = set(clusters)
            self.clustering_annotations[annotation] = cset
            for c in self.clustering_annotations.values():
                if c is not cset and cset.intersection(c):
                    print('Error: duplicated cluster id',
                          self.clustering_annotations)

    def del_annotation(self, annotation):
        if self.clustering_annotations is not None and \
                annotation in self.clustering_annotations:
                del self.clustering_annotations[annotation]

    # Loading and saving
    def load_rdc_(self, f, what, imgnum):
        grp = f['SpectralData/Image_%d' % imgnum]

        if what in ['all', 'roi']:
            if 'ROI' in grp:
                roi = grp['ROI']
                if roi.shape != (self.raw.shape[0],):
                    raise ValueError('Wrong number of values for ROI (%d)' %
                                     (len(roi)))
                roi = roi[:]
            else:
                roi = None

        if what in ['all', 'decomposition']:
            dcroi = None
            dcerrors = None
            if 'Decomposition' in grp:
                dc = grp['Decomposition']
                dcsettings = dict(dc.attrs.items())
                pixels = 0
                if 'ROI' in dc:
                    dcroi = dc['ROI'][:]
                    if dcroi.shape != (self.raw.shape[0],):
                        raise ValueError(
                            'Wrong number of values for '
                            'Decomposition ROI (%d)' % (len(dcroi)))
                    pixels = dcroi.sum()
                if not pixels:
                    pixels = self.raw.shape[0]
                comps = dcsettings['Components']
                dcdata = {}
                for ds in dc['Data'].values():
                    it = ds.attrs['Iteration']
                    conc = None
                    if 'Concentrations' in ds:
                        conc = ds['Concentrations'][:]
                        if conc.shape[1] != pixels:
                            raise ValueError('Concentration pixel count mismatch')
                        if conc.shape[0] != comps:
                            raise ValueError(
                                'Concentration component count mismatch')
                    spect = None
                    if 'Spectra' in ds:
                        spect = ds['Spectra'][:]
                        if spect.shape[1] != self.raw.shape[1]:
                            raise ValueError('Spectra wavenumber count mismatch')
                        if spect.shape[0] != comps:
                            raise ValueError('Spectra component count mismatch')
                    dcdata[it] = {'concentrations': conc, 'spectra': spect}
                if 'Errors' in dc:
                    dcerrors = dc['Errors'][:]
                    # if len(dcerrors) >= dcsettings['Iterations']:
                    #     raise ValueError('Too many elements in Errors')
            else:
                dcsettings = None
                dcdata = None

        if what in ['all', 'clustering']:
            caroi = None
            calabels = None
            casettings = None
            caannot = None
            if 'Clustering' in grp:
                ca = grp['Clustering']
                casettings = dict(ca.attrs.items())
                pixels = 0
                if 'ROI' in ca:
                    caroi = ca['ROI'][:]
                    if caroi.shape != (self.raw.shape[0],):
                        raise ValueError(
                            'Wrong number of values for '
                            'Clustering ROI (%d)' % (len(caroi)))
                    pixels = caroi.sum()
                if not pixels:
                    pixels = self.raw.shape[0]
                maxclust = casettings['Clusters']
                if 'Labels' in ca:
                    calabels = ca['Labels'][:]
                    if calabels.shape != (pixels,):
                        raise ValueError('Cluster label count mismatch')
                    if calabels.max() < 0 or calabels.max() >= maxclust:
                        raise ValueError('Cluster label range error')
                if 'Annotations' in ca:
                    caannot = {}
                    for ann in ca['Annotations'].values():
                        atxt = ann.attrs['Text']
                        avals = ann[:]
                        if len(avals) > 0 and (
                                min(avals) < 0 or np.max(avals) >= maxclust):
                            raise ValueError('Annotation cluster range error')
                        if atxt in caannot:
                            caannot[atxt].update(set(avals))
                        else:
                            caannot[atxt] = set(avals)

        if what in ['all', 'roi']:
            self.roi = roi
        if what in ['all', 'decomposition']:
            self.decomposition_settings = dcsettings
            self.decomposition_roi = dcroi
            self.decomposition_data = dcdata
            self.decomposition_errors = dcerrors
        if what in ['all', 'clustering']:
            self.clustering_settings = casettings
            self.clustering_roi = caroi
            self.clustering_labels = calabels
            self.clustering_label_set = set(calabels) if \
                calabels is not None else None
            # if caannot is not None:
            #     for k, v in caannot.items():
            #         caannot[k] = list(v)
            self.clustering_annotations = caannot

    def load_rdc(self, filename, *, what='all'):
        """
        Load ROI/decomp/clust from a named file.

        Parameters
        ----------
        filename : str
            A named file
        what : str
            One of 'all', 'roi', 'decomposition', 'clustering'

        Returns
        -------
        success : bool
            Whether data was loaded

        """
        # if filename is None:
        #     filename = self.rdc_filename()
        with h5py.File(filename, mode='r') as f:
            return self.load_rdc_(f, what, 0)

    def save_rdc_(self, f, what, imgnum):
        grp = f.require_group('SpectralData')
        grp.attrs.create('Creator', 'OCTAVVS')
        grp = grp.require_group('Image_%d' % imgnum)

        def replace_data(grp, name, arr):
            if name in grp:
                if arr is not None and grp[name].shape == arr.shape:
                    grp[name][...] = arr
                    return
                del grp[name]
            if arr is not None:
                grp.create_dataset(name, data=arr)

        if what in ['all', 'roi']:
            replace_data(grp, 'ROI', self.roi)

        if what in ['all', 'decomposition'] and \
            self.decomposition_settings is not None:
            dc = grp.require_group('Decomposition')
            for k, v in self.decomposition_settings.items():
                dc.attrs[k] = v
            replace_data(dc, 'ROI', self.decomposition_roi)
            ddg = dc.require_group('Data')
            for k, ds in ddg.items():
                it = ds.attrs['Iteration']
                if it not in self.decomposition_data \
                    or k != 'Iter_%d' % it:
                    del ddg[k]
            for it, vals in self.decomposition_data.items():
                ds = ddg.require_group('Iter_%d' % it)
                ds.attrs['Iteration'] = it
                replace_data(ds, 'Concentrations', vals['concentrations'])
                replace_data(ds, 'Spectra', vals['spectra'])
            replace_data(dc, 'Errors', self.decomposition_errors)

        if what in ['all', 'clustering'] and\
            self.clustering_settings is not None:
            ca = grp.require_group('Clustering')
            for k, v in self.clustering_settings.items():
                ca.attrs[k] = v
            replace_data(ca, 'ROI', self.clustering_roi)
            replace_data(ca, 'Labels', self.clustering_labels)
            if 'Annotations' in ca:
                del ca['Annotations']
            if self.clustering_annotations is not None:
                ag = ca.require_group('Annotations')
                for i, (ann, labels) in enumerate(
                        self.clustering_annotations.items()):
                    ds = ag.create_dataset('Annotation_%d' % i,
                                           data=list(labels))
                    ds.attrs['Text'] = ann


    def save_rdc(self, *, filename, what='all'):
        """
        Save data to a named file.

        Parameters
        ----------
        filename : str
            A named file; possibly from rdc_filename
        what : str
            One of 'all', 'roi', 'decomposition', 'clustering'

        Returns
        -------
        None.

        """
        with h5py.File(filename, mode='a') as f:
            return self.save_rdc_(f, what, 0)

    def save_roi(self, *, filename, fmt='.mat'):
        """
        Save ROI mask to a named file.

        Parameters
        ----------
        filename : str
            A named file
        fmt : str
            One of '.rdc', '.roi', '.mat', '.csv'

        Returns
        -------
        None.

        """
        print(f'save "{fmt}"')
        if fmt == '.rdc' or fmt == '.roi':
            self.save_rdc(filename=filename, what='roi')
            return
        if self.roi is None:
            roi = np.zeros(self.wh, dtype=int)
        else:
            roi = self.roi.reshape(self.wh).astype(int)
        if fmt == '.mat':
            scipy.io.savemat(filename, mdict={'ROI': roi},
                             appendmat=False, do_compression=True)
        elif fmt == '.csv':
            with open(filename, 'w') as f:
                writer = csv.writer(f)
                writer.writerows(roi)
        else:
            raise ValueError('Unknown ROI save format %s' % fmt)

    def get_headers_as_lists(self):
        headers = []
        headers.append(['#Input file', self.curFile])
        if self.clustering_settings['Input'] == 'decomposition':
            for k, v in self.decomposition_settings.items():
                headers.append(['#Decomposition ' + k, v])
        for k, v in self.clustering_settings.items():
            headers.append(['#Clustering ' + k, v])
        return headers

    def save_spectra_csv(self, writer, average):
        writer.writerows(self.get_headers_as_lists())
        if average:
            unused = self.get_unannotated_clusters()
            # # If all are unannotated, output cluster labels instead
            # by_cluster = len(unused) == len(self.clustering_label_set)
            header = ['Wavenumber/%s' % average]
            avg_indexes = []
            if self.clustering_roi is None:
                roi_indexes = np.arange(len(self.raw))
            else:
                roi_indexes = self.clustering_roi.nonzero()[0]
            if average == 'cluster':
                for c in self.clustering_label_set:
                    header.append('Cluster %d' % c)
                    avg_indexes.append(
                        roi_indexes[self.clustering_labels == c])
            else:
                for a, cs in self.clustering_annotations.items():
                    if cs:
                        header.append(a)
                        avg_indexes.append(roi_indexes[
                            np.isin(self.clustering_labels, list(cs))])
                if unused:
                    header.append('(unannotated)')
                    avg_indexes.append(roi_indexes[
                        np.isin(self.clustering_labels, list(unused))])
            writer.writerow(['Nspectra'] + [len(ixs) for ixs in avg_indexes])
            writer.writerow(header)
            for i, wn in enumerate(self.wavenumber):
                row = [wn]
                for aix in avg_indexes:
                    row.append(self.raw[aix, i].mean())
                writer.writerow(row)
        else:
            cluster_annots = {c: '(unannotated)' for c in
                              self.get_unannotated_clusters()}
            for a, cc in self.clustering_annotations.items():
                for c in cc:
                    cluster_annots[c] = a
            headers = [['x'], ['y'], ['Cluster'], ['Wavenumber/Annotation']]
            if self.clustering_roi is None:
                roi_indexes = np.arange(len(self.raw))
            else:
                roi_indexes = self.clustering_roi.nonzero()[0]
            for i, ix in enumerate(roi_indexes):
                if self.pixelxy is None:
                    xy = (ix % self.wh[0], ix // self.wh[0])
                else:
                    xy = self.pixelxy[ix]
                headers[0].append(xy[0])
                headers[1].append(xy[1])
                c = self.clustering_labels[i]
                headers[2].append(c)
                headers[3].append(cluster_annots[c])
            writer.writerows(headers)
            writer.writerows(np.hstack((self.wavenumber[:,None],
                                        self.raw[roi_indexes,:].T)))

    def save_annotated_spectra(self, filename, filetype, average):
        if average not in ['annotation', 'cluster', None]:
            raise ValueError("Invalid value for 'average'")
        if filetype == 'csv':
            with open(filename, 'w', newline='') as f:
                self.save_spectra_csv(csv.writer(f), average=average)
        else:
            raise ValueError('Invalid filetype %d' % filetype)


