#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 18 14:45:53 2021

@author: carl
"""

import numpy as np
import scipy, scipy.optimize, scipy.spatial.distance
import math
import time
import os
from threadpoolctl import threadpool_limits
from functools import partial
from .util import pca_nipals

try:
    # os.environ["NUMBA_DISABLE_INTEL_SVML"] = "1"
    import numba
    import numba_nnls
    nnls = numba_nnls.nnls_007_111
    # nnls = numba_nnls.nnls_112_114 # crashes? icc_rt?
    use_numba = True
except ModuleNotFoundError:
    nnls = scipy.optimize.nnls
    use_numba = False
    print("INFO: numba_nnls not found; for improved MCR-ALS performance "
          "consider installing https://github.com/Nin17/numba-nnls")

def simplisma(d, nr, f):
    """
    The SIMPLISMA algorithm for finding a set of 'pure' spectra to serve
    as starting point for MCR-ALS etc.
    Reference Matlab Code:
        J. Jaumot, R. Gargallo, A. de Juan, R. Tauler,
        Chemometrics and Intelligent Laboratoty Systems, 76 (2005) 101-110

    Parameters
    ----------
    d : array(nwavenums, nspectra)
        input spectra.
    nr : int
        number of output components.
    f : float
        noise threshold.

    Returns
    -------
    spout: array(nr, nspectra)
        concentration profiles of 'purest' spectra.
    imp : array(nr, dtype=int)
        indexes of the 'purest' spectra.
    """

    nrow = d.shape[0]
    ncol = d.shape[1]
    s = d.std(axis=0)
    m = d.mean(axis=0)
    mf = m + m.max() * f
    p = s / mf

    # First Pure Spectral/Concentration profile
    imp = np.empty(nr, dtype=np.int64)
    imp[0] = p.argmax()

    #Calculation of correlation matrix
    l2 = s**2 + mf**2
    dl = d / np.sqrt(l2)
    c = (dl.T @ dl) / nrow

    #calculation of the first weight
    w = (s**2 + m**2) / l2
    p *= w
    #calculation of following weights
    dm = np.zeros((nr+1, nr+1))
    for i in range(1, nr):
        dm[1:i+1, 1:i+1] = c[imp[:i], :][:, imp[:i]]
        for j in range(ncol):
            dm[0, 0] = c[j, j]
            dm[0, 1:i+1] = c[j, imp[:i]]
            dm[1:i+1, 0] = c[imp[:i], j]
            w[j] = np.linalg.det(dm[0:i+1, 0:i+1])
        imp[i] = (p * w).argmax()

    ss = d[:,imp]
    spout = ss / np.sqrt(np.sum(ss**2, axis=0))
    return spout.T, imp

def simplisma_with_reference(data, nr, f, refdata):
    """
    Like @simplisma but with m reference spectra that become the first
    m spectra of the output.

    Parameters
    ----------
    data : array(nspectra, nwavenums)
        Input spectra. Note the transposed order vs simplisma.
    refdata : array(nrefspectra, nwavenums)
        Reference spectra.
    nr : int
        Number of output components, nr >= len(refdata)
    f : float
        Noise threshold.

    Returns
    -------
    spout: array(nr, nspectra)
        concentration profiles of reference and 'purest' spectra.
    """
    s_comps = nr - len(refdata)
    if s_comps < 0:
        raise ValueError("Number of simplisma output spectra cannot be "
                         "lower than number of reference spectra")
    if s_comps == 0:
        return refdata.copy()

    # s_data = data
    # for i in range(len(refdata)-1, 0-1, -1):
    #     proj = np.dot(s_data, refdata[i]) / (refdata[i] ** 2).sum()**.5
    #     print("proj before",i, (proj**2).sum()**.5)
    #     s_data = s_data - proj[:, None] @ refdata[i][None, :]
    # for i in range(len(refdata)-1, 0-1, -1):
    #     proj = np.dot(s_data, refdata[i]) / (refdata[i] ** 2).sum()**.5
    #     print("proj after",i, (proj**2).sum()**.5)
    # s_init = simplisma(s_data.T, s_comps, f)
    # return np.vstack((refdata, data[s_init[1]]))

    s_data = data
    for i in range(len(refdata)-1, 0-1, -1):
        proj = np.dot(s_data, refdata[i]) / (refdata[i] ** 2).sum()**.5
        s_data = s_data - proj[:, None] @ refdata[i][None, :]

    # Components first, then spectra
    c_other = simplisma(s_data, s_comps, f)[0]
    # s_init = np.linalg.lstsq(s_pre.T, data, rcond=-1)[0]
    # import matplotlib.pyplot as plt
    # plt.figure("initcomps")
    # plt.cla()
    # plt.plot(c_other.T)

    s_init = []
    for i in range(s_data.shape[1]):
        r, _ = nnls(c_other.T, s_data[:, i])
        s_init.append(r)
    s_init = np.array(s_init).T
    s_init = s_init / s_init.mean(1, keepdims=True)
    return np.vstack((refdata, s_init))

    # c_ref = []
    # for i in range(len(data)):
    #     r, _ = scipy.optimize.nnls(refdata.T, data[i])
    #     c_ref.append(r)
    # print("cs", np.array(c_ref).T.shape, c_other.shape)
    # comps = np.vstack((np.array(c_ref).T, c_other))
    # import matplotlib.pyplot as plt
    # plt.figure("initcomps")
    # plt.cla()
    # plt.plot(comps.T)
    # s_init = []
    # for i in range(data.shape[1]):
    #     r, _ = scipy.optimize.nnls(comps.T, data[:, i])
    #     s_init.append(r)
    # print("s", np.array(s_init).T.shape, refdata.shape)
    # s_init = np.array(s_init).T
    # s_init[:len(refdata), :] = refdata
    # s_init = s_init / s_init.mean(1, keepdims=True)

    return s_init


def clustersubtract(data, components, skewness=100, power=2, verbose=False):
    """
    Create initial spectra for MCR-ALS based on successively removing
    what appears to be the strongest remaining component.

    Parameters
    ----------
    data : array (nspectra, nfeatures)
        Spectral data.
    components : int
        Number of components to return.
    skewness : float, optional
        Asymmetry between positive and negative residuals when computing
        how much of the previous component to remove from the data.
        The default is 100.
    power : float, optional
        The sum of residuals is raised to this power before summation to
        determine the leading remaining component.

    Returns
    -------
    initial_spectra : array (components, nfeatures)
    """
    def typical_cluster(data, first):
        # draw sqrt(n) random data points, r
        # find closest in r for each s in data
        # for r with most s, return mean of those s
        r = np.random.choice(
            len(data), min(200, math.floor(math.sqrt(len(data)))))
        rd = data[r]
        nearest = scipy.spatial.distance.cdist(
            rd, data, 'cosine').argmin(axis=0)
        # Mean of those who are nearest the biggest cluster
        if first:
            selected = np.bincount(nearest).argmax()
        else:
            sums = data.sum(1) ** power
            selected = np.bincount(nearest, weights=sums).argmax()
        return data[nearest == selected].mean(0)

    comps = []
    sgn = np.ones_like(data, dtype=bool)
    for c in range(components):
        tc = typical_cluster(data, c == 0)
        tc = np.maximum(tc, 0)
        tc = tc / (tc * tc).sum() ** .5

        comps.append(tc)
        for i in range(10):
            ww = 1 * sgn + skewness * ~sgn
            a = (data * ww * tc).sum(1) / (ww * tc * tc).sum(1)
            oldsgn = sgn
            sgn = data > a[:, None] @ tc[None, :]
            if verbose:
                chg = (sgn != oldsgn).sum()
                print(f'clsub iter {c:3d} {i:2d} changed {chg}')
            if np.array_equal(sgn, oldsgn):
                break
        data = data - a[:, None] @ tc[None, :]
    return np.array(comps)

def numpy_scipy_threading_fix_(func):
    """
    This decorator for mcr_als prevents threading in BLAS if scipy's old NNLS
    is used, because for some reason NNLS won't be parallelized if called
    shortly after lstsq or @. This makes a *massive* difference to the
    time needed for Anderson acceleration, where the BLAS calls themselves
    take negligible time. For mixed NNLS/lstsq solving (of MCR-ALS on
    derivatives) it's less obvious whether NNSL or lstsq should be allowed
    to be parallelized.
    Note: This issue is seen on my Linux setups and might be highly version-
    dependent.
    Update 2025: The rewritten non-Fortran NNLS doesn't have this problem
    but is slower overall. Solution: Numba and numba_nnls,
    https://github.com/Nin17/numba-nnls
    """
    def check(*args, **kwargs):
        th = kwargs.get('blas_threads', 1)
        kwargs.pop('blas_threads', None)
        if (('nonnegative' not in kwargs or np.any(kwargs['nonnegative']))
            and th):
            with threadpool_limits(th, 'blas'):
                return func(*args, **kwargs)
        else:
            return func(*args, **kwargs)
    return check

if use_numba:
    @numba.jit
    def multi_nnls(A : np.ndarray, bs : np.ndarray, out : np.ndarray):
        err2 = 0
        for i in range(len(bs)):
            out[i], res = nnls(A, bs[i])
            err2 = err2 + res * res
        return err2
else:
    def multi_nnls(A : np.ndarray, bs : np.ndarray, out : np.ndarray):
        """
        Run NNLS on multiple vectors.

        Parameters
        ----------
        A : np.ndarray
            (m, n) input to nnls.
        bs : np.ndarray
            (k, m) set of k vectors.
        out : np.ndarray
            (k, n) set of k solution vectors.
        Returns
        -------
        err2 : float
            Sum of all squared residuals.
        """
        err2 = 0
        for i in range(len(bs)):
            out[i], res = nnls(A, bs[i])
            err2 = err2 + res * res
        return err2


# @numpy_scipy_threading_fix_
def mcr_als(sp, initial_A, *, maxiters=100, nonnegative=(True, True),
            tol_abs_error=0, tol_rel_improv=None, tol_rel_iters=10,
            tol_iters_after_best=None,
            maxtime=None, callback=None, acceleration=None, normalize=None,
            fixed_components_A=None, fixed_features_A=None,
            mutable_components_B=slice(None),
            initial_B=None,
            modify_before_Astep=None,
            contrast_weight=None, return_time=False,
            starttime=None):
    """
    Perform MCR-ALS nonnegative matrix decomposition on the matrix sp.
    The input array is decomposed as A@B where A and B are spectra and
    concentrations or vice versa.

    Parameters
    ----------
    sp : array(nsamples, nfeatures)
        Spectra to be decomposed.
    initial_A : array(ncomponents, nfeatures)
        Initial spectra (or concentrations).
    maxiters : int
        Maximum number of iterations.
    nonnegative : pair of bool, default (True, True)
        True if (initial, other) components must be non-negative
    tol_abs_error : float, optional
        Error target (mean square error).
    tol_rel_improv : float, optional
        Stop at average relative improvement of less than tol_rel_improv.
    tol_rel_iters : int, optional
        Number of iterations to evaluate tol_rel_improv over.
    tol_iters_after_best : int, optional
        Stop after this many iterations since last best error.
    maxtime : float, optional
        Stop after this many seconds of process time have elapsed
    callback : func(it : int, err : list(float), A : array, B : array)
        Progress callback for every iteration.
    acceleration : str, optional
        None or 'Anderson'.
        Anderson acceleration operates on whole iterations (A or B updates),
        mixing earlier directions to step towards the fixed point. This
        implementation restarts from basic updates when those would be
        better.
    normalize : str, optional
        Which matrix to l2 normalize: None, 'A' or 'B'
    fixed_components_A : int, optional
        Number of components of initial_A that are kept as-is; e.g.
        n endmembers for unmixing. The first n components are kept unchanged
        and will be present in the returned A and B. The caller will have to
        reorder the input/output as needed.
    fixed_features_A : int, optional
        Number of initial features of sp and A that are kept as-is. These
        features will typically be used to push B in a desired direction.
    modify_before_Astep : None or func(A : array, B : array, sp : array)
        Function called before updates of A, intended for rewriting of the
        (original) input data (sp) or the B array, e.g. for smoothing of
        B or updating averaged target values.
    mutable_components_B : slice, range or bool array, optional
        Components of B that will be updated. Default: all components.
        This is useful when some components have already been fitted.
    initial_B : array(ncomponents, nsamples), optional
        Necessary when mutable_components_B is used; the values for the
        non-mutable components will be taken from here.
    contrast_weight : (str, float), optional, deprecated
        Increase contrast in one matrix by mixing the other, named matrix
        ('A' or 'B') with the mean of its vectors. If A is spectra,
        try contrast_weight=('B', 0.05) to increase spectral contrast.
        See Windig and Keenan, Applied Spectroscopy 65: 349 (2011).
    return_time : bool, default False
        Measure and return process_time at each iteration.

    Returns
    -------
    A : array(ncomponents, nfeatures)
        Spectra (at lowest error)
    B : array(ncomponents, nsamples)
        Concentrations at lowest error
    error : list(float)
        Mean square error at every iteration
    process_time : list(float)
        Time relative start at each iteration, only if return_time is True.
    """
    if normalize not in [None, 'A', 'B']:
        raise ValueError("Normalization must be None, 'A' or 'B'")

    nsamp, nfeat = sp.shape
    ncomp = initial_A.shape[0]

    if initial_A.shape[1] != nfeat:
        raise ValueError("Mismatching size of data and initial vectors; "
                         f"{sp.shape} vs {initial_A.shape}")
    iAT = initial_A.T
    nfixed_a = 0
    if fixed_components_A is not None:
        nfixed_a = fixed_components_A
    if nfixed_a > ncomp:
        raise ValueError(
            f"The number of fixed components of A ({nfixed_a}) "
            f"cannot exceed the total number of components ({ncomp})")
    nfreeze = 0
    if fixed_features_A is not None:
        nfreeze = fixed_features_A
    if normalize == 'A':
        A = np.zeros(iAT.shape)
        A[:nfreeze] = iAT[:nfreeze]
        norm = np.linalg.norm(iAT[nfreeze:], axis=0)
        np.divide(iAT[nfreeze:], norm, where=norm!=0, out=A[nfreeze:])
    else:
        A = iAT.copy()

    B = np.empty((ncomp, nsamp)) if initial_B is None else initial_B.copy()
    if mutable_components_B is not None:
        fixed_B = np.ones(ncomp, dtype=bool)
        fixed_B[mutable_components_B] = False
    else:
        fixed_B is None

    newA = newB = None
    errors = []
    errorbest = None
    error = preverror = None

    cw = 0
    if contrast_weight is not None:
        if contrast_weight[0] == 'A':
            cw = contrast_weight[1]
        elif contrast_weight[0] == 'B':
            cw = -contrast_weight[1]
        else:
            raise ValueError("contrast_weight must be ('A'|'B', [0-1])")

    if acceleration == 'Anderson':
        ason_m = 2 # Memory depth
        ason_Bmode = False # Currently accelerating B steps?
        ason_g = None
        ason_G = []
        ason_X = []
    elif acceleration:
        raise ValueError("acceleration must be None or 'Anderson'")

    if starttime is None:
        starttime = time.process_time()
    times = []

    # Loop logic:
    # Half-iterations with ba=0 (update A) and ba=1 (update B).
    # Take a half-step (A or B):
    #   Update B from (prevA := newA (if set) or A) [or vice versa A<->B]
    #   Forget newA
    # If no acceleration: just keep iterating.
    # If currently accelerating the just-updated A/B:
    #   If error got worse:
    #       Reset acceleration memory
    #       redo the step from A/B (not newA/B)
    #       switch to acceleration mode A<->B [can be optional]
    #   else if not (very first iteration and accelerating B):
    #       First acc. step:
    #           Just save delta-A (X) [or B]
    #       Else:
    #           save delta-delta (G), then compute accelerated newA [or newB]
    # So: A is ok. newA is proposed accelerated A. Evaluated to make B which
    #   is rejected if error grows compared with A; then a normal step from
    #   A to B is guaranteed to be ok. Acceleration continues from that B.
    for it in range(maxiters):
        ba = 0
        retry = False
        while ba < 2:
            if not retry:# unnecessary?
                preverror = error
            if ba == 0:
                if newA is None:
                    newA = A
                prevA = newA
                if cw > 0:
                    newA = (1 - cw) * newA + cw * newA.mean(1)[:,None]
                if fixed_B is not None:
                    tmpsp = sp - (newA[:, fixed_B] @ B[fixed_B]).T
                else:
                    tmpsp = sp
                if nonnegative[1]:
                    if fixed_B is None:
                        tmpB = np.empty_like(B.T)
                        error = multi_nnls(newA, tmpsp, tmpB)
                        B = tmpB.T
                    else:
                        if not retry: # only needed when ason_Bmode==0 ?
                            B = np.copy(B)
                        tmpB = np.empty(shape=B[mutable_components_B].T.shape)
                        error = multi_nnls(newA[:, mutable_components_B],
                                           tmpsp, tmpB)
                        B[mutable_components_B] = tmpB.T
                else:
                    if not retry: # only needed when ason_Bmode==0 ?
                        B = np.copy(B)
                    B[mutable_components_B], res, _, _ = np.linalg.lstsq(
                        newA[:, mutable_components_B], tmpsp, rcond=-1)
                    error = res.sum()
                if normalize == 'B':
                    norm = np.linalg.norm(
                        B[mutable_components_B], axis=1, keepdims=True)
                    np.divide(B[mutable_components_B], norm, where=norm!=0,
                              out=B[mutable_components_B])

                newA = None
            else:
                if newB is None:
                    newB = B
                prevB = newB
                if cw < 0:
                    newB = (1 + cw) * newB - cw * newB.mean(1)[:,None]
                if modify_before_Astep is not None:
                    modify_before_Astep(A, newB, sp)

                if nfixed_a == ncomp:
                    # Single pass, nothing more to fit.
                    error = ((sp[:, nfreeze:] - (A[nfreeze:] @ newB).T
                             )**2).sum()
                    break
                elif nfixed_a > 0:
                    tmpsp = sp[:, nfreeze:] - (
                        A[nfreeze:, :nfixed_a] @ newB[:nfixed_a]).T
                else:
                    tmpsp = sp[:, nfreeze:]

                if nonnegative[0]:
                    if not retry: # only when ason_Bmode == 1 ?
                        A = A.copy()
                    error = multi_nnls(newB[nfixed_a:, :].T,
                                       tmpsp.T, A[nfreeze:, nfixed_a:])
                else:
                    tmpA, res, _, _ = np.linalg.lstsq(
                        newB[nfixed_a:, :].T, tmpsp, rcond=-1)
                    A[nfreeze:, nfixed_a:] = tmpA.T
                    error = res.sum()

                if normalize == 'A':
                    norm = np.linalg.norm(A[:, nfixed_a:], axis=0)
                    np.divide(A[:, nfixed_a:], norm, where=norm!=0,
                              out=A[:, nfixed_a:])
                newB = None

            if acceleration is None:
                pass
            elif ba == ason_Bmode:
                if retry:
                    retry = False
                    ason_Bmode = not ason_Bmode # Alternate A/B acceleration
                elif len(ason_X) > 1 and error > preverror:
                    ason_X = []
                    ason_G = []
                    retry = True
                    ba = ba - 1
                else:
                    pass
            elif ason_Bmode == 1 and it == 0:
                pass
            else:
                prevg = ason_g
                ason_g = ((A - prevA) if ba else (B - prevB)).flatten()
                if len(ason_X) < 1:
                    ason_X.append(ason_g)
                else:
                    ason_G.append(ason_g - prevg)
                    while(len(ason_G) > ason_m):
                        ason_G.pop(0)
                        ason_X.pop(0)
                    Garr = np.asarray(ason_G)
                    try:
                        gamma = np.linalg.lstsq(Garr.T, ason_g, rcond=None)[0]
                    except np.linalg.LinAlgError:
                    #     gamma = scipy.linalg.lstsq(Garr.T, ason_g)[0]
                    # except scipy.linalg.LinAlgError:
                        print('lstsq failed to converge; '
                              'restart at iter %d' % it)
                        ason_X = []
                        ason_G = []
                    else:
                        dx = ason_g - gamma @ (np.asarray(ason_X) + Garr)
                        ason_X.append(dx)
                        if ba:
                            newA = prevA + dx.reshape(A.shape)
                            if nonnegative[0]:
                                np.maximum(0, newA, out=newA)
                        else:
                            newB = prevB + dx.reshape(B.shape)
                            if nonnegative[1]:
                                np.maximum(0, newB, out=newB)
            ba = ba + 1

        curtime = time.process_time() - starttime
        times.append(curtime)
        errors.append(error)
        if not it or error < errorbest:
            errorbest = error
            Abest = A
            Bbest = B
            iterbest = it
            if nfixed_a == ncomp:
                break
        if it:
            if error < tol_abs_error:
                break
            if tol_rel_improv and it > tol_rel_iters:
                emax = max(errors[-tol_rel_iters-1:-2])
                if (emax - errors[-1]) <= \
                    tol_rel_improv * tol_rel_iters * emax:
                        break
            if tol_iters_after_best is not None:
                if iterbest + tol_iters_after_best < it:
                    break
            if maxtime and curtime >= maxtime:
                break
        if callback is not None:
            callback(it, errors, A.T, B)

    if nfixed_a and not np.isscalar(fixed_components_A):
        # Could remove the fixed components from A here, but it's not clear
        # if that's more convenient when anyway they're in B too.
        pass

    if return_time:
        return Abest.T, Bbest, errors, times
    return Abest.T, Bbest, errors


def mcr_als_freeze(sp, initial_A, *, maxiters=100,
                   tol_rel_improv=None, callback=None,
                   freeze_components_B, freeze_scheme,
                   add_residual_components=None, print_progress=True,
                   **kwargs):
    """
    Perform MCR-ALS nonnegative matrix decomposition on the matrix sp.
    The input array is decomposed as A@B where A and B are spectra and
    concentrations or vice versa.

    This version adds gradually applied constraints on the B matrix: The
    values (e.g. concentrations) are iteratively constrained to their mean
    values across all or a subset of the samples. This is useful for unmixing
    of sets of samples (e.g. time courses or from different individuals).
    Parameters are passed throught to mcr_als.

    Parameters
    ----------
    sp : array(nsamples, nfeatures)
        Spectra to be decomposed.
    initial_A : array(ncomponents, nfeatures)
        Initial spectra (or concentrations).
    maxiters : int, optional
        Maximum number of iterations.
    tol_rel_improv : float, optional
        Stop at average relative improvement of less than tol_rel_improv.
    callback : func(it : int, err : list(float), A : array, B : array), optional
        Progress callback for every iteration.
    freeze_components_B : int or [int] or {int: [range, ...]}
        Components that are to be made constant across samples by means of
        growing Lagrange multipliers.
        If int, take that many of the first components. The set/dict forms
        specify component indexes and optionally sample ranges. With the
        latter, an endmember can be frozen to different levels independently
        in different sets of samples.
    freeze_scheme : ndarray(iter) or ndarray(iter, index)
        Lagrange multiplier scheme for forcing the constant components from
        free to constant or constrained. If 2D, the index refers to the order
        of the components in freeze_components_B.
        There will always be a first unconstrained iteration. In the last
        iteration, a value of Inf is allowed and recommended for components
        that are to be made constant.
        Recommended starting point: [1, 3, 10, numpy.inf]
    add_residual_components : int, optional
        Add this many new components based on principal components of
        residuals after the initial MCR-ALS step.
        If you want to freeze a newly added residual component, it first
        needs one iteration with lambda=0 or it will be frozen to 0.
    print_progress : bool, optional
        Print some useful numbers for each iteration.

    Returns
    -------
    A : array(ncomponents, nfeatures)
        Spectra (at lowest error)
    B : array(ncomponents, nsamples)
        Concentrations at lowest error
    error : list(float)
        Mean square error at every iteration
    process_time : list(float)
        Time relative start at each iteration
    """

    starttime = time.process_time()
    nsamp, nfeat = sp.shape
    ncomp = initial_A.shape[0]

    if type(freeze_components_B) is dict:
        f_ranges = [(f, frng) for f, frng
                    in freeze_components_B.items()]
    else:
        if np.isscalar(freeze_components_B):
            freeze_components_B = list(range(freeze_components_B))
        f_ranges = [(f, [range(nsamp)]) for f in freeze_components_B]

    niter = len(freeze_scheme)
    iters = np.geomspace(maxiters ** .75, maxiters, num=niter + 1, dtype=int)
    if tol_rel_improv is None:
        tolrel = [None] * (niter + 1)
    else:
        tolrel = np.geomspace(tol_rel_improv ** .5, tol_rel_improv,
                              num=niter + 1)
    kwargs['return_time'] = True

    cbinfo = [0]
    def extended_cb(it, err, A, B):
        if callback is not None:
            callback(cbinfo[0], err, A, B)
        cbinfo[0] += 1

    def freeze_components(A, B, sp_out, *, lamb, franges):
        for fi, (f, rngs) in enumerate(franges):
            l = lamb if np.isscalar(lamb) else lamb[fi]
            for rng in rngs:
                sp_out[rng, fi] = B[f, rng].mean() * l

    if print_progress:
        print(f"Running initial MCR-ALS, size {sp.shape} / {initial_A.shape}")
    A, B, full_error, full_process_time = mcr_als(
        sp, initial_A, maxiters=iters[-1], tol_rel_improv=tolrel[-1],
        callback=extended_cb, starttime=starttime, **kwargs)
    if print_progress:
        print(f"  MCR-ALS in {full_process_time[-1]:.2f} s, "
              f"{len(full_error)} iters, err {full_error[-1]:.5g}")

    if len(freeze_scheme) == 0:
        return A, B, full_error, full_process_time

    if add_residual_components:
        res = sp - B.T @ A
        pca = pca_nipals(res, add_residual_components)
        A = np.vstack((A, pca * A.max() / pca.max(1, keepdims=True)))
        B = np.vstack((B, np.zeros((add_residual_components, nsamp))))
    # Components may have been added
    ncomp = len(A)

    nfreeze = len(f_ranges)
    fsp = np.hstack((np.zeros((nsamp, nfreeze)), sp.copy()))
    fA = np.hstack((np.zeros((ncomp, nfreeze)), A.copy()))

    for i, lamb in enumerate(freeze_scheme):
        # On the last pass, identify (fi, f) that are to be removed from
        # fsp, fA, f_ranges, f_comps, nfreeze and that will be marked
        # as non-mutable.
        if (i == len(freeze_scheme) - 1) and np.any(np.isinf(lamb)):
            nonfrozen = np.ones(ncomp, dtype=bool)
            frozenixs = []
            for fi, (f, rngs) in enumerate(f_ranges):
                l = lamb if np.isscalar(lamb) else lamb[fi]
                if np.isinf(l):
                    nonfrozen[f] = False
                    frozenixs.append(fi)
                    for rng in rngs:
                        B[f, rng] = B[f, rng].mean()
            fsp = np.delete(fsp, frozenixs, axis=1)
            fA = np.delete(fA, frozenixs, axis=1)
            f_ranges = [x for i, x in enumerate(f_ranges)
                        if i not in frozenixs]
            nfreeze = len(f_ranges)
            if not np.isscalar(lamb):
                lamb = np.delete(lamb, frozenixs)
        else:
            nonfrozen = slice(None)

        f_comps = [f for f, rngs in f_ranges]
        fA[f_comps, :nfreeze] = np.eye(nfreeze) * lamb
        freeze_components(None, B, fsp, lamb=lamb, franges=f_ranges)

        if print_progress:
            print(f"Running MCR-ALS, λ={lamb}, size {fsp.shape} / {fA.shape}")
        fA, B, error, process_time = mcr_als(
            fsp, fA, maxiters=iters[i], tol_rel_improv=tolrel[i],
            fixed_features_A=nfreeze,
            modify_before_Astep=partial(
                freeze_components, lamb=lamb, franges=f_ranges),
            mutable_components_B=nonfrozen,
            initial_B=B,
            callback=extended_cb, starttime=starttime, **kwargs)
        full_error.extend(error)
        full_process_time.extend(process_time)
        if print_progress:
            err = ((sp - B.T @ fA[:, nfreeze:])**2).sum()
            print(f"  time={process_time[-1]:.2f} s, "
                  f"{len(error)} iters, err {error[-1]:.5g} (tot {err:.5g})")

    A = fA[:, nfreeze:]
    return A, B, full_error, full_process_time

