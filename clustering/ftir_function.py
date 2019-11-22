"""
Compilation of function for the ftir main
"""

import numpy as np
import scipy.io as sio
from scipy.signal import savgol_filter, wiener
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import cv2
from skimage import filters
from sklearn.cluster import SpectralClustering, KMeans
from scipy.sparse import spdiags, csc_matrix
from scipy.sparse.linalg import spsolve
from scipy import sparse
from scipy.spatial import ConvexHull
from scipy.interpolate import interp1d
import scipy as sc




"""
#######################read data from matlab input file#######################
"""
def readmat(filename):
     s = sio.loadmat(filename)
     info = sio.whosmat(filename)
     ss = s[(str(info[0][0]))]
     wavenumber=ss[:,0]
     sizex = len(wavenumber)
     sizemx, sizemy = np.shape(ss)
     sp = np.zeros((sizemx, sizemy-1))
     sp = ss[:,1:sizemy]
     res = int(np.sqrt(sizemy-1))
     p = sp.reshape(sizex,res,res,order='C')       
     return res,p,wavenumber, sp


#(1) Based on area under spectrum
def proarea(d3image,x):
    i,j,k = np.shape(d3image)
    x = np.sort(x) #because data are scanned from high to low
    cc=np.zeros((j,k))
    for ii in range(0,j):
         for jj in range(0,k):
              cc[ii,jj]= np.trapz(d3image[:,ii,jj],x)
    return cc

#area of single spectra
def area(y):
    area = np.trapz(y)
    return area

#(2) Based on the wavenumber (cm-1)
def prowavenum(d3image,wavenums,wavenumv):
    i,j,k = np.shape(d3image)
    cc=np.zeros((j,k))
    idx = (np.abs(wavenums-wavenumv)).argmin()
    cc = d3image[idx,:,:]
    return cc

#(3) Based on Maximum Intensity Projection 
def promip(d3image):
    i,j,k = np.shape(d3image)
    brp = d3image[:,:,:]
    cmode1 = brp.max(0) #max
    return cmode1

#(4) Based on local area under curve
#prerequisite Gaussian Function
def proloa(d3image,wavenum,wavenumv):
    i,j,k = np.shape(d3image)
    cc=np.zeros((j,k))
#    uu=np.zeros((j,k))
    x = wavenum[::-1]
    i1 = (np.abs(x-(wavenumv-100))).argmin()
    i2 = (np.abs(x-(wavenumv+100))).argmin()  
    init_vals = [1, 2000, 500]  # for [amp, cen, wid]
    for ii in range(0,j):
        for jj in range(0,k):
#            uu= np.trapz(d3image[:,ii,jj],x)
            spectra = d3image[:,ii,jj]
            spectra = spectra[::-1]
            yplot = spectra[i1:i2+1]
            xplot = x[i1:i2+1]
            best_vals, covar = curve_fit(gaussian, xplot, yplot, p0=init_vals)
            cc[ii,jj]= np.trapz(gaussian(xplot, *best_vals),xplot)#/uu
 #           cc[ii,jj]= np.trapz(yplot,xplot)
    return cc
"""
###############################################################################
"""
"""
Function to evaluat the initial value using SIMPLISMA    
"""
def initi_simplisma(d,nr,f):
    """
    Function to calculte the pure profiles
    Reference Matlab Code:
         J. Jaumot, R. Gargallo, A. de Juan, R. Tauler,
         Chemometrics and Intelligent Laboratoty Systems, 76 (2005) 101-110
    
   
    ---:
        input : float d(nrow,ncol) = original spectrum
                integer nr = the number of pure components
                float f = number of noise allowed eg. 0.1 (10%)
        
        output: float spout(nr,nrow) = purest number component profile
                integer imp(nr) = indexes of purest spectra                   
    """  
    nrow = d.shape[0]
    ncol = d.shape[1]
    
    s = d.std(axis=0)
    m = d.mean(axis=0)
    mf = m + m.max() * f
    p = s / mf

    # First Pure Spectral/Concentration profile
    imp = np.empty(nr, dtype=np.int) 
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
        dm[1:i+1,1:i+1] = c[imp[:i],:][:,imp[:i]]
        for j in range(ncol):
            dm[0,0] = c[j,j]
            dm[0,1:i+1]=c[j,imp[:i]]
            dm[1:i+1,0]=c[imp[:i],j]
            w[j] = np.linalg.det(dm[0:i+1, 0:i+1])

        imp[i] = (p * w).argmax()
        
    ss = d[:,imp]
    spout = ss / np.sqrt(np.sum(ss**2, axis=0))
    return spout.T, imp


"""
Created on Mon Feb  4 11:48:28 2019

@author: Carl Troein
"""
def straight(x, y):
    """
    Return a straight line baseline correction.
    x: wavenumbers, sorted either way
    y: spectrum or spectra at those wavenumbers; shape (..., wavenumber)
    Returns: baseline of the spectrum, measured at the same points
    """
    # Create baseline using linear interpolation between vertices
    if x[0] < x[-1]:
        return interp1d(x[[0,-1]], y[...,[0,-1]], assume_sorted=True)(x)
    return interp1d(x[[-1,0]], y[...,[-1,0]], assume_sorted=True)(x)


def asls(y, lam, p, niter=20):
    """
    Return the baseline computed by Asymmetric least squares background correction, AsLS.
    Ref: Baseline correction with asymmetric least squares smoothing. PHC Eilers & HFM Boelens.
        Leiden University Medical Centre Report, 2005
    y: one spectrum to correct, or multiple as an array of shape (spectrum, wavenumber)
    lam: lambda, the smoothness parameter
    p: p, the asymmetry parameter, typically .001 to 0.1
    niter: maximum number of iterations
    """
    L = y.shape[-1]
    D = sparse.csc_matrix(np.diff(np.eye(L), 2))
    D = lam * D.dot(D.T)
    multi = y.ndim > 1
    y = y.copy() if multi else [ y ]
    for yy in y:
        w = np.ones(L)
        for i in range(niter):
            Z = sparse.spdiags(w, 0, L, L) + D
            z = spsolve(Z, w*yy)
            wnew = p * (yy > z) + (1-p) * (yy < z)
            if np.array_equal(wnew, w):
                break
            w = wnew
        if not multi:
            return z
        yy[:] = z
    return y

def iasls(y, lam, lam1, p, niter=30):
    """
    Return the baseline computed by Improved asymmetric least squares background correction, IAsLS.
    Ref: Baseline correction for Raman spectra using an improved asymmetric least squares method.
        Shixuan He, Wei Zhang, Lijuan Liu, Yu Huang, Jiming He, Wanyi Xie, Peng Wu and Chunlei Du.
        Anal. Methods, 2014, 6, 4402-4407. DOI: 10.1039/C4AY00068D
    In this implementation, W is not squared so p carries the same meaning as in AsLS.
    y: one spectrum to correct, or multiple as an array of shape (spectrum, wavenumber)
    lam: lambda, the smoothness parameter
    lam1: lambda1, the 1st derivatives smoothness parameter
    p: p, the asymmetry parameter
    niter: maximum number of iterations
    """
    L = y.shape[-1]
    D = sparse.csc_matrix(np.diff(np.eye(L), 2))
    D = lam * D.dot(D.T)
    D1 = sparse.csc_matrix(np.diff(np.eye(L), 1))
    D1 = lam1 * D1.dot(D1.T)
    multi = y.ndim > 1
    y = y.copy() if multi else [ y ]
    for yy in y:
        w = np.ones(L)
        W = sparse.spdiags(w, 0, L, L)
#        W = W @ W.T
        z = spsolve(W + D, w*yy)
        w = p * (yy > z) + (1-p) * (yy < z)
        for i in range(niter):
            W = sparse.spdiags(w, 0, L, L)
#            W = W @ W.T
            z = spsolve(W + D + D1, (W + D1)*yy)
            wnew = p * (yy > z) + (1-p) * (yy < z)
            if np.array_equal(wnew, w):
                break
            w = wnew
        if not multi:
            return z
        yy[:] = z
    return y

def arpls(y, lam, ratio=1e-6, niter=50):
    """
    Return the baseline computed by asymmetric reweighted penalized least squares smoothing, arPLS.
    Ref: Baseline correction using asymmetrically reweighted penalized least squares smoothing
        Sung-June Baek, Aaron Park, Young-Jin Ahn and Jaebum Choo
        Analyst, 2015, 140, 250-257. DOI: 10.1039/C4AN01061B
    In this implementation, W is not squared so p carries the same meaning as in AsLS.
    y: one spectrum to correct, or multiple as an array of shape (spectrum, wavenumber)
    lam: lambda, the smoothness parameter
    lam1: lambda1, the 1st derivatives smoothness parameter
    p: p, the asymmetry parameter
    niter: maximum number of iterations
    """
    L = y.shape[-1]
    D = sparse.csc_matrix(np.diff(np.eye(L), 2))
    D = lam * D.dot(D.T)
    multi = y.ndim > 1
    y = y.copy() if multi else [ y ]
    for yy in y:
        w = np.ones(L)
        while(True):
            W = sparse.spdiags(w, 0, L, L)
            z = sparse.linalg.spsolve(W + D, w * yy)
            d = yy - z
            dn = d[d < 0]
            s = dn.std()
            wt = 1. / (1 + np.exp(2 / s * (d - (2*s-dn.mean()))))
            if np.linalg.norm(w - wt) / np.linalg.norm(w) < ratio:
                break
            w = wt
        if not multi:
            return z
        yy[:] = z
    return y


def rubberband(x, y):
    """
    Rubberband baseline correction of one or more spectra.
    x: wavenumbers, sorted in either direction
    y: spectrum at those wavenumbers, or multiple spectra as array of shape (spectrum, wavenumber)
    :return: baseline of the spectrum, measured at the same points
    """
    if x[0] > x[-1]:
        return rubberband(x[::-1], y[...,::-1])[...,::-1]

    multi = y.ndim > 1
    y = y.copy() if multi else [ y ]
    for yy in y:
        # Find the convex hull
        v = ConvexHull(np.column_stack((x, yy))).vertices
        # Rotate convex hull vertices until they start from the lowest one
        v = np.roll(v, -v.argmin())
        # Leave only the ascending part
        v = v[:v.argmax()+1]
        # Create baseline using linear interpolation between vertices
        b = np.interp(x, x[v], yy[v])
        if not multi:
            return b
        yy[:] = b
    return y

def concaverubberband(x, y, iters):
    """
    Concave rubberband baseline correction. This algorithm removes more than a straight line, alternating with
    normal rubberband to bring negative points up again. It does not converge nicely and will eat up all the data
    if run with many iterations.
    Parameters:
    x: wavenumbers, sorted from low to high (todo: implement high-to-low)
    y: spectrum at those wavenumbers
    iters: iterations to run; note that this algorithm doesn't converge nicely
    Returns: baseline of the spectrum, measured at the same points
    """
    origy = y
    multi = y.ndim > 1
    y = y.copy() if multi else [ y.copy() ]
    for yy in y:
        yy -= rubberband(x, yy);
        for i in range(iters):
            F = .1 * (yy.max() - yy.min())
            xmid = .5 * (x[-1] + x[0])
            d2 = .25 * (x[-1] - x[0]) ** 2
            yy += F * (x - xmid)**2 / d2
            yy -= rubberband(x, yy);
    if not multi:
        return origy - y[0]
    return origy - y




def cut_spectrum(wn, y, ranges):
    """
    Cut a set of spectra, leaving only the given wavenumber range(s).
    Parameters:
    wn: array of wavenumbers, sorted in either direction
    y: array of spectra, shape (..., wavenumber)
    ranges: list or numpy array of shape (..., 2) with desired wavenumber ranges in pairs (low, high)
    Returns: (wavenumbers, spectra) with data in the given wavenumber ranges
    """
    if isinstance(ranges, list):
        ranges = np.array(ranges)
    inrange = lambda w: ((w >= ranges[...,0]) & (w <= ranges[...,1])).any()
    ix = np.array([inrange(w) for w in wn])
    return wn[ix], y[...,ix]


def listindexes(l):
    r = {}
    for i in range(len(l)):
        v = l[i]
        if v in r:
            r[v].append(i)
        else:
            r[v] = [i]
    return r