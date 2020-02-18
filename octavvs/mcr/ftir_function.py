"""
Compilation of function for the ftir main
"""

import numpy as np
import scipy.io as sio
#from scipy.signal import savgol_filter, wiener
#import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
#from skimage import filters
#from sklearn.cluster import SpectralClustering, KMeans
#from scipy.sparse import spdiags, csc_matrix
#from scipy.sparse.linalg import spsolve
#from scipy import sparse
#from scipy.spatial import ConvexHull
#from scipy.interpolate import interp1d
#import scipy as sc




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
     sp = ss[:,1:]
     if (len(info)) > 1:
         (l,*_) = s['wh']
         w , h = l
     else:      
         w = int(np.sqrt(sp.shape[1]))
         h = sp.shape[1] // w
         if w * h != sp.shape[1]:
             w = sp.shape[1]
             h = 1
     p = sp.reshape(sizex,h,w,order='C')
     return  w, h, p, wavenumber, sp


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



def listindexes(l):
    r = {}
    for i in range(len(l)):
        v = l[i]
        if v in r:
            r[v].append(i)
        else:
            r[v] = [i]
    return r
