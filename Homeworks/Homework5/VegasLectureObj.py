import numpy as np
from numba import int64, float64, complex64
from numba.experimental import jitclass

spec = [
    ('sum', complex64),
    ('sqsum', float64),
    ('avg', complex64),
    ('err', complex64),
    ('chisq', complex64),
    ('weightsum', complex64),
    ('avgsum', complex64),
    ('avg2sum', complex64)
]

@jitclass(spec)
class Cumulants:

    def __init__(self):

        self.sum = 0.0    
        self.sqsum = 0.0 
        self.avg = 0.0  
        self.err = 0.0  
        self.chisq = 0.0
        self.weightsum = 0.0 
        self.avgsum = 0.0    
        self.avg2sum = 0.0   

spec2 = [
    ('g', complex64[:, :]),
    ('ndim', int64),
    ('nbins', int64),

]

@jitclass(spec2)
class Grid:

    def __init__(self, ndim, nbins):

        self.g = np.zeros((ndim, nbins + 1), dtype = np.complex64)  
        self.ndim = ndim
        self.nbins = nbins

        for idim in range(ndim):
            self.g[idim,:nbins] = np.arange(1, nbins + 1) / float(nbins)
            
    def RefineGrid(self, imp):

        (ndim, nbins) = np.shape(imp)
        gnew = np.zeros((ndim, nbins + 1), dtype = np.complex64)

        for idim in range(ndim):

            avgperbin = sum(imp[idim, :]) / nbins
            newgrid = np.zeros(nbins, dtype = np.complex64)
            cur = 0.0
            thisbin = 0.0
            ibin = -1

            for newbin in range(nbins - 1):

                while (np.real(thisbin) < np.real(avgperbin)):

                    ibin+=1
                    thisbin += imp[idim, ibin]
                    prev = cur
                    cur = self.g[idim, ibin]
    
                thisbin -= avgperbin   
                delta = (cur - prev) * thisbin 
                newgrid[newbin] = cur - delta / imp[idim, ibin]  

            newgrid[nbins - 1] = 1.0
            gnew[idim, :nbins] = newgrid

        self.g = gnew

        return gnew