import numpy as np
from numba import jit
import matplotlib.pyplot as plt
import VegasLectureObj as VLO
import Linhard as Lin

@jit(nopython = True)
def smfun(x):

    if (np.real(x) > 0):

        return ((x - 1) / np.log(x))**(1.5)

    else:

        return 0

@jit(nopython = True)
def Smoothen(fxbin):

    (ndim, nbins) = np.shape(fxbin)
    final = np.zeros(np.shape(fxbin), dtype = np.complex64)

    for idim in range(ndim):

        fxb = np.copy(fxbin[idim, :])
        fxb[:nbins - 1] += fxbin[idim, 1:nbins]
        fxb[1 : nbins]  += fxbin[idim, :nbins-1]
        fxb[1 : nbins-1] *= 1 / 3
        fxb[0] *= 1 / 2
        fxb[nbins - 1] *= 1 / 2
        norm = sum(fxb)

        fxb *= 1.0 / norm 
        final[idim, :] = fxb

    return final

@jit(nopython = True)
def SetFxbin(fxbin, bins, wfun):

    (n, ndim) = bins.shape

    for dim in range(ndim):

        for i in range(n):

            fxbin[dim, bins[i, dim]] = fxbin[dim, bins[i, dim]] + np.abs(wfun[i])        

@jit(nopython = True)
def integrant(x):

    return 1 / (1 - np.cos(x[:, 0]) * np.cos(x[:, 1]) * np.cos(x[:, 2])) / np.pi**3

def Vegas(integrant, a, b, maxeval, nstart, nincrease, grid, cum):

    ndim = grid.ndim  
    nbins = grid.nbins
    unit_dim = (b - a)**ndim  
    nbatch = 1000             
    neval = 0

    print ("""Vegas parameters:
    ndim = """+str(ndim)+"""
    limits = """+str(a)+" "+str(b)+"""
    maxeval = """+str(maxeval)+"""
    nstart = """+str(nstart)+"""
    nincrease = """+str(nincrease)+"""
    nbins = """+str(nbins)+"""
    nbaths = """+str(nbatch)+"\n")

    bins = np.zeros((nbatch, ndim)) 
    
    all_nsamples = nstart
    for iter in range(1000): 

        wgh = np.zeros(nbatch)           
        fxbin = np.zeros((ndim, nbins), dtype = np.complex64)  

        for nsamples in range(all_nsamples, 0, -nbatch): 
            
            n = min(nbatch, nsamples)
            xr = np.random.random((n, ndim)) 
            pos = xr * nbins                  
            bins = np.array(pos, dtype = int)     
            wgh = np.ones(nbatch) / all_nsamples
            new_xr = np.array(xr, dtype = np.complex64)

            for dim in range(ndim):   

                gi = grid.g[dim, bins[:, dim]]            
                gm = grid.g[dim, bins[:, dim] -1]         
                diff = gi - gm                       
                gx = gm + (pos[:, dim] - bins[:, dim]) * diff 
                new_xr[:, dim] = gx * (b - a) + a                   
                wgh = wgh * (diff * nbins)                
            
            fx = integrant.GetLinhard(new_xr)  
            neval += n 
            
            wfun = wgh * fx                   
            cum.sum += sum(wfun)    
            cum.sqsum += sum(wfun).real
            wfun2 = np.abs(wfun * np.conj(wfun)) * all_nsamples    
            SetFxbin(fxbin, bins, wfun2[0,:])
        
        w1 = cum.sqsum - np.abs(cum.sum)**2
        w = (all_nsamples - 1) / w1          
        cum.weightsum += w         
        cum.avgsum += w * cum.sum   
        cum.avg2sum += w * cum.sum**2  
        
        cum.avg = cum.avgsum / cum.weightsum    
        cum.err = np.sqrt(1 / cum.weightsum)       
     
        if iter > 0:
            cum.chisq = (cum.avg2sum - 2 * cum.avgsum * cum.avg + cum.weightsum * cum.avg**2) / iter
    
        print ("Iteration {:3d}: I= {:10.8f} +- {:10.8f}  chisq= {:10.8f} number of evaluations = {:7d} ".format(iter+1, cum.avg*unit_dim, cum.err*unit_dim, cum.chisq, neval))
        imp = Smoothen(fxbin)
        grid.RefineGrid(imp)
        
        cum.sum = 0                   
        cum.sqsum = 0 
        all_nsamples += nincrease  

        if (neval >= maxeval): 
            break
        
    cum.avg *= unit_dim
    cum.err *= unit_dim

def plot():

    # Linhard Parameters
    rs = 2
    kF = (9 * np.pi / 4)**(1 / 3) / rs
    T = 0.02 * kF**2
    broad = 0.002 * kF**2
    q = 0.1 * kF
    Omega = np.linspace(0, 0.5 * kF**2, 100)

    L = Lin.Linhard(Omega, q, kF, T, broad)
    linhard_exact = kF / 2 * np.pi**2

    # Integration Bounds
    lower = -3 * kF
    upper = 3 * kF

    # Vegas Parameters
    ndim = 3
    maxeval = 10000000
    nbins = 128
    nstart = 200000
    nincrease = 100000

    # Initialize Objects
    cum = VLO.Cumulants()
    grid = VLO.Grid(ndim,nbins)
    np.random.seed(0)

    # Run Vegas Integration
    Vegas(L, lower, upper, maxeval, nstart, nincrease, grid, cum)
    print(cum.avg, '+-', cum.err, 'exact=', linhard_exact, 'real error=', abs(cum.avg-linhard_exact) / linhard_exact)

    fig, (ax1) = plt.subplots(1, 1)
    fig.suptitle("Monte Carlo Integration")

    ax1.plot(grid.g[0, :nbins])
    ax1.plot(grid.g[1, :nbins])
    ax1.plot(grid.g[2, :nbins])

    plt.show()

plot()