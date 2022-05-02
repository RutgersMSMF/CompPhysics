from scipy import *
from numpy import *
from pylab import *
import time
from numba import jit 

@jit(nopython=True) 
def NumbaTricorn(ext, max_steps, Nx, Ny):

    # Similar, Non Analytic Mapping
    # The Tricorn Fractal
    
    data = ones((Nx, Ny)) * max_steps

    for i in range(Nx):

        for j in range(Ny):

            x = ext[0] + (ext[1] - ext[0]) * i / (Nx - 1.0)
            y = ext[2] + (ext[3] - ext[2]) * j / (Ny - 1.0)
            z0 = x + y * 1j
            z = 0j

            for itr in range(max_steps):

                if abs(z) > 2.0:
                    data[j, i] = itr
                    break

                # Complex Conjugate
                z = np.conjugate(z * z) + z0

    return data

def ax_update(ax): 
     
    ax.set_autoscale_on(False) 
    
    xstart, ystart, xdelta, ydelta = ax.viewLim.bounds
    xend = xstart + xdelta
    yend = ystart + ydelta
    ext=array([xstart,xend,ystart,yend])
    data = NumbaTricorn(ext, max_steps, Nx, Ny) 
    

    im = ax.images[-1]  
    im.set_data(data)  
    im.set_extent(ext)          
    ax.figure.canvas.draw_idle()

Nx = 1000
Ny = 1000
max_steps = 1000 

ext = [-3/2, 1, -3/2, 3/2]
    
t0 = time.time()    
data = NumbaTricorn(array(ext), max_steps, Nx, Ny)
t1 = time.time()
print('Python: ', t1 - t0)

fig, ax = subplots(1, 1)
ax.imshow(data, extent = ext, aspect = 'equal', origin = 'lower')
ax.set_title("Tricorn Set")
    
ax.callbacks.connect('xlim_changed', ax_update)
ax.callbacks.connect('ylim_changed', ax_update)
    
show()
