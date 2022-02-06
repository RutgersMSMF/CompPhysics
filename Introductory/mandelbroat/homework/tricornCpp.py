from scipy import *
from pylab import *
import time
from timeit import default_timer as timer
from numba import jit

import tricorn

def TricornPython(ext, max_steps, Nx, Ny):

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

                z = np.conjugate(z * z) + z0

    return data

@jit(nopython=True)
def TricornNumba(ext, max_steps, Nx, Ny):

    data = ones((Nx, Ny)) * max_steps

    for i in range(Nx):

        for j in range(Ny):

            x = ext[0] + (ext[1] - ext[0]) * i / (Nx - 1.0)
            y = ext[2] + (ext[3] - ext[2]) * j / (Ny - 1.0)
            z0 = complex(x, y)
            z = 0j

            for itr in range(max_steps):

                if z.real * z.real + z.imag * z.imag > 4.0:
                    
                    data[j, i] = itr
                    break

                z = np.conjugate(z * z) + z0

    return data

def TricornPybind11(ext, max_steps, Nx, Ny):
    data = ones((Ny, Nx));
    tricorn.tri(data, Nx, Ny, max_steps, ext)
    return data

Nx = 1000
Ny = 1000
max_steps = 1000

ext = [-3/2, 1, -3/2, 3/2]

t0 = time.time()
t0_ = time.process_time()
data = TricornPybind11(ext, max_steps, Nx, Ny)
t1 = time.time()
t1_ = time.process_time()
print('pybind11: walltime: ', t1 - t0, 'cputime: ', t1_ - t0_)
imshow(data, extent=ext)
show()

t0 = time.time()
t0_ = time.process_time()
data = TricornNumba(array(ext), max_steps, Nx, Ny)
t1 = time.time()
t1_ = time.process_time()
print('numba: walltime: ', t1 - t0, 'cputime: ', t1_ - t0_)
imshow(data, extent=ext)
show()