import numpy as np
from numba import jit 
from numba import int64, float64
from numba.experimental import jitclass

@jit(nopython = True)
def ferm(x):

    if x > 700:

        return 0

    else:

        return 1 / (np.exp(x) + 1)

@jit(nopython = True)
def get_linhard(momentum, Omega, kF, T, broad, nrm):

    q = momentum[0, :] 
    k = momentum[1, :]
    e_k_q = np.linalg.norm(k - q)**2 - kF**2
    e_k = np.linalg.norm(k)**2 - kF**2
    dfermi = ferm(e_k_q / T) - ferm(e_k / T)

    return -2 * nrm * dfermi / (Omega -e_k_q + e_k + broad * 1j)

spec = [
    ('Omega', float64[:]),
    ('kF', float64),
    ('T', float64),
    ('broad', float64),
    ('nrm', float64),
    ('Ndim', int64)
]

@jitclass(spec)
class Linhard:

    def __init__(self, Omega, kF, T, broad):

        self.Omega = Omega
        self.kF = kF
        self.T = T
        self.broad = broad
        self.nrm = 1 / (2 * np.pi)**3
        self.Ndim = 3

    def GetLinhard(self, x):

        return get_linhard(x, self.Omega, self.kF, self.T, self.broad, self.nrm)