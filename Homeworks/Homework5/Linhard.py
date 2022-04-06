import numpy as np
from numba import jit 
from numba import int64, float64
from numba.experimental import jitclass

@jit(nopython=True)
def ferm(x):

    if x > 700:

        return 0

    else:

        return 1 / (np.exp(x) + 1)

@jit(nopython=True)
def get_linhard(x, Omega, q, res, kF, T, broad, nrm):

    for i in range(x.shape[0]):

        k = x[i, 0:3]
        e_k_q = np.linalg.norm(k-q)**2 - kF * kF
        e_k = np.linalg.norm(k)**2 - kF * kF
        dfermi = (ferm(e_k_q/T) - ferm(e_k/T))
        res[:, i] = -2 * nrm * dfermi / (Omega-e_k_q + e_k + broad * 1j)

    return res

spec = [
    ('Omega', float64),
    ('q', float64[:]),
    ('kF', float64),
    ('T', float64),
    ('broad', float64),
    ('nrm', float64),
]

@jitclass(spec)
class Linhard:

    def __init__(self, Omega, q, kF, T, broad):

        self.Omega = Omega
        self.q = np.array([0, 0, q])
        self.kF = kF
        self.T = T
        self.broad = broad
        self.nrm = 1 / (2 * np.pi)**3

    def GetLinhard(self, x):

        res = np.zeros((100, x.shape[0]), dtype = np.complex64)

        return get_linhard(x, self.Omega, self.q, res, self.kF, self.T, self.broad, self.nrm)