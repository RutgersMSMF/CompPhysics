import numpy as np
from numba import jit
from numba import float64
from numba.experimental import jitclass

spec = [
    ('sum', float64),
    ('sum_squared', float64),
    ('integral', float64),             
    ('integral_squared', float64),  
    ('best_integral', float64),
    ('best_integral_squared', float64),    
    ('variance', float64),
    ('weight', float64),
    ('weighted_sum', float64) ,
    ('chi_squared', float64)  
]

N = 100000

@jitclass(spec)
class Variables:

  def __init__(self):

    self.sum = 0
    self.sum_squared = 0
    self.integral = 0
    self.integral_squared = 0
    self.best_integral = 0
    self.best_integral_squared = 0
    self.variance = 0
    self.weight = 0
    self.weighted_sum = 0
    self.chi_squared = 0

@jit(nopython = True)
def get_variance(Var):

    return (Var.integral_squared - Var.integral**2) / N

@jit(nopython = True)
def get_weight(Var):

    w0 = np.sqrt(Var.integral_squared * N) 
    w1 = (w0 + Var.integral) * (w0 - Var.integral) 
    w = (N - 1) / w1  

    return w

@jit(nopython = True)
def get_weighted_sum(Var):

    w0 = np.sqrt(Var.integral_squared * N) 
    w1 = (w0 + Var.integral) * (w0 - Var.integral) 
    w = (N - 1) / w1      
    Var.weighted_sum += w

    return Var.weighted_sum

@jit(nopython = True)
def get_best_integral(Var):

    best =  (Var.weight * Var.integral) / (Var.weighted_sum)

    return best

@jit(nopython = True)
def get_best_integral_squared(Var):

    best = (Var.weight * Var.integral_squared) / (Var.weighted_sum)

    return best

@jit(nopython = True)
def get_chi_squared(Var, index):

    chi = (Var.weight * Var.integral_squared) - 2 * (Var.weight * Var.integral * Var.best_integral) + (Var.weight * Var.best_integral**2) / index

    return chi