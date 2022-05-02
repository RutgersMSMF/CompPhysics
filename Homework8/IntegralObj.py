from numba import int64, float64
from numba.experimental import jitclass

spec = [
    ('kF', float64),
    ('cutoff', float64),
    ('dkF', float64),
    ('Nitt', int64),
    ('Ncount', int64),
    ('Nwarm', int64),
    ('Nsteps', int64),
    ('tmeasure', int64),
    ('Nbins', int64),
    ('V0norm', float64),
    ('dexp', int64),
    ('recomputew', int64),
    ('per_recompute', int64)
]

@jitclass(spec)
class params:
    
    def __init__(self):
        self.kF = 1.             # typical momentum
        self.cutoff = 3*self.kF  # integration cutoff
        self.dkF = 0.1*self.kF   # the size of a step
        
        self.Nitt = 2000000   # total number of MC steps
        self.Ncount = 50000    # how often to print
        self.Nwarm = 1000# warmup steps
        self.tmeasure = 10   # how often to measure
        
        self.Nbins = 129       # how many bins for saving the histogram
        self.V0norm = 2e-2    # starting V0
        self.dexp = 6         # parameter for fm at the first iteration, we will use 
        
        self.recomputew = 2e4 / self.tmeasure # how often to check if V0 is correct
        self.per_recompute = 7 # how often to recompute fm auxiliary measuring function






