import numpy as np 
from numba import jit
import matplotlib.pyplot as plt
import VegasIntegralObj as VO

N = 100000

@jit(nopython = True)
def f(x):

    return 1 / (1 - np.cos(x[:, 0]) * np.cos(x[:, 1]) * np.cos(x[:, 2])) / np.pi**3

@jit(nopython = True)
def refine_grid(a, b, index):
    """
    Adjust Grid According to Dominant Contributions Using Stratified Sampling
    """

    grid = np.zeros(2 * index + 1)
    delta = (b - a) / (2 * index)

    for i in range(len(grid)):
        grid[i] = a + i * delta

    return grid

@jit(nopython = True)
def vegas_integral(Var, rv, a, b):
    """
    1. Perform Monte Carlo Integration
    2. Update Object Class
    """

    s = 0
    s_squared = 0
    s += np.sum(f(rv))
    s_squared += np.sum(f(rv)**2)

    Var.sum = s
    Var.sum_squared = s_squared
    Var.integral += (b - a) * s / N
    Var.integral_squared += (b - a) * s_squared / N

    return 0

@jit(nopython = True)
def vegas(Var):
    """
    Implementation of Vegas Integration Routine.
    Adaptive Numerical Integration For Higher Order Functions.
    Increase Number of Points Where Integrand is Larger, Resulting in Reduction of Error.
    """

    # Function Bounds
    a = -np.pi
    b = np.pi

    # Parameters
    n_dim = 3
    max_eval = 20

    for i in range(max_eval):

        # Adaptive Bounds
        new_grid = refine_grid(a, b, i + 1)

        Var.integral = 0
        Var.integral_squared = 0

        # Iterate Thru New Grid
        for j in range(len(new_grid) - 1):

            rv = np.random.uniform(new_grid[j], new_grid[j + 1], (N, n_dim))
            vegas_integral(Var, rv, new_grid[j], new_grid[j + 1])

        Var.weight = VO.get_weight(Var)
        Var.weighted_sum = VO.get_weighted_sum(Var)
        Var.best_integral = VO.get_best_integral(Var)
        Var.best_integral_squared = VO.get_best_integral_squared(Var)
        Var.variance = VO.get_variance(Var)
        Var.chi_squared = VO.get_chi_squared(Var, i + 1)

        print("Iteration: ", i, "Area: ", Var.integral)

    return 0

def plot():

    Var = VO.Variables()
    true_value = 1.3932 * 2

    vegas(Var)
    print("True Integral: ", true_value)
    print("Vegas Integral: ", Var.integral)
    print("Chi Squared Value: ", Var.chi_squared)
    print("Vegas Integral Error: ", np.abs(Var.integral - true_value))

    plt.show()

plot()