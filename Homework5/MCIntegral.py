import numpy as np 
from numba import jit
import matplotlib.pyplot as plt

N = 10000

@jit(nopython = True)
def f(x):

    return (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * x**2)

@jit(nopython = True)
def mc_integral(rv, a, b):

    sum = 0
    E = np.zeros(len(rv))

    for i in range(len(rv)):
        sum += f(rv[i]) 
        E[i] = (b - a) * sum / (i + 1)

    return (sum / N, E)

@jit(nopython = True)
def adaptive_mc_integral(a, b):

    n = 100
    err_tol = 1e-8

    sum = 0
    rv = np.random.uniform(a, b, n)

    for i in range(n):
        sum += f(rv[i])
    
    original = (b - a) * sum / (n)

    c = (a + b) / 2

    lower_sum = 0
    upper_sum = 0
    rvl = np.random.uniform(a, c, n)
    rvu = np.random.uniform(c, b, n)

    for i in range(n):
        lower_sum += f(rvl[i])
        upper_sum += f(rvu[i])
        
    composite = (b - a) * (lower_sum + upper_sum) / (2 * n)

    if (np.abs(original - composite) > err_tol):
        lower = adaptive_mc_integral(a, c)
        upper = adaptive_mc_integral(b, c)
        composite = lower + upper

    return composite

def plot():

    a = -6
    b = 6 

    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle("Monte Carlo Integration")

    # Plot Function 

    x = np.linspace(a, b)
    fun = f(x)

    ax1.plot(fun)
    ax1.set_title("Gaussian Distribution")

    # Generate Random Points
    rv = np.random.uniform(a, b, N)

    ax2.scatter(rv, np.linspace(0, 1, N))
    ax2.set_title("Uniform Random Variables")

    # Compute Integral
    area, error = mc_integral(rv, a, b)
    integral = (b - a) * area
    print("Monte Carlo Integral: ", integral)

    fig, (ax1) = plt.subplots(1, 1)
    fig.suptitle("Monte Carlo Integration")

    ax1.plot(error)
    ax1.set_title("Error Analysis")

    adaptive_integral = adaptive_mc_integral(-6, 6)
    print("Adaptive Monte Carlo Integral: ", adaptive_integral)

    plt.show()

plot()

