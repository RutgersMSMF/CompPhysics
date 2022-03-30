import numpy as np 
from numba import jit
import matplotlib.pyplot as plt

N = 10000

@jit(nopython = True)
def f(x):

    return (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * x**2)

@jit(nopython = True)
def compute_area(rv, a, b):

    sum = 0
    E = np.zeros(len(rv))

    for i in range(len(rv)):
        sum += f(rv[i])
        E[i] = (b - a) * sum / (i + 1)

    return (sum / N, E)

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
    area, error = compute_area(rv, a, b)
    integral = (b - a) * area
    print("Monte Carlo Integral: ", integral)

    fig, (ax1) = plt.subplots(1, 1)
    fig.suptitle("Monte Carlo Integration")

    ax1.plot(error)
    ax1.set_title("Error Analysis")

    plt.show()

plot()

