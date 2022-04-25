import numpy as np
from numba import jit
import matplotlib.pyplot as plt

@jit(nopython = True)
def sampled_distribution(x):

    return 1 / np.sqrt(2 * np.pi) * np.exp(-0.5 * x**2)

@jit(nopython = True)
def target_distribution(x):

    return np.cos(50 * x) + np.sin(20 * x)

@jit(nopython = True)
def metropolis_hastings(function, a, b, max_itr):

    # Initialize Point
    x = np.zeros(max_itr)
    x[0] = np.random.uniform(a, b, 1)[0]

    for i in range(1, max_itr):

        # Propose New Point
        propose = x[i - 1] + np.random.normal(0, 1, 1)[0]
        A = np.minimum(1, function(propose) / function(x[i - 1]))

        # Metropolis Hastings 
        if A > np.random.uniform(0, 1, 1)[0]:
            x[i] = propose
        
        else: 
            x[i] = x[i - 1]

    return x


if __name__ == "__main__":

    def main():

        a = 0
        b = 1
        max_itertion = 100000

        x = metropolis_hastings(target_distribution, a, b, max_itertion)

        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.suptitle("Metropolis Hastings")

        interval = np.linspace(a, b, 1000)
        fx = target_distribution(interval)

        ax1.plot(interval, fx)
        ax1.set_title("Target Distribution")

        ax2.hist(x, bins = 100)
        ax2.set_title("Proposal Distribution")

        plt.show()

    main()
