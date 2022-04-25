import numpy as np
from numba import jit
import matplotlib.pyplot as plt

@jit(nopython = True)
def rigid_function(x):

    return np.cos(50 * x) + np.sin(20 * x)

@jit(nopython = True)
def simualted_annealing(function, a, b, max_itr):

    # Initialize Starting Point
    s = np.random.uniform(a, b, 1)[0]
    temperature = np.zeros(max_itr)

    for i in range(max_itr):

        # Decrement "Temperature"
        T = 1 - (i / max_itr)
        temperature[i] = T

        # Pick Random Neighbor
        s_new = np.random.uniform(a, b, 1)[0]

        # Evaluate Function
        fs = function(s)
        fs_new = function(s_new)

        diff = fs - fs_new 

        if diff > 0:

            s = s_new

        else:

            # Metropolis Hastings
            if np.exp(-(fs_new - fs) / T) > np.random.uniform(0, 1, 1)[0]:

                # Update Point
                s = s_new

    # Evaluate Point
    optimum = function(s)

    return temperature, (s, optimum)

if __name__ == "__main__":

    def main():

        a = 0
        b = 1
        max_iterations = 100000

        # Call Simulated Annealing
        temperature, (x, y) = simualted_annealing(rigid_function, a, b, max_iterations)

        # Plot Data
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.suptitle("Simulated Annealing")

        # Function
        interval = np.linspace(a, b, 1000)
        fx = rigid_function(interval)

        ax1.plot(interval, fx)
        ax1.plot(x, y, 'x', color = 'red')
        ax1.set_title("Rigid Function")

        ax2.plot(temperature)
        ax2.set_title("Cooling Function")

        plt.show()

    main()


