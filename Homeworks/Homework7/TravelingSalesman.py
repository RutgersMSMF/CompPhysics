import numpy as np
from numba import jit
import matplotlib.pyplot as plt

@jit(nopython = True)
def generate_grid(N):
    """
    Generates random city from uniform points
    """

    x = np.random.uniform(0, 1, N)
    y = np.random.uniform(0, 1, N)

    return x, y

@jit(nopython = True)
def get_euclidean_distance(x, y):
    """
    Returns the Euclidean Norm
    """

    norm = 0

    for i in range(len(x) - 1):

        dx = x[i + 1] - x[i]
        dy = y[i + 1] - y[i]
        norm += np.sqrt((dx + dy)**2)

    return norm

@jit(nopython = True)
def reverse(x, y):
    """
    An alternative path is generated
    The cities in the chosen segment are reversed in order of visit
    """

    new_x = np.zeros(len(x), dtype = np.float64)
    new_y = np.zeros(len(y), dtype = np.float64)

    # Fetch Random Cities
    a = int(np.random.uniform(0, len(x), 1)[0])
    b = int(np.random.uniform(0, len(x), 1)[0])

    if a == b:
        while a == b:
            b = int(np.random.uniform(0, len(x), 1)[0])

    new_x = np.copy(x)
    new_y = np.copy(y)

    new_x[a], new_x[b] = new_x[b], new_x[a]
    new_y[a], new_y[b] = new_x[b], new_x[a]

    return new_x, new_y

@jit(nopython = True)
def transpose(x, y):
    """
    Segment is clipped out of its current position in the path
    Spliced in at a randomly chosen point in the remainder of the path
    """

    new_x = np.zeros(len(x), dtype = np.float64)
    new_y = np.zeros(len(y), dtype = np.float64)

    # Fetch Random Cities
    a = int(np.random.uniform(0, len(x), 1)[0])
    b = int(np.random.uniform(0, len(x), 1)[0])
    c = int(np.random.uniform(0, len(x), 1)[0])
    d = int(np.random.uniform(0, len(x), 1)[0])

    if a == c:
        while a == c:
            c = int(np.random.uniform(0, len(x), 1)[0])

    if b == d:
        while b == d:
            d = int(np.random.uniform(0, len(x), 1)[0])

    new_x = np.copy(x)
    new_y = np.copy(y)

    new_x[a], new_x[c] = new_x[c], new_x[a]
    new_y[a], new_y[c] = new_x[c], new_x[a]

    new_x[b], new_x[d] = new_x[d], new_x[b]
    new_y[b], new_y[d] = new_x[d], new_x[b]

    return new_x, new_y

@jit(nopython = True)
def optimize_grid(x, y, max_itr):
    """
    Implements Simuleated Annealing to Optimize the Traveling Salesman Problem
    """

    last_error = get_euclidean_distance(x, y)
    error = last_error

    count = 0

    while count < max_itr:

        # Decrease "Temperature"
        T = 1 - (count / max_itr)
        rv = np.random.uniform(0, 1, 1)[0]

        if rv < 0.5:

            # Reverse Method
            temp_x, temp_y = reverse(x, y)

            # Check Error
            last_error = error
            error = get_euclidean_distance(temp_x, temp_y)
            diff = last_error - error

            if diff > 0:

                x = temp_x
                y = temp_y

            else: 

                # Metropolis Hastings
                if np.exp(-(error - last_error) / T) > np.random.uniform(0, 1, 1)[0]:

                    # Update Point
                    x = temp_x
                    y = temp_y

        else:

            # Transpose Method
            temp_x, temp_y = transpose(x, y)

            # Check Error
            last_error = error
            error = get_euclidean_distance(temp_x, temp_y)
            diff = last_error - error

            if diff > 0:

                x = temp_x
                y = temp_y

            else: 

                # Metropolis Hastings
                if np.exp(-(error - last_error) / T) > np.random.uniform(0, 1, 1)[0]:

                    # Update Point
                    x = temp_x
                    y = temp_y

        count+=1

    return x, y

if __name__ == "__main__":

    def main():

        N = 100
        max_itr = 100000
        x, y = generate_grid(N)

        norm = get_euclidean_distance(x, y)
        print("Initial Configutation Distance: ", norm)

        rx, ry = reverse(x, y)
        rnorm = get_euclidean_distance(rx, ry)
        print("Reverse Configuration Distance: ", rnorm)

        tx, ty = transpose(x, y)
        tnorm = get_euclidean_distance(tx, ty)
        print("Transpose Configuration Distance: ", tnorm)

        # Test Run

        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.suptitle("Traveling Salesman")

        final_x, final_y = optimize_grid(x, y, 10000)
        print("Optimized Euclidean Distance: ", get_euclidean_distance(final_x, final_y))

        ax1.plot(x, y, '--x', color = 'red')
        ax1.set_title("Initial Configuration")

        ax2.plot(final_x, final_y, '--x', color = 'green')
        ax2.set_title("Final Configuration")

        # Begin Optimization

        N = np.linspace(50, 1000, 100, dtype = np.int64)
        max_itr = np.zeros(len(N))
        norm = np.zeros(len(N))

        for i in range(len(N)):

            x, y = generate_grid(N[i])
            max_itr[i] = N[i] * 100

            final_x, final_y = optimize_grid(x, y, max_itr[i])
            fnorm = get_euclidean_distance(final_x, final_y)
            norm[i] = fnorm

        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.suptitle("Traveling Salesman")

        ax1.plot(x, y, '--x', color = 'red')
        ax1.set_title("Initial Configuration")

        ax2.plot(final_x, final_y, '--x', color = 'green')
        ax2.set_title("Final Configuration")

        fig, (ax1, ax2) = plt.subplots(1, 2)

        ax1.plot(N, norm, '--x', color = 'red')
        ax1.set_title("Optimal Distances")

        finite_difference = np.zeros(len(N) - 1)
        for k in range(1, len(N) - 1):

            finite_difference[k] = (norm[k + 1] - norm[k - 1]) / (N[k + 1] - N[k - 1])

        ax2.plot(finite_difference)
        ax2.set_title("Finite Difference")

        plt.show()

        return 0 

    main()

