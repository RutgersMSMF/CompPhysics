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

    norm = np.sqrt((x[0] - x[len(x) - 1])**2 + (y[0] - y[len(y) - 1])**2)

    for i in range(len(x) - 1):

        dx = x[i + 1] - x[i]
        dy = y[i + 1] - y[i]
        norm += np.sqrt(dx**2 + dy**2)

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

    if a < b:

        d = int((b - a) / 2) + 1

        for i in range(d):

            new_x[a + i], new_x[b - i] = new_x[b - i], new_x[a + i]
            new_y[a + i], new_y[b - i] = new_y[b - i], new_y[a + i]

    else: 

        d = int((a - b) / 2) + 1

        for i in range(d):

            new_x[a - i], new_x[b + i] = new_x[b + i], new_x[a - i]
            new_y[a - i], new_y[b + i] = new_y[b + i], new_y[a - i]

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

    if a == b:
        while a == b:
            b = int(np.random.uniform(0, len(x), 1)[0])

    if a < b and b < len(x):
        c = int(np.random.uniform(b, len(x), 1)[0])
    
    elif b < a and a < len(x):
        c = int(np.random.uniform(a, len(x), 1)[0])
    
    else: 
        c = 0

    N = len(x)
    new_x = np.copy(x)
    new_y = np.copy(y)

    if a < b:

        d = np.abs(b - a)

        for i in range(d):

            new_x[(a + i) % N], new_x[(c + i) % N] = new_x[(c + i) % N], new_x[(a + i) % N]
            new_y[(a + i) % N], new_y[(c + i) % N] = new_y[(a + i) % N], new_y[(c + i) % N]

    else: 

        d = np.abs(b - a)

        for i in range(d):

            new_x[(b + i) % N], new_x[(c + i) % N] = new_x[(c + i) % N], new_x[(b + i) % N]
            new_y[(b + i) % N], new_y[(c + i) % N] = new_y[(c + i) % N], new_y[(b + i) % N]

    return new_x, new_y

@jit(nopython = True)
def optimize_grid(x, y, max_itr):
    """
    Implements Simuleated Annealing to Optimize the Traveling Salesman Problem

    Computes the Euclidean Norm at each iteration and measures the loss function
    """

    loss_function = np.zeros(int(max_itr), dtype = np.float64)
    last_error = get_euclidean_distance(x, y)
    loss_function[0] = last_error

    for i in range(1, max_itr):

        temp_x = np.zeros(len(x))
        temp_y = np.zeros(len(y))

        # Decrease "Temperature"
        T = 1 - (i / max_itr)
        rv = np.random.uniform(0, 1, 1)[0]

        if rv < 0.5:

            # Reverse Method
            temp_x, temp_y = reverse(x, y)

            # Check Error
            error = get_euclidean_distance(temp_x, temp_y)
            diff = error - last_error

            # Metropolis Hastings
            if diff < 0 or np.exp(-(diff / T)) > np.random.uniform(0, 1, 1)[0]:

                # Update City
                x = temp_x
                y = temp_y
                loss_function[i] = error

                # Update Error
                last_error = error
            
            else:

                loss_function[i] = loss_function[i -  1]
                # Update Error
                last_error = error

        else:

            # Transpose Method
            temp_x, temp_y = transpose(x, y)

            # Check Error
            error = get_euclidean_distance(temp_x, temp_y)
            diff = error - last_error

            # Metropolis Hastings
            if diff < 0 or np.exp(-(diff / T)) > np.random.uniform(0, 1, 1)[0]:

                # Update City
                x = temp_x
                y = temp_y
                loss_function[i] = error

                # Update Error
                last_error = error

            else:

                loss_function[i] = loss_function[i -  1]
                # Update Error
                last_error = error

    return x, y, loss_function

if __name__ == "__main__":

    def main():

        N = 100
        max_itr = 100 * N
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

        final_x, final_y, loss_function = optimize_grid(x, y, max_itr)
        print("Optimized Euclidean Distance: ", get_euclidean_distance(final_x, final_y))

        ax1.plot(x, y, '--x', color = 'red')
        ax1.set_title("Initial Configuration")

        ax2.plot(final_x, final_y, '--x', color = 'green')
        ax2.set_title("Final Configuration")

        fig, (ax1) = plt.subplots(1 ,1)

        ax1.plot(loss_function)
        ax1.set_title("Loss Function")

        # Begin Optimization

        N = np.linspace(50, 1000, 100, dtype = np.int64)
        max_itr = np.zeros(len(N))
        norm = np.zeros(len(N))

        for i in range(len(N)):

            x, y = generate_grid(N[i])
            max_itr[i] = N[i] * 100

            final_x, final_y, _ = optimize_grid(x, y, max_itr[i])
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

