import numpy as np 
from numba import jit 
import matplotlib.pyplot as plt

@jit(nopython = True)
def generate_city(N):
    """
    Generate Random Configuration
    """

    return np.random.random((N, 2))

@jit(nopython = True)
def get_euclidean_distance(grid):
    """
    Returns the Euclidean Norm
    """

    dx = grid[0][0] - grid[len(grid) - 1][0]
    dy = grid[0][1] - grid[len(grid) - 1][1]
    norm = np.sqrt(dx**2 + dy**2)

    for i in range(len(grid) - 1):

        dx = grid[i + 1][0] - grid[i][0]
        dy = grid[i + 1][1] - grid[i][1]
        norm += np.sqrt(dx**2 + dy**2)

    return norm

@jit(nopython=True)
def find_segment(R):

    nct = len(R) 

    while True:

        n0 = int(nct * np.random.uniform(0, 1, 1)[0])
        n1 = int((nct-1) * np.random.uniform(0, 1, 1)[0])
        
        if n1 >= n0: 
            n1 +=1
        
        if n1 < n0: 
            (n0,n1) = (n1,n0)
        
        nn = (nct-(n1-n0+1))  
        if nn >= 3: 
            break

    n2 = (n0-1) % nct
    n3 = (n1+1) % nct

    return (n0,n1,n2,n3)

@jit(nopython = True)
def get_distance(x, y):

    return np.sqrt(x**2 + y**2)

def get_cost_reverse(R, city, n0, n1, n2, n3):

    de = get_distance(R[city[n2]], R[city[n1]]) + get_distance(R[city[n0]], R[city[n3]])
    de-= get_distance(R[city[n2]], R[city[n0]]) + get_distance(R[city[n1]], R[city[n3]])

    return sum(de)

def get_reverse(R, city, n0, n1, n2, n3):

    newcity = np.copy(city)

    for j in range(n1-n0+1):
        newcity[n0+j] = city[n1-j]

    return newcity

@jit(nopython=True)
def find_t_segment(R):

    (n0,n1,n2,n3) = find_segment(R)
    nct = len(R)

    nn = nct - (n1-n0+1)  # number for the rest of the cities
    n4 = (n1+1 + int(np.random.uniform(0, 1, 1)[0] * (nn-1)) ) % nct # city on the rest of the path
    n5 = (n4+1) % nct

    return (n0,n1,n2,n3,n4,n5)

def get_cost_transpose(R, city, n0,n1,n2,n3,n4,n5):

    de = -get_distance(R[city[n1]], R[city[n3]])
    de-= get_distance(R[city[n0]], R[city[n2]])
    de-= get_distance(R[city[n4]], R[city[n5]])
    de+= get_distance(R[city[n0]], R[city[n4]])
    de+= get_distance(R[city[n1]], R[city[n5]])
    de+= get_distance(R[city[n2]], R[city[n3]])

    return sum(de)

def get_transpose(R, city, n0,n1,n2,n3,n4,n5):

    nct = len(R)
    newcity = []
    # Segment in the range n0,...n1
    for j in range(n1-n0+1):
        newcity.append( city[ (j+n0)%nct ] )
    # is followed by segment n5...n2
    for j in range( (n2-n5)%nct + 1):
        newcity.append( city[ (j+n5)%nct ] )
    # is followed by segement n3..n4
    for j in range( (n4-n3)%nct + 1):
        newcity.append( city[ (j+n3)%nct ] )
    return newcity

# @jit(nopython = True)
def optimize_grid(R, max_itr):
    """
    Implements Simuleated Annealing to Optimize the Traveling Salesman Problem

    Computes the Euclidean Norm at each iteration and measures the loss function
    """

    loss_function = np.zeros(max_itr)
    loss_function[0] = get_euclidean_distance(R)
    city = range(len(R))

    for i in range(1, max_itr):

        # Decrease "Temperature"
        T = 1 - (i / max_itr)
        rv = np.random.uniform(0, 1, 1)[0]

        if rv < 0.5:

            # Reverse Method
            n0, n1, n2, n3 = find_segment(R)
            diff = get_cost_reverse(R, city, n0, n1, n2, n3)

            if diff < 0 or np.exp(-(diff / T)) > np.random.uniform(0, 1, 1)[0]:

                # Update City
                city = get_reverse(R, city, n0, n1, n2, n3)
                loss_function[i] = loss_function[i - 1] + diff
            
            else:
                loss_function[i] = loss_function[i -  1]

        else:

            # Transpose Method
            n0, n1, n2, n3, n4, n5 = find_t_segment(R)
            diff = get_cost_transpose(R, city, n0, n1, n2, n3, n4, n5)

            if diff < 0 or np.exp(-(diff / T)) > np.random.uniform(0, 1, 1)[0]:

                # Update City
                city = get_transpose(R, city, n0, n1, n2, n3, n4, n5)
                loss_function[i] = loss_function[i - 1] + diff
            
            else:
                loss_function[i] = loss_function[i -  1]

    return city, R, loss_function

if __name__ == "__main__":

    def main():

        N = 100
        max_itr = N * 100

        random_grid = generate_city(N)
        print("Initial Euclidean Distance: ", get_euclidean_distance(random_grid))

        city, R, loss_function = optimize_grid(random_grid, max_itr)

        plt.plot(city, R, '--x')

        plt.show()

    main()