import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq
from scipy.integrate import simps

# Define Global Parameters
N = 1000
R = np.linspace(1e-8, 50, N)

def schroedinger_eqn(r, l, E):
    """
    Returns the Solution to the Schroedinger Equation
    """

    return l * (l + 1.0) / r**2 - 2.0 / r - E

def compute_schroedinger(r, l, E):
    """
    Solves the Schroedinger Equation Using Numerov Method
    """

    ur = schroedinger_eqn(r[::-1], l, E)
    ur = numerov_method(E, l)
    norm = simps(ur**2, x = R)
    X = ur * 1 / np.sqrt(abs(norm))

    return X

# Numerov Method to Solve ODE
def numerov_method(E, l):
    """
    Fast Numerical Method Derived by Taylor Series Expansion
    Solves Second Order Differential Equation Without First Order Terms
    """

    # Define Mesh Width
    h = R[1] - R[0]

    # Call Schroedinger
    F = schroedinger_eqn(R[::-1], l, E)

    X = np.zeros(N)
    X[0] = 0
    X[1] = 10**(-8) * h

    # Initial Iteration
    w_naught = X[0] * (1 - h**2 / 12 * F[0])
    w_n = X[1] * (1 - h**2 / 12 * F[1])
    xi = X[1]
    fi = F[1]

    for i in range(2, N):

        w_N = 2 * w_n - w_naught + h**2 * fi * xi  
        fi = F[i]              
        xi = w_N / (1 - h**2 / 12 * fi)     
        X[i] = xi                
        w_naught = w_n
        w_n = w_N

    return X[::-1]

def shooting_method(E, l):
    """
    Numerical Method for Reducing a Boundary Problem to an Initial Value Problem
    """

    X = numerov_method(E, l)
    X /= R**l

    n = X[0]
    N = X[1]

    shot = n + (N - n) * (0.0 - R[0]) / (R[1] - R[0])

    return shot

def find_bound_states():
    """
    Solve For Roots Using Brents Method
    """

    Energy = []
    levels = [0, 1, 2, 3, 4, 5, 6]

    states = np.zeros(2)
    interval = -1.2 / np.linspace(1, 50, N)

    count = 0
    while(count < len(levels)):

        for i in range(N):

            if i == 0:

                # Initial Case
                states[0] = shooting_method(interval[i], levels[count])

            else:

                # Further Iterations
                states[1] = shooting_method(interval[i], levels[count])

                if (states[0] * states[1]) < 0:
                    a = interval[i - 1]
                    b = interval[i]
                    root = brentq(shooting_method, a, b, xtol = 1e-16, args = (levels[count]))

                    temp = (levels[count], root)
                    Energy.append(temp)
                
                states[0] = states[1]

        count+=1

    return Energy

def cmpKey(x):

    return x[1] + x[0] / 10000

def charge_density(bound_states, Z):
    """
    In Quantum Mechanics, a Particle does not have Precise Position. 
    A Particle is Represented by a Probability Distribution.
    """

    level = 0
    psi = np.zeros(N)

    for i in range(len(bound_states)):

        ur = compute_schroedinger(R, bound_states[i][0], bound_states[i][1])
        dN = 2 * (2 * bound_states[i][0] + 1)

        if level + dN <= Z:
            ferm = 1
        else:
            ferm = (Z - level) / dN

        dpsi = ur**2 * ferm * dN / (4 * np.pi * R**2)
        psi += dpsi
        level += dN
        print('adding state', (bound_states[i][0], bound_states[i][1]), 'with fermi=', ferm)

        if level >= Z: 
            return psi

    return psi

if __name__ == "__main__":

    def electron_states(bound_states):

        # Compute Electron States

        states = np.zeros(len(bound_states))
        planck = 6.626e-34
        rydberg = 1.096e7
        light = 2.99e8
        orbital = 1

        for i in range(1, len(bound_states)):

            states[i] = -planck * light * rydberg * (orbital**2 / i**2)

        # Compute Second Part 

        rho = np.zeros(N)

        for i in range(len(bound_states)):

            ur = compute_schroedinger(R, bound_states[i][0], bound_states[i][1])    
            rho += ur**2 / (4 * np.pi * R**2)

        return states, rho

    # Acts as Main Function
    def plot():

        # Part One
        f, (ax1, ax2) = plt.subplots(1, 2)
        f.suptitle("Schroedinger Equation")

        # Call Numerov Method
        X = compute_schroedinger(R, 0, -1)

        ax1.plot(X)
        ax1.set_title("Numerov Method: {l = -1, E = 0}")

        # Print Bound States
        bound_states = find_bound_states()
        print("Bound States: ", bound_states)
        bound_states = sorted(bound_states, key = cmpKey)

        # Call Charge Density
        Z = 28
        density = charge_density(bound_states, Z)
        ax2.plot(R, density * (4 * np.pi * R**2))
        ax2.set_title("Charge Density")

        # Part One
        f2, (ax1, ax2) = plt.subplots(1, 2)
        f2.suptitle("Schroedinger Equation")

        # Call Electron States
        es = electron_states(bound_states)

        ax1.plot(es[0])
        ax1.set_title("Electron States")

        ax2.plot(es[1])
        ax2.set_title("Rho")

        plt.show()

    plot()
