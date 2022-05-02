import numpy as np
from numba import jit
import matplotlib.pyplot as plt

@jit(nopython = True)
def get_exponents(J, T):

    arr = np.zeros(9)

    arr[4 + 4] = np.exp(-8 * J / T)
    arr[4 + 2] = np.exp(-4 * J / T)
    arr[4 + 0] = 1
    arr[4 - 2] = np.exp(4 * J / T)
    arr[4 - 4] = np.exp(8 * J / T)
    
    return arr

@jit(nopython = True)
def get_random_grid(N):

    arr = np.sign(np.random.uniform(-1, 1, (N, N)))

    return arr

@jit(nopython = True)
def compute_weiss_field(a, b, c, d):

    return a + b + c + d

@jit(nopython = True)
def compute_energy(L):

    N = len(L)
    Energy = 0

    for i in range(len(L)):

        for j in range(len(L[i])):

            a = L[(i + 1) % N, j]
            b = L[i, (j + 1) % N]
            c = L[(i - 1) % N, j]
            d = L[i, ( j - 1) % N]

            weiss_field = compute_weiss_field(a, b, c, d)
            Energy += -weiss_field * L[i][j]

    return Energy / 2

@jit(nopython = True)
def run_ising_model(itr, T, L, exponents, warm_up, measure_step):
    """
    The Ising Model is a Mathematical Model of Ferromagnetism in Statistical Mechanics. 
    The Model Consists of Discrete Varaibles that can be +1 or -1.
    """

    # Compute Initial States
    Energy = compute_energy(L)
    Mag = np.sum(L)
    N = len(L)

    # Initialize Variables
    N1 = 0
    E1 = 0
    E2 = 0
    M1 = 0
    M2 = 0

    for i in range(itr):

        # Randomly Select Spin, Compute Weiss Field
        x = int(np.random.uniform(0, 20, 1)[0])
        y = int(np.random.uniform(0, 20, 1)[0])
        S = L[x][y]

        a = L[(x + 1) % N, y]
        b = L[x, (y + 1) % N]
        c = L[(x - 1) % N, y]
        d = L[x, (y - 1) % N]       

        weiss_field = compute_weiss_field(a, b, c, d) 

        # Accept or Reject Trial Step
        index = int(4 + S * weiss_field)
        P = exponents[index]

        if P > np.random.uniform(0, 1, 1)[0]:
            L[x][y] = -S
            Energy += 2 * S * weiss_field
            Mag -= 2 * S
        
        if i > warm_up and i % measure_step == 0:
            N1 += 1
            E1 += Energy
            M1 += Mag
            E2 += Energy**2
            M2 += Mag**2

    N2 = N**2
    E = E1 / N1
    M = M1 / N1
    cv = (E2 / N1 - E**2) / T**2  
    chi = (M2 / N1 - M**2) / T     

    return M / N2, E / N2, cv / N2, chi / N2

if __name__ == '__main__':

    def plot():

        J = 1
        T = 2
        N = 20 

        # Step One: Compute Exponents
        exponents = get_exponents(J, T)
        print("Exponents: ", exponents)

        # Step Two: Initialize Random Gird
        lattice = get_random_grid(N)
        print("Lattice Slice: ", lattice[0])

        # Step Three: Check Energy
        E = compute_energy(lattice)
        print("Energy: ", E)

        # Test Run Ising Model:
        itr = 5000000
        warm_up = 1000
        measure_step = 5
        ising_test = run_ising_model(itr, T, lattice, exponents, warm_up, measure_step)

        print("Ising Test: ", ising_test)

        # Actual Ising Model Run
        wT = np.linspace(5, 0.5, 30)
        wMag = []
        wEne = []
        wCv = []
        wChi = []

        print("Ising Model Running...")
        for i in range(len(wT)):

            exponents = get_exponents(J, wT[i])
            
            (M, E, cv, chi) = run_ising_model(itr, wT[i], lattice, exponents, warm_up, measure_step)

            wMag.append(M)
            wEne.append(E)
            wCv.append(cv)
            wChi.append(chi)

            print('T = ', wT[i], 'M = ', M, 'E = ', E, 'cv = ', cv, 'chi = ', chi)

        print("Ising Model Terminated...")

        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.suptitle("Ising Model")

        ax1.plot(wT, wEne, label = 'E(T)')
        ax1.plot(wT, wCv, label = 'cv(T)')
        ax1.plot(wT, wMag, label = 'M(T)')
        ax1.legend(loc = 'best')

        ax2.plot(wT, wChi, label = 'chi(T)')
        ax2.legend(loc = 'best')

        plt.show()

    plot()

