import numpy as np
from numba import jit
import matplotlib.pyplot as plt
import IsingModel as IM

@jit(nopython = True)
def prepare_energy(N):

    interval = np.arange(-int(N**2 / 2), int(N**2 / 2) + 1)
    Energies = np.zeros(len(interval), dtype = np.int64)

    for i in range(len(interval)):

        Energies[i] = 4 * interval[i]

    new_energy = np.zeros(len(Energies) - 2, dtype = np.int64)

    index = 0
    length = len(Energies)
    for i in range(length):

        if i != 1 and i != length - 2:
            new_energy[index] = Energies[i]
            index+=1
        
    Energies = new_energy 
    Emin = Energies[0]
    Emax = Energies[-1]

    length = int(Emax + 1 - Emin)
    indE = -np.ones(length, dtype = np.int64) 

    for i, E in enumerate(Energies):

        indE[int(E - Emin)] = i

    return (Energies, indE, Emin)

@jit(nopython = True)
def run_wang_landau(max_itr, L, flatness, Energies, indE):

    N = len(L)
    Energy = int(IM.compute_energy(L))

    Emin = Energies[0]
    Emax = Energies[-1]

    lngE = np.zeros(len(Energies), dtype = np.float64)
    Hist = np.zeros(len(Energies), dtype = np.float64)

    lnf = 1.0   
    N2 = N**2

    for i in range(max_itr):

        # Randomly Select Spin, Compute Weiss Field
        t = int(np.random.rand() * N2)
        x, y = (int(t / N), t % N)
        S = L[x][y]

        a = L[(x + 1) % N, y]
        b = L[x, (y + 1) % N]
        c = L[(x - 1) % N, y]
        d = L[x, (y - 1) % N] 

        weiss_field = IM.compute_weiss_field(a, b, c, d)
        Enew = Energy + int(2 * L[x, y] * weiss_field) 

        lgnew = lngE[indE[Enew - Emin]]
        lgold = lngE[indE[Energy - Emin]]
        P = 1.0

        if lgold - lgnew < 0 : 
            P = np.exp(lgold - lgnew) 

        if P > np.random.uniform(0, 1, 1)[0]:
            L[x, y] = -S
            Energy = Enew

        Hist[indE[Energy - Emin]] += 1
        lngE[indE[Energy - Emin]] += lnf
        
        if (i + 1) % 1000 == 0:
            aH = sum(Hist) / N2 
            mH = min(Hist)

            if mH > aH * flatness:  
                Hist[:] = 0
                lnf /= 2
                print(i, 'histogram is flat', mH, aH, 'f=', np.exp(lnf))

    return (lngE, Hist, Energies)

@jit(nopython = True)
def normalization(lngE):

    if lngE[-1] > lngE[0]:
        lgC = np.log(4) - lngE[-1] - np.log(1 + np.exp(lngE[0] - lngE[-1]))  
    
    else:
        lgC = np.log(4) - lngE[0] - np.log(1 + np.exp(lngE[-1] - lngE[0]))
    
    lngE += lgC

    return lngE

@jit(nopython = True)
def get_thermodynamics(lngE, Energies, T, N):

    Z = 0
    Ev = 0
    E2v = 0

    for i, E in enumerate(Energies):

        w = np.exp(lngE[i] - lngE[0] - (E - Energies[0]) / T) 
        Z += w
        Ev += w*E
        E2v += w*E**2

    Ev *= 1 / Z
    E2v *= 1 / Z
    cv = (E2v - Ev**2) / T**2

    return (Ev / (N**2), cv / (N**2))

if __name__ == '__main__':

    def plot():

        N = 32
        itr = int(10e9)
        exact = [2, 2048, 4096, 1057792, 4218880, 371621888, 2191790080, 100903637504, 768629792768, 22748079183872]

        mean = np.zeros(5)
        std = np.zeros(5)

        for i in range(1):

            print("Iteration: ", i)

            # Generate Random Grid
            lattice = IM.get_random_grid(N)

            # Prepare Energies
            Energy, indE, Energy_min = prepare_energy(N)

            # Run Wang Landau Simulation
            flatness = 0.80

            lngE, Hist, Energies = run_wang_landau(itr, lattice, flatness, Energy, indE)

            # Normalization
            normal = normalization(lngE)    

            mean[i] = np.mean(lngE)
            std[i] = np.std(lngE)

        # Thermodynamics
        T = np.linspace(0.5, 4.0, 300)
        thm = []

        for i in range(300):
            thm.append(get_thermodynamics(normal, Energies, T[i], N))

        thm = np.array(thm)

        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.suptitle("Wang Landau Model")

        ax1.plot(T, thm[:,0], label = 'E(T)')
        ax1.plot(T, thm[:,1], label = 'cv(T)')

        ax2.plot(normal)
        ax2.set_title("Density Curve")

        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.suptitle("Wang Landau Model")

        numerical = np.exp(np.log(lngE[:10]))

        ax1.plot(np.log(exact), label = 'Exact')
        ax1.plot(numerical, label = "Numerical")
        ax1.set_title("Actual vs Numerical")

        ax2.plot(np.abs(np.log(exact) - numerical))
        ax2.set_title("Difference")

        fig, (ax1) = plt.subplots(1, 1)
        fig.suptitle("Wang Landau Model")

        ax1.plot(mean, label = "Mean")
        ax1.plot(std, label = "Standard Deviation")
        ax1.set_title("Statistics")

        plt.show()

    plot()