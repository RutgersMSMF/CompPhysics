import numpy as np
from numba import jit
import matplotlib.pyplot as plt
from scipy.integrate import simps
from scipy.optimize import brentq

import sys
sys.path.insert(0, 'C:/Users/steve/Documents/GitHub/CompPhysics/Homeworks/Homework3')
import Schroedinger

from Excor import ExchangeCorrelation

N = 1000

@jit(nopython=True)
def numerov(f, x0, dx, dh):

    X = np.zeros(len(f))

    X[0] = x0
    X[1] = x0 + dh * dx

    h2 = dh * dh
    h12 = h2 / 12

    w0 = x0 * (1 - h12 * f[0])
    w1 = X[1] * (1 - h12 * f[1])
    xi = X[1]
    fi = f[1]

    for i in range(2,len(f)):
        w2 = 2 * w1 - w0 + h2 * fi * xi
        fi = f[i]
        xi = w2 / (1 - h12 * fi)
        X[i] = xi
        w0 = w1
        w1 = w2

    return X

@jit(nopython = True)
def numerov_up(U, x0, dh):

    X = np.zeros(len(U))

    X[0] = x0
    X[1] = dh**2 + x0

    h2 = dh**2
    h12 = h2 / 12

    w0 = X[0] - h12 * U[0]
    w1 = X[1] - h12 * U[1]
    Ui = U[1]

    for i in range(2, len(U)):
        w2 = 2 * w1 - w0 + h2 * Ui
        Ui = U[i]
        xi = w2 + h12 * Ui
        X[i] = xi
        w0 = w1 
        w1 = w2

    return X

@jit(nopython = True)
def hartree(psi, Z, r):

    ux = -8 * np.pi * r * psi
    U2 = numerov_up(ux, 0.0, r[1] - r[0])
    alpha2 = (2 * Z - U2[-1]) / r[-1]
    U2 += alpha2 * r

    return U2

@jit(nopython = True)
def rs(x):

    if x < 1e-100: 
        return 1e100

    return (3 / (4 * np.pi * x))**(1 / 3)

def schroedinger_eqn(r, l, E, Uks):
    """
    Returns the Solution to the Schroedinger Equation
    """

    return (l * (l + 1) / r + Uks) / r - E

def compute_schroedinger(r, l, E, Uks):
    """
    Solves the Schroedinger Equation Using Numerov Method
    """

    f = schroedinger_eqn(r[::-1], l, E, Uks)
    ur = numerov(f, 0.0, -1e-10,  r[1] - r[0])[::-1]
    norm = simps(ur**2, x = r)
    X = ur / np.sqrt(abs(norm))

    return X

def shooting_method(Esearch, l, Uks, r):
    """
    Numerical Method for Reducing a Boundary Problem to an Initial Value Problem
    """

    X = compute_schroedinger(r, l, Esearch, Uks)
    X *= 1 / r**l

    poly = np.polyfit(r[:4], X[:4], deg = 3)    

    return np.polyval(poly, 0.0)

def find_bound_states(Uks, Esearch, l, R):
    """
    Solve For Roots Using Brents Method
    """

    Energy = []
    states = np.zeros(2)

    for i in range(len(Esearch)):

        if i == 0:

            # Initial Case
            states[0] = shooting_method(Esearch[i], l, Uks, R)
            print(states[0])

        else:

            # Further Iterations
            states[1] = shooting_method(Esearch[i], l, Uks, R)

            if (states[0] * states[1]) < 0:
                a = Esearch[i - 1]
                b = Esearch[i]
                root = brentq(shooting_method, a, b, xtol = 1e-16, args = (l, Uks, R))

                temp = (l, root)
                Energy.append(temp)
                print('Found bound state at E=%14.9f' % root)
            
            states[0] = states[1]

    return Energy

def cmpKey(x):

    return x[1] + x[0] / 10000

def charge_density(bound_states, Z, Uks, R):
    """
    In Quantum Mechanics, a Particle does not have Precise Position. 
    A Particle is Represented by a Probability Distribution.
    """

    Ebs = 0
    level = 0
    psi = np.zeros(len(R))

    for i in range(len(bound_states)):

        ur = compute_schroedinger(R, bound_states[i][0], bound_states[i][1], Uks)
        dN = 2 * (2 * bound_states[i][0] + 1)

        if level + dN <= Z:
            ferm = 1
        else:
            ferm = (Z - level) / dN

        dpsi = ur**2 * ferm * dN / (4 * np.pi * R**2)
        psi += dpsi
        level += dN
        Ebs += bound_states[i][1] * dN * ferm
        print('adding state', (bound_states[i][0], bound_states[i][1]), 'with fermi=', ferm)

        if level >= Z: 
            return psi, Ebs

    return psi, Ebs

def new_density(exc, Z, R):

    Eold = 0
    nmax = 3
    mixr = 1/2
    Etol = 1e-8
    psi_old = 0

    N = 2**14 + 1
    Uks = -2 * np.ones(N)

    E0 = -1.2 * Z**2
    Eshift = 0.5                                                                                                                     
    Esearch = -np.logspace(-4, np.log10(-E0 + Eshift), 200)[::-1] + Eshift

    for i in range(len(Esearch)):

        Bnd = []
        for l in range(nmax - 1):
            Bnd += find_bound_states(Uks, Esearch, l, R)
        
        Bnd = sorted(Bnd, key=cmpKey)
        psi_new, Ebs = charge_density(Bnd, Z, Uks, R)

        if i == 0:
            psi = psi_new
        else:
            psi = psi_new * mixr + (1 - mixr) * psi_old

        psi_old = np.copy(psi)

        U2 = hartree(psi, Z, R)

        Vxc = np.zeros(len(psi))
        for j in range(len(psi)):

            Vxc[j] = 2 * exc.Vc(rs(psi[j])) + 2 * exc.Vx(rs(psi[j]))
        
        Uks = U2 - 2 * Z + Vxc * R

        ExcVxc = np.zeros(len(psi))
        for j in range(len(psi)):

            ExcVxc[j] = 2 * exc.EcVc(rs(psi[j])) + 2 * exc.ExVx(rs(psi[j]))

        pot = (ExcVxc * R - 0.5 * U2) * R * psi * 4 * np.pi
        epot = simps(pot, x = R)
        Etot = epot + Ebs
        
        print('Total density has weight', simps(psi * (4 * np.pi * R**2), x = R))        
        print('Iteration', i, 'Etot[Ry] = ', Etot, 'Etot[Hartre] = ', Etot/2, 'Diff = ', np.abs(Etot - Eold))

        if  i > 0 and abs(Etot-Eold) < Etol: 
            break

        Eold = Etot

    return psi

def compute_total_energy(new_pdf, exch_corr, r):
    """
    E[_xc] = E[_x] + E[_c]
    """

    total_E = simps(new_pdf * exch_corr, x = r)

    return total_E

def plot():

    R = np.linspace(1e-8, 50, N)

    # Print Bound States
    bound_states = Schroedinger.find_bound_states()
    bound_states = sorted(bound_states, key = Schroedinger.cmpKey)

    Z = 28
    psi = Schroedinger.charge_density(bound_states, Z)

    f, (ax1, ax2) = plt.subplots(1, 2)
    f.suptitle("Charge Density")

    ax1.plot(psi * (4 * np.pi * R**2))
    ax1.set_title("Probability Density")

    ax2.plot(psi)
    ax2.set_title("Psi")

    f, (ax1, ax2) = plt.subplots(1, 2)
    f.suptitle("Hartree Equation")

    U2 = hartree(psi, Z, R)

    ax1.plot(U2)
    ax1.set_title("Numerov")

    exc = ExchangeCorrelation()

    Vxc = np.zeros(len(psi))
    for i in range(len(psi)):

        Vxc[i] = 2 * exc.Vc(rs(psi[i])) + 2 * exc.Vx(rs(psi[i]))

    Uks = U2 - 2 * Z + Vxc * R

    ax2.plot(-Uks / 2)   
    ax2.set_title("Exchange Correlation")

    # Part Two
    Z = 8
    R = np.linspace(1e-8, 20, 2**14 + 1)

    new_pdf = new_density(exc, Z, R)
    
    f, (ax1) = plt.subplots(1, 1)
    f.suptitle("New Electron Density")

    ax1.plot(new_pdf * (4 * np.pi * R**2))

    f, (ax1, ax2) = plt.subplots(1, 2)
    f.suptitle("New Hartree Equation")

    U2 = hartree(new_pdf, Z, R)

    ax1.plot(U2)
    ax1.set_title("New Numerov")

    Vxc = np.zeros(len(new_pdf))
    for i in range(len(new_pdf)):

        Vxc[i] = 2 * exc.Vc(rs(new_pdf[i])) + 2 * exc.Vx(rs(new_pdf[i]))

    Uks = U2 - 2 * Z + Vxc * R

    ax2.plot(-Uks / 2)   
    ax2.set_title("New Exchange Correlation")

    total_energy = compute_total_energy(new_pdf * (4 * np.pi * R**2), -Uks / 2, R)
    print("********************")
    print("Total Energy: ", total_energy)

    plt.show()

plot()