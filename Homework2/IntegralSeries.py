import numpy as np
from numba import jit
import matplotlib.pyplot as plt

# Implement Adaptive Quadrature
@jit(nopython = True)
def integrate(alpha, zeta, a, b, n):

    m = (a + b) / 2.0 
    tol = 10**(-16)

    h = (b - a)
    trapezoid = h / 2 * (f(alpha, zeta, a, n) + f(alpha, zeta, b, n))
    composite = h / 4 * (f(alpha, zeta, a, n) + 2 * f(alpha, zeta, m, n) + f(alpha, zeta, b, n))

    diff = abs(composite - trapezoid)
    area = composite

    if diff > tol:
        lower = integrate(alpha, zeta, a, m, n)
        upper = integrate(alpha, zeta, m, b, n)
        area = lower + upper

    return area

# Implement Taylor Series Method
def taylor_method(alpha, zeta, n, m, string):

    sum = 0
    count = 0
    while count < m:

        if string == "Up" :
            sum += (-zeta) ** count / (alpha ** (count + 1)) * (1 / (n + count + 1))
            
        if string == "Down":
            sum += (-alpha) ** count / (zeta ** (count + 1)) * (1 / (n + count + 1))

        count+=1

    return sum

# Integral Function
@jit(nopython = True)
def f(alpha, zeta, x, n):

    return x**(n + 1) / (zeta + alpha * x)

# Adaptive Function 
def K_adaptive(alpha, zeta, a, b, n):

    return integrate(alpha, zeta, a, b, n)

# Taylor Series
def K_taylor(alpha, zeta, n, m, string):

    return taylor_method(alpha, zeta, n, m, string)

# Upward Recursion Method
def upward_recursion(alpha, zeta, kn, n):

    return (1 / (alpha * (n + 1))) - ((zeta / alpha) * kn)

# Downward Recursion Method
def downward_recursion(alpha, zeta, kn, n):

    return ((-alpha / zeta) * kn) + (1 / (zeta * (n + 1)))

# Numerical Stability Recursion
def combined_recursion(alpha, zeta, kn, n):

    if np.abs(alpha / zeta) > 1:
        return upward_recursion(alpha, zeta, kn, n)

    else:
        return downward_recursion(alpha, zeta, kn, n)

def homework():

    # Arrays to Return
    adaptive_upward = np.zeros(10)
    adaptive_downward = np.zeros(10)
    taylor_upward = np.zeros(10)
    taylor_downward = np.zeros(10)

    # Declare Bounds
    a = 0
    b = 1
    n = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    # Adaptive Method Upward Recursion
    alpha = 15
    zeta = 2
    k_naught = K_adaptive(alpha, zeta, a, b, n[0])
    adaptive_upward[0] = k_naught
    print("Initial Area Upward Adaptive Method: ", k_naught)

    for i in range(1, len(n)):
        area = combined_recursion(alpha, zeta, k_naught, n[i])
        adaptive_upward[i] = area
        k_naught = area

    # Adaptive Method Downward Recursion
    alpha = 2
    zeta = 15
    k_naught = K_adaptive(alpha, zeta, a, b, n[-1])
    adaptive_downward[0] = k_naught
    print("Initial Area Downward Adaptive Method: ", k_naught)

    index = 1
    for j in range(len(n) - 1, 0, -1):
        area = combined_recursion(alpha, zeta, k_naught, n[j])
        adaptive_downward[index] = area
        k_naught = area
        index+=1

    # Taylor Method Upward Recursion
    alpha = 15
    zeta = 2
    m = int(np.log(10**-16) / np.log(alpha / zeta))
    k_naught = K_taylor(alpha, zeta, n[0], abs(m), "Up")
    taylor_upward[0] = k_naught
    print("Initial Area Upward Taylor Method: ", k_naught)

    for i in range(1, len(n)):
        area = combined_recursion(alpha, zeta, k_naught, n[i])
        taylor_upward[i] = area
        k_naught = area

    # Taylor Method Downward Recursion
    alpha = 2
    zeta = 15
    m = int(np.log(10**-16) / np.log(alpha / zeta))
    k_naught = K_taylor(alpha, zeta, n[-1], m, "Down")
    taylor_downward[0] = k_naught
    print("Initial Area Downward Taylor Method: ", k_naught)

    index = 1
    for j in range(len(n) - 1, 0, -1):
        area = combined_recursion(alpha, zeta, k_naught, n[j])
        taylor_downward[index] = area
        k_naught = area
        index+=1

    return (adaptive_upward, adaptive_downward, taylor_upward, taylor_downward)

def plot():

    # Call Homework Function
    au, ad, tu, td = homework()

    # Part One
    f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
    f.suptitle("Recursive Series Integrals")

    ax1.plot(au)
    ax1.set_title('Adaptive Method Upward Recursion')

    ax2.plot(ad[::-1]) # Reverse Order of Array
    ax2.set_title('Adaptive Method Downward Recursion')

    ax3.plot(tu)
    ax3.set_title('Taylor Method Upward Recursion')

    ax4.plot(td[::-1]) # Reverse Order of Array
    ax4.set_title('Taylor Method Downward Recursion')

    # Part Two
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle("Recursive Series Integrals")

    ax1.plot(au, label = "Adaptive Up")
    ax1.plot(ad[::-1], label = "Adaptive Down")
    ax1.plot(tu, label = "Taylor Up")
    ax1.plot(td[::-1], label = "Taylor Down")
    ax1.set_title('Recursive Methods Comparison')
    ax1.legend(loc = 'best')

    ax2.plot(au - tu, label = "Up Difference")
    ax2.plot(ad[::-1] - td[::-1], label = "Down Difference")
    ax2.set_title('Recursive Methods Difference')
    ax2.legend(loc = 'best')

    plt.show()

plot()
