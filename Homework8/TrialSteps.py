import numpy as np
from numba import jit

@jit(nopython = True)
def trial_step_zero(qx, momentum):
    """
    Input: 
    1. Integration Interval
    2. A Multidimensional Array of Momentum

    Output: 
    1. K_new: 3D Vector
    2. Ka_new: Length of Vector
    3. trial_ratio: Probability to Accept
    4. accept: boolean variable defaults to True for initial case
    5. tiQ: Randomly drawn bin from integration interval
    """

    # Randomly Select Length of Vector
    tiQ = int(np.random.uniform(0 , 1, 1)[0] * len(qx))   
    Ka_new = qx[tiQ]
    
    # Compute Spherical Angles
    th = np.pi * np.random.uniform(0, 1, 1)[0]
    phi = 2 * np.pi * np.random.uniform(0, 1, 1)[0]  
    
    # Trial Step Probability in Sperical Coordinates
    sin_th = np.sin(th)                               
    Q_sin_th = Ka_new * sin_th
    K_new = np.array([Q_sin_th * np.cos(phi), Q_sin_th * np.sin(phi), Ka_new * np.cos(th)])

    q2_sin2_old = sum(momentum[0, :2]**2)    
    q2_old = q2_sin2_old + momentum[0, 2]**2 
    trial_ratio = 1
    
    # Error Checking
    if q2_old != 0:                      
        sin_th_old = np.sqrt(q2_sin2_old / q2_old)  

        if sin_th_old != 0:          
            trial_ratio = sin_th / sin_th_old

    # Defaults to True for Initial
    accept = True

    return (K_new, Ka_new, trial_ratio, accept, tiQ)

@jit(nopython = True)
def trial_step_one(iloop, momentum, dkF, cutoff):
    """
    Input:
    1. Randomly Drawn Bin to Change
    2. Multidimensional Array of Momentum 
    3. Step Size
    4. Integral Cutoff

    Output: 
    1. K_new: 3D Vector
    2. Ka_new: Length of Vector
    3. trial_ratio: Probability to Accept
    4. accept: boolean variable 
    """

    # Cartesian Coordinates
    dk = (2 * np.random.uniform(0, 3, 1)[0] -1) * dkF  

    # Vector
    K_new = momentum[iloop,:] + dk     
    Ka_new = np.linalg.norm(K_new)  

    trial_ratio = 1            

    # Now a Stochastic Variable       
    accept = (Ka_new <= cutoff)        

    return (K_new, Ka_new, trial_ratio, accept)


@jit(nopython=True)
def get_new_x(momentum, K_new, iloop):
    """
    Input:
    1. Multidimensional Array of Momentum
    2. Length of Vector
    3. Randomly Drawn bin from Integral Bounds

    Output:
    1. New Momentum
    """

    tmomentum = np.copy(momentum)
    tmomentum[iloop,:] = K_new  

    return tmomentum




