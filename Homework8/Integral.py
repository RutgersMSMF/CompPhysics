import time 
import numpy as np
import matplotlib.pyplot as plt

import Linhard as L 
import TrialSteps as TS
import IntegralObj as Obj
from Weight import measureWeight

def metropolis_integral(function, qx, P, mweight):
    """
    Computes the Integral Using Metropolis Method from Lecture

    Input: 
    1. The Function to Integrate
    2. Integration Mesh, Discrete Points
    3. Parameters

    Output:
    1. Approximate Area Under Curve
    """

    # Time Computational Speed
    # tm1 = time.time()

    # CHANGE: Store Final Variables as Complex Type in Pval
    Pval = np.zeros((len(qx), len(function.Omega)), dtype = np.complex64) 

    # Set Variables
    Pnorm = 0.0            
    Pval_sum = 0.0         
    Pnorm_sum = 0.0       
    V0norm = P.V0norm     
    dk_hist = 1.0          
    Ndim = function.Ndim       
    inc_recompute = (P.per_recompute + 0.52) / P.per_recompute 

    # Initialize Momentum
    momentum = np.zeros((Ndim, 3), dtype = np.complex64) 
    iQ = int(len(qx) * np.random.uniform(0, 1, 1)[0]) 
    momentum[1:, :] = np.random.uniform(0, Ndim - 1, 3) * P.kF / np.sqrt(3)
    momentum[0, :] = [0, 0, qx[iQ]] 

    fQ = function.GetLinhard(momentum), V0norm * mweight(momentum)  

    # t_sim, t_mes, t_prn, t_rec = 0, 0, 0, 0
    Nmeasure = 0  
    Nall_q, Nall_k, Nall_w, Nacc_q, Nacc_k = 0, 0, 0, 0, 0
    c_recompute = 0 

    # Begin Integration 
    for itt in range(P.Nitt):

        # Initialize Starting Time
        # t0 = time.time()

        # Randomly Draw Bin
        iloop = int(Ndim * np.random.uniform(0, 1, 1)[0])  
        accept = False

        # Initial Trial Step
        if (iloop == 0):                     
            Nall_q += 1                                      
            (K_new, Ka_new, trial_ratio, accept, tiQ) = TS.trial_step_zero(qx, momentum)

        # All Other Steps
        else:   
            Nall_k += 1                        
            (K_new, Ka_new, trial_ratio, accept) = TS.trial_step_one(iloop, momentum, P.dkF, P.cutoff)

        # Trial Step Acceptance
        if accept == True:

            # Update Momentum
            tmomentum = TS.get_new_x(momentum, K_new, iloop)
            fQ_new = function.GetLinhard(tmomentum), V0norm * mweight(tmomentum)

            # CHANGE: Modify Trial Ratio
            ratio = (np.abs(fQ_new[0][0]) + fQ_new[1]) / (np.abs(fQ[0][0]) + fQ[1]) * trial_ratio 

            # Metropolis Hastings Step
            accept = abs(ratio) > 1 - np.random.uniform(0, 1, 1)[0]
            
            if accept == True: 

                momentum[iloop] = K_new
                fQ = fQ_new

                # Initial Trial Step
                if iloop == 0:
                        Nacc_q += 1  
                        iQ = tiQ     
                
                # All Other Steps
                else:
                        Nacc_k += 1

        # t1 = time.time()
        # t_sim += t1 - t0

        if (itt >= P.Nwarm and itt % P.tmeasure == 0): 

            Nmeasure += 1

            # Initialize Weight
            W = np.abs(fQ[0][0]) + fQ[1]

            # CHANGE: Adjust Weight
            f0 = fQ[0] / W
            f1 = fQ[1]/W       

            # CHANGE: Integration Boundary
            Pval[iQ,:] += f0                  
            Pnorm += f1                  
            Pnorm_sum += f1               

            # CHANGE: Total Accumulation  
            Wphs = np.abs(f0[0])               
            Pval_sum += Wphs
 
            mweight.Add_to_K_histogram(dk_hist * Wphs, momentum, P.cutoff, P.cutoff)

            # Recompute Variables
            if (itt > 10000 and itt % (P.recomputew * P.tmeasure) == 0):

                P_v_P = Pval_sum / Pnorm_sum * 0.1 
                change_V0 = 0

                # Lower Threshold
                if (P_v_P < 0.25 and itt < 0.2 * P.Nitt):  
                    change_V0 = -1  
                    V0norm /= 2  
                    Pnorm /= 2   
                    Pnorm_sum /= 2  

                # Upper Threshold
                if (P_v_P > 4.0 and itt < 0.2 * P.Nitt):
                    change_V0 = 1   
                    V0norm *= 2  
                    Pnorm *= 2
                    Pnorm_sum *= 2

                # Conditional for Change
                if change_V0 == True:       

                    schange = ["V0 reduced to ", "V0 increased to"]
                    print("Iteration: ", (itt / 1e6, P_v_P))
                    print("Change: ", schange[int((change_V0 + 1) / 2 )], V0norm)

                    # CHANGE: Data Type to Complex
                    Pval = np.zeros(np.shape(Pval), dtype = np.complex64)  
                    Pnorm = 0
                    Nmeasure = 0

                if (c_recompute == 0 and itt < 0.7 * P.Nitt):

                    # t5 = time.time()
                    P.per_recompute = int(P.per_recompute * inc_recompute + 0.5)
                    dk_hist *= 5 * mweight.Normalize_K_histogram()

                    if dk_hist < 1e-8: 
                        dk_hist = 1.0

                    mweight.Recompute()
                    fQ = function.GetLinhard(momentum), V0norm * mweight(momentum)          
                    # t6 = time.time()
                    print("Iteration: ", (itt / 1e6, fQ[1]))
                    # t_rec += t6-t5

                c_recompute += 1

                if (c_recompute >= P.per_recompute): 
                    c_recompute = 0 

        # t2 = time.time()
        # t_mes += t2-t1

        if (itt + 1)% P.Ncount == 0:

            P_v_P = Pval_sum/Pnorm_sum * 0.1 
            Qa = qx[iQ]                     
            ka = np.linalg.norm(momentum[1,:])  

            # CHANGED: Trial Ratio
            ratio = (abs(fQ_new[0][0]) + fQ_new[1]) / (abs(fQ[0][0]) + fQ[1]) 

            # CHANGED: Iteration Statistics
            print("Iteration: ", itt/1e6)
            print("Mesh Porint: ", Qa)
            print("Norm: ", ka)
            print("New Function: ", abs(fQ_new[0][0]))
            print("New Function: ", fQ_new[1])
            print("Old Function: ", abs(fQ[0][0]))
            print("Old Function: ", fQ[1])
            print("P: ", P_v_P)
            print("####################")
        
        # t3 = time.time()
        # t_prn += t3-t2
    
    Pval *= len(qx) * V0norm / Pnorm  
    # tp1 = time.time()

    print("Total Acceptance Rate: ", (Nacc_k + Nacc_q) / (P.Nitt + 0.0))
    print("K Acceptance: ", Nacc_k / (Nall_k + 0.0))
    print("Q Acceptance: ", Nacc_q / (Nall_q + 0.0))

    print("K Trials: ", Nall_k / (P.Nitt + 0.0))
    print("Q Trials: ", Nall_q / (P.Nitt + 0.0))

    # print("Simulation Time: ", t_sim)
    # print("Measurement Time: ", t_mes)
    # print("Recompute Time: ", t_rec)
    # print("Print Time: ", t_prn)
    # print("Total Time: ", tp1 - tm1)

    return Pval, mweight

if __name__ == "__main__":

    def main():

        # Define Parameters
        rs = 2.0
        kF = (9 * np.pi/4)**(1/3) / rs   
        nF = kF / (2 * np.pi**2)            
        T = 0.02 * kF**2                  
        broad = 0.002 * kF**2              
        cutoff = 3 * kF             
        omega = np.linspace(0, kF**2, 100)      
        qx = np.linspace(0.1 * kF, 0.4 * kF, 4)

        # Initialize Objects
        Linhard = L.Linhard(omega, kF, T, broad)
        Parameters = Obj.params()
        Parameters.Nitt = 5000000
        Ndim = Linhard.Ndim

        # Initialize Momentum
        momentum = np.zeros((Ndim, 3), dtype = np.complex64) 
        iQ = int(len(qx) * np.random.uniform(0, 1, 1)[0]) 
        momentum[1:, :] = np.random.uniform(0, Ndim - 1, 3) * Parameters.kF / np.sqrt(3)
        momentum[0, :] = [0, 0, qx[iQ]] 

        # Test Trial Step Zero
        K_new, Ka_new, trial_ratio, accept, tiQ = TS.trial_step_zero(qx, momentum)

        iloop = int( Ndim * np.random.uniform(0, 1, 1)[0] ) 

        # Test Trial Step One
        K_new, Ka_new, trial_ratio, accept = TS.trial_step_one(iloop, momentum, Parameters.dkF, Parameters.cutoff)

        # Test New Momentum
        new_momentum = TS.get_new_x(momentum, K_new, iloop)

        # Import Weight from Lecture
        mweight = measureWeight(Parameters.dexp, Parameters.cutoff, Parameters.kF, Parameters.Nbins, Ndim)

        # Call Metropolis Integration Routine

        # Initialize Starting Time
        start = time.time()
        Pval, mweight = metropolis_integral(Linhard, qx, Parameters, mweight)
        end = time.time()
        print("Metropolis Integration Total Time: ", end - start)

        for iq in range(np.shape(Pval)[0]):

            plt.plot(omega, Pval[iq,:].real / nF, label = "Real")
            plt.plot(omega, Pval[iq,:].imag / nF, label = "Imaginary")

        plt.title("Metropolis Integration")
        plt.legend(loc = "best")
        plt.show()

        return 0

    main()


