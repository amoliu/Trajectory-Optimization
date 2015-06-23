import numpy as np

class iLQR(object):
    def __init__(self):
        """
        Initialize constants
        """
        pass

    def run_algorithm(self):
        """
        (TODO): Figure out how to take into account the dynamics (using the simulation)
        
        """
        """
        Psuedo-code
        Loop until convergence(control sequence u doesn't change much):
            1. Compute A_k, B_k for all time k (linearization)
            2. Compute S_k for all time k using backwards recursion, boundary condition: S_N = Q_f
            3. Compute K, K_v, K_u for all time k
            4. Compute v_k for all time k
            5. Computer delta u_k = -K delta x_k - K_v v_{k+1} - K_u u_k
        Can be done using dynamic programming
        """
        pass

        
