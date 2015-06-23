import numpy as np
from compute_linear_approx_dynamics_model import compute_linear_approx_dynamics_model
from compute_quadratic_approx_cost import compute_quadratic_approx_cost

class iLQR(object):
    def __init__(self, f, T, dt, init_state, control_init, traj_init=None):
        """
        Initialize constants
        """
        self.f = f
        self.T = T
        self.dt = dt
        self.init_state = init_state
        self.control_init = control_init
        self.traj_init = traj_init

    def execute(self, x_init, U):
        X = np.zeros((len(x_init), self.T))
        X[:, 0] = x_init
        x_next = x_init

        # (TODO) figure out whether to add the (T+1)th state

        for i in range(self.T - 1):
            x_next = self.f(x_next, U[:,i], dt)
            X[:, i+1] = x_next

        return X

    def run_algorithm(self):
        """
        (TODO): Figure out how to take into account the dynamics (using the simulation)
        
        """
        """
        Initialization: either (a) a control policy (b) a sequence of states + a sequence of
                        control policy

        Psuedo-code
        Loop until convergence(control sequence u doesn't change much):
            1. Compute A_k, B_k for all time k (linearization)
            2. Compute S_k for all time k using backwards recursion, boundary condition: S_N = Q_f
            3. Compute K, K_v, K_u for all time k
            4. Compute v_k for all time k
            5. Computer delta u_k = -K delta x_k - K_v v_{k+1} - K_u u_k
        Can be done using dynamic programming
        """
        converged = False
        f = self.f
        dt = self.dt
        curr_X = np.zeros((len(self.init_state), T))
        curr_init_state = self.init_state
        curr_X[:,0] = curr_init_state
        curr_U = self.control_init
        
        while not converged:
            # Execute current policy and record the current state-input trajectory {x}, {u}
            if self.traj_init == None:
                curr_X = self.execute(curr_init_state, curr_U)

            # Compute LQ approximation of the optimal control problem around state-input trajectory
            # by computing a first-order Taylor expansion of the dynamics model and 
            # a second order Taylor expansion of the cost function

            ## First order taylor approximation of the dynamics model
            linear_approx_dynamics_model = compute_linear_approx_dynamics_model(f, curr_X, curr_Y, self.dt)

            ## Second order Taylor expansion of the cost function
            quadratic_approx_cost_func = compute_quadratic_approx_cost(curr_X, curr_Y)

            # Use the LQR backups to solve for the optimal control policy for LQ approximation obtained
            # in previous state
            # (TODO)

            
