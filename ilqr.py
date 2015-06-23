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
        X = np.zeros((self.T, len(x_init))
        X[0] = x_init
        x_next = x_init

        # (TODO) figure out whether to add the (T+1)th state
        for i in range(self.T - 1):
            x_next = self.f(x_next, U[i], dt)
            X[i+1] = x_next

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
        curr_X = np.zeros((T, len(self.init_state)))
        curr_init_state = self.init_state
        curr_X[0] = curr_init_state
        curr_U = self.control_init
        
        nX = len(self.init_state)
        nU = len(curr_U)
        
        while not converged:
            # Execute current policy and record the current state-input trajectory {x}, {u}
            if self.traj_init == None:
                curr_X = self.execute(curr_init_state, curr_U)

            # Compute LQ approximation of the optimal control problem around state-input trajectory
            # by computing a first-order Taylor expansion of the dynamics model and 
            # a second order Taylor expansion of the cost function

            ## First order taylor approximation of the dynamics model
            A_X, B_U = compute_linear_approx_dynamics_model(f, curr_X, curr_Y, self.dt)

            ## Second order Taylor expansion of the cost function
            Q_X, R_U = compute_quadratic_approx_cost(curr_X, curr_Y)

            # Use the LQR backups to solve for the optimal control policy for LQ approximation obtained
            # in previous state

            # Compute S_k, K, K_v, K_u, v_k (Consider now only the case that Q_i are all the same)
            """
            Write out the equations here:



            """
            S = np.zeros((T+1, nX, nX))
            Q_N = Q_X[-1]
            S[-1] = Q_N
            DELTA_U = np.zeros((T, nU))
            V = np.zeros((T, nX)) # (TODO) What should V[T] be
            for i in range(T):
                index = T - i - 1

                x = curr_X[index]
                u = curr_U[index]

                A = A_X[index]
                B = B_U[index]

                S_n = S[index+1]
                K_help = np.inv(B.T.dot(S_n).dot(B) + R)
                K = K_help.dot(B.T).dot(S_n).dot(A)

                S_k = A.T.dot(S_n).dot(A - B.dot(K)) + Q
                S[index] = S_k

                K_v = K_help.dot(B.T)
                K_u = K_help.dot(R)

                v_n = V[index+1]
                v = (A - B.dot(K)).dot(v) - K.T.dot(R).dot(u) + Q.dot(x)
                V[index] = v

                delta_u = -K.dot(delta_x) - K_v.dot(v_next) - K_u.dot(u)
                DELTA_U[index] = delta_u


            # Test convergence

    return curr_U

