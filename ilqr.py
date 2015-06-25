import numpy as np
from compute_linear_approx_dynamics_model import compute_linear_approx_dynamics_model
from compute_quadratic_approx_cost import compute_quadratic_approx_cost
from compute_quadratic_approx_cost_pendulum import compute_quadratic_approx_cost_pendulum

class iLQR(object):
    def __init__(self, f, T, dt, init_state, control_init, cost_func, traj_init=None, Q=None, Q_f=None):
        """
        Initialize constants
        """
        self.f = f
        self.T = T
        self.dt = dt
        self.init_state = init_state
        self.control_init = control_init
        self.traj_init = traj_init
        self.Q = Q
        self.Q_f = Q_f
        self.cost_func = cost_func

    def execute_control(self, f, x_init, U, dt):
        """
        Executes the control and plot the trajectory
        """
        T = len(U)
        traj = np.zeros((T + 1, len(x_init)))
        traj[0] = x_init
        for i in range(1, T+1):
            control = U[i-1]
            next_state = f(traj[i - 1], control, dt)
            traj[i] = next_state

        return traj

    def run_algorithm(self, threshold):
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
        T = self.T
        dt = self.dt
        Q = self.Q
        Q_f = self.Q_f
        cost_func = self.cost_func

        curr_X = np.zeros((T+1, len(self.init_state)))
        curr_X[0] = self.init_state
        curr_U = self.control_init
        prev_X = None
        prev_U = None
        
        nX = len(self.init_state)
        nU = curr_U.shape[1]

        max_iter = 5
        iteration = 0
        
        while not converged and iteration < max_iter:
            # Execute current policy and record the current state-input trajectory {x}, {u}
            if self.traj_init == None:
                curr_X = self.execute_control(f, self.init_state, curr_U, dt)

            # Compute LQ approximation of the optimal control problem around state-input trajectory
            # by computing a first-order Taylor expansion of the dynamics model and 
            # a second order Taylor expansion of the cost function
            #
            ## First order taylor approximation of the dynamics model
            A_X, B_U = compute_linear_approx_dynamics_model(f, curr_X, curr_U, self.dt)

            ## Second order Taylor expansion of the cost function
            ## (TODO) Use an actual approximation and use the below function instaed
            # Q_X, R_U = compute_quadratic_approx_cost(curr_X, curr_U)
            Q_X, R_U = compute_quadratic_approx_cost_pendulum(curr_X, curr_U, Q_f, Q)

            # Use the LQR backups to solve for the optimal control policy for LQ approximation obtained
            # in previous state


            # Compute S_k, K, K_v, K_u, v_k (Consider now only the case that Q_i are all the same)
            """
            Write out the equations here:
            """
            Q_N = Q_X[-1]
            V_N = Q_N.dot(curr_X[-1])

            S = np.zeros((T+1, nX, nX))
            S[-1] = Q_N

            V = np.zeros((T+1, nX))
            V[-1] = V_N

            R = R_U[0] #(TODO)
            Q = Q_X[0] #(TODO)

            DELTA_U = np.zeros((T, nU))

            for i in range(T):
                index = T - i - 1

                x = curr_X[index]
                u = curr_U[index]

                A = A_X[index]
                B = B_U[index]

                S_n = S[index+1]
                K_help = np.linalg.inv(B.T.dot(S_n).dot(B) + R)
                K = K_help.dot(B.T).dot(S_n).dot(A)

                S_k = A.T.dot(S_n).dot(A - B.dot(K)) + Q
                S[index] = S_k

                K_v = K_help.dot(B.T)
                K_u = K_help.dot(R)

                v_n = V[index+1]
                v = (A - B.dot(K)).dot(v_n) - K.T.dot(R).dot(u) + Q.dot(x)
                V[index] = v

                # delta_x =  (TODO) Figure this out and double check this??
                delta_x = curr_X[index+1] - x
                delta_u = -K.dot(delta_x) - K_v.dot(v_n) - K_u.dot(u)
                DELTA_U[index] = delta_u


            curr_U = curr_U + DELTA_U # (TODO) Is this correct??

            # Test convergence
            if prev_X != None and prev_U != None:
                diff_X = curr_X - prev_X
                diff_U = curr_U - prev_U
                abs_diff_X = np.sum(np.sum(abs(diff_X), axis=1), axis=0)
                abs_diff_U = np.sum(np.sum(abs(diff_U), axis=1), axis=0)
                abs_diff = abs_diff_X + abs_diff_U

                if abs_diff < threshold:
                    converged = True

            prev_X = curr_X
            prev_U = curr_U
            
            iteration += 1

        return curr_U

