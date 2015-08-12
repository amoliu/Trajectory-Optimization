import numpy as np

class LQR(object):
    def __init__(self, T, dt, x_init):
        """
        Initialize constants
        """
        self.T = T
        self.dt = dt
        self.x_init = x_init

    def run_algorithm(self, A_all, B_all, Q_all, R_all):
        """
        Find {u} such that the cost function
        x_f'Q_fx_f + \sum (x_i'Q_ix_i + u_i'R_iu_i)
        is minimized
        """
        x_init = self.x_init
        T = self.T
        nX = Q_all[0].shape[0]
        nU = R_all[0].shape[0]
        dt = self.dt

        K = np.zeros((T, nU, nX))
        P = np.zeros((T, nX, nX))

        P[i] = np.zeros((nX, nX))

        for i in range(T):
            Q = Q_all[i]
            R = R_all[i]
            A = A_all[i]
            B = B_all[i]
            K_curr = -np.linalg.inv(R + B.T.dot.P[i].dot(B)).dot(B.T).dot(P[i]).dot(A)
            tmp = A + B.dot(K)
            P_curr = Q + K_curr.T.dot(R).dot(K_curr) + tmp.T.dot(P[i]).dot(tmp)
            K[i+1] = K_curr
            P[i+1] = P_curr

        U = np.zeros((T, nU))
        x_prev = x_init 
        for i in range(T):
            U[i] = K[i].dot(x_prev)
            x_prev = A_all[i].dot(x_prev) + B_all[i].dot(U[i])
            
        return U
