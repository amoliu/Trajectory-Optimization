import numpy as np

def compute_quadratic_approx_cost(cost_f, X, U):
    """
    All the cost function
    Write it in the form of 

    J_0 = 1/2(x_N - x*)'Q_f(X_N - x*) 
         + 1/2 sum_{i=0}^{N-1} (x_k'Qx_k + u_k'Ru_l)

    """

    # How do I implement this shit..

    T = len(U)
    assert T+1 == len(X)
    nX = X.shape[1]
    nU = U.shape[1]

    cost_traj = np.empty((T+1, nX, nX))
    cost_control = np.empty((T, nU, nU))



    return (cost_traj, cost_control)
