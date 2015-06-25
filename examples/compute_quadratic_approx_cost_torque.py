import numpy as np

def compute_quadratic_approx_cost_torque(X, U, Q_f, Q):
    """
    Computes the quadratic approximation of the cost
    All the cost function

    Write it in the form of 
    J_0 = 1/2(x_N - x*)'Q_f(X_N - x*) 
         + 1/2 sum_{i=0}^{N-1} (x_k'Qx_k + u_k'Ru_l)

    """
    T = len(U)
    nX = X.shape[1]
    nU = U.shape[1]

    cost_traj = np.empty((T+1, nX, nX))
    cost_control = np.empty((T, nU, nU))

    r = pow(10, -4)

    # Below is the computation for the cost of the trajectory
    cost_traj[-1] = Q_f
    for i in range(T):
        cost_traj[i] = Q

    # Below is the computation for the cost of the control
    for i in range(T):
        cost_control[i] = r * np.eye(2)

    return (cost_traj, cost_control)
