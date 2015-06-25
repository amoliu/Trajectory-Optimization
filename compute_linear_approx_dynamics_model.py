import numpy as np
from utils.numerical_jac_dynamics import numerical_jac_dynamics

def compute_linear_approx_dynamics_model(f, curr_X, curr_U, dt, eps=pow(10, -5)): 
    T = curr_U.shape[0]
    assert curr_X.shape[1] == curr_U.shape[1]+1
    
    linearized_X = np.empty((T, len(curr_X), len(curr_X))) # should this have been (T+1) instead?
    linearized_U = np.empty((T, len(curr_X), len(curr_U)))

    # (TODO) parallize this 
    for i in range(T):
        x = curr_X[i]
        u = curr_Y[i]
        (jac_x, jac_u) = numerical_jac_dynamics(f, x, u, dt, eps)
        linearized_X[i] = jac_x
        linearized_U[i] = jac_u

    # (TODO) Do i need to take care of T+1 here?

    return (linearized_X, linearized_U)
