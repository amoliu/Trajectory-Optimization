import numpy as np
from utils.numerical_jac_dynamics import numerical_jac_dynamics

def compute_linear_approx_dynamics_model(f, curr_X, curr_U, dt, eps=pow(10, -5)): 
    T = curr_X.shape[1]
    assert curr_X.shape[1] == curr_U.shape[1]
    
    linearized_X = np.empty((T - 1, len(curr_X), len(curr_X)))
    linearized_U = np.empty((T - 1, len(curr_X), len(curr_U)))

    # (TODO) parallize this
    for i in range(T-1):
        x = curr_X[:,i]
        u = curr_Y[:,i]
        (jac_x, jac_u) = numerical_jac_dynamics(f, x, u, dt, eps)
        linearized_X[:,i] = jac_x
        linearized_U[:,i] = jac_u

    return (linearized_X, linearized_U)
