import numpy as np
from math import cos

def getJacX(x, u, dt):
    m = 1
    l = 1
    g = 9.8
    mu = 0.01
    x1 = x[0]
    x2 = x[1]
    u = u[0]
    jac = np.zeros((len(x), len(x)))
    jac[0, 0] = 1
    jac[1, 0] = dt * (g / l * cos(x1))
    jac[0, 1] = dt
    jac[1, 1] = 1 - mu / (m * pow(l, 2)) * dt
    return jac

def getJacU(x, u, dt):
    m = 1
    l = 1
    g = 9.8
    mu = 0.01
    jac = np.zeros((len(x), len(u)))
    jac[0, 0] = 0
    jac[1, 0] = 1 / (m * pow(l, 2)) * dt
    return jac

def compute_linear_pendulum(f, curr_X, curr_U, dt, eps=pow(10, -5)): 
    T = curr_U.shape[0]
    assert curr_X.shape[0] == curr_U.shape[0]+1

    nX = curr_X.shape[1]
    nU = curr_U.shape[1]
    
    linearized_X = np.empty((T, nX, nX)) # should this have been (T+1) instead?
    linearized_U = np.empty((T, nX, nU))

    # (TODO) parallelize this 
    for i in range(T):
        x = curr_X[i]
        u = curr_U[i]
        linearized_X[i] = getJacX(x, u, dt)
        linearized_U[i] = getJacU(x, u, dt)

    # (TODO) Do i need to take care of T+1 here?

    return (linearized_X, linearized_U)
