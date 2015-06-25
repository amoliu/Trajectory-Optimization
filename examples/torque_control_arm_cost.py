import numpy as np

def inverted_pendulum_cost(X, U):
    """
    Given sequence of states and control u, returns 
    the cost
    """
    T = len(U)
    assert T + 1 == len(X)

    nX = X.shape[1]
    nY = U.shape[1]

    Q_f = np.eye(nX)
    final_state = nX[-1]
    r = pow(10, -5)

    cost = 0
    cost += final_state.T.dot(Q_f).dot(final_state)
    cost += r * np.sum(np.sum(np.power(Q, 2)))
    cost *= 0.5

    return cost
