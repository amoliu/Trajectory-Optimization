import numpy as np
from ilqr import iLQR

"""
This file contains a test example of a simple pendulum
"""


def main():
    # Consider a particular example (maybe an example from the paper)?
    m = 1
    l = 1
    g = 9.8
    mu = 0.01
    r = pow(10, -5)

    T = 50
    nX = 10

    # Get a sequence of control input u_1,..., u_T
    solver = iLQR()
    U = solver.run_algorithm() # column i of U should be control input at time i

    X = np.zeros((nX, T))
    # (TODO)Execute the control sequence
    X[:,0] = starting_state
    for t in range(T - 1):
        X[:,t+1] = f(X[:,t], U[:,t])

    # (TODO) Plot the trajectory


if __main__ == "__main__":
    main()
