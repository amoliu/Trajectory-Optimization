import numpy as np
from ilqr import iLQR

"""
This file should contain the a test example
"""


def main():
    # (TODO)Consider a particular exmaple
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
