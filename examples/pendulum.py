import numpy as np
from ilqr import iLQR
from math import sin, pi
import matplotlib.pyplot as plt
from inverted_pendulum_dynamics import f

"""
This file contains a test example of a simple pendulum
"""

def execute_trajectory(x_init, U, dt):
    """
    Executes the control and plot the trajectory
    """
    T = len(u)
    traj = np.zeros((T + 1, len(x_init)))
    traj[0] = x_init
    for i in range(1, T+1):
        control = U[i-1]
        next_state = f(traj[i - 1], control, dt)
        traj[i] = next_state

    return traj

def plot_trajectory(traj):
    X1 = traj[:,0] # theta
    X2 = traj[:,1] # velocity
    plt.plot(X1, X2)
    plt.xlabel("theta")
    plt.ylabel("velocity")
    plt.show()

def main():
    # Consider a particular example (maybe an example from the paper)?
    T = 50
    nX = 10
    dt = 0.1

    x_init = np.array([pi/2, 0])
    # Get a sequence of control input u_1,..., u_T
    solver = iLQR()
    U = solver.run_algorithm() # column i of U should be control input at time i


    # Execute the control sequence
    traj = execute_trajectory(x_init, U, dt)

    plot_trajectory(traj)


if __main__ == "__main__":
    main()
