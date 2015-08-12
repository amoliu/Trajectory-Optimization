import numpy as np
from ilqr import iLQR
from math import sin, pi
import matplotlib.pyplot as plt
from torque_control_arm_dynamics import f
from compute_quadratic_approx_cost_torque import compute_quadratic_approx_cost_torque
import argparse

"""
This file contains a test example of a simple pendulum
"""

def execute_trajectory(x_init, U, dt):
    """
    Executes the control and plot the trajectory
    """
    T = len(U)
    traj = np.zeros((T + 1, len(x_init)))
    traj[0] = x_init
    for i in range(1, T+1):
        control = U[i-1]
        next_state = f(traj[i - 1], control, dt)
        traj[i] = next_state

    return traj

def plot_trajectory(traj):
    X1 = traj[:,0] # theta1
    X2 = traj[:,1] # theta2
    T = len(traj)
    t = np.arange(0, T)

    plt.plot(t, X1, "r--", t, X2, "bs")
    plt.xlabel("Time")
    plt.ylabel("Theta1, Theta2")
    plt.show()

def main(T, dt):
    T = T
    dt = dt

    x_init = np.array([1.5, 1.5, -0.05, -0.05])
    U_init = np.zeros((T, 2))

    Q_f = np.eye(4) # terminal cost
    Q = np.zeros((4, 4)) # cost matrix for states

    ilqr_solver = iLQR(f, T, dt, x_init, U_init, compute_quadratic_approx_cost_torque, Q = Q, Q_f = Q_f)
    threshold = pow(10, -3)
    U = ilqr_solver.run_algorithm(threshold) # column i of U should be control input at time i

    # Execute the control sequence
    traj = execute_trajectory(x_init, U, dt)

    plot_trajectory(traj)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("T", type=int, help="number of time steps")
    parser.add_argument("dt", type=float, help="dt")
    args = parser.parse_args()
    T = args.T
    dt = args.dt

    main(T, dt)
