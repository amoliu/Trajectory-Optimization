import numpy as np
from ilqr import iLQR
from math import sin, pi
import matplotlib.pyplot as plt
from inverted_pendulum_dynamics import f
# from inverted_pendulum_cost import inverted_pendulum_cost
from compute_quadratic_approx_cost_pendulum import compute_quadratic_approx_cost_pendulum
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
    
    final_state = traj[-1]
    print("The final state is {}".format(final_state))
    return traj

def plot_trajectory(traj):
    X1 = traj[:,0] # theta
    X2 = traj[:,1] # velocity
    plt.plot(X1, X2)
    plt.xlabel("theta")
    plt.ylabel("velocity")
    plt.show()

def main(T, dt):
    # Consider a particular example (maybe an example from the paper)?
    T = T
    dt = dt

    x_init = np.array([pi, 0])
    U_init = np.zeros((T, 1))

    Q_f = np.eye(2) # terminal cost
    Q = np.zeros((2, 2)) # cost matrix for states

    ilqr_solver = iLQR(f, T, dt, x_init, U_init, compute_quadratic_approx_cost_pendulum, Q = Q, Q_f = Q_f)
    threshold = pow(10, -6)
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
