import numpy as np
from math import sin, cos

def getM(a1, a2, a3, theta2):
    M = np.zeros((2, 2))
    M[0, 0] = a1 + 2 * a2 * cos(theta2)
    M[0, 1] = a3 + a2 * cos(theta2)
    M[1, 0] = a3 + a2 * cos(theta2)
    M[1, 1] = a3
    return M

def getC(a2, theta2, dtheta1, dtheta2):
    C = np.zeros(2)
    C[0] = -dtheta2 * (2 * dtheta1 + dtheta2)
    C[1] = pow(dtheta1, 2)
    C = C * a2 * sin(theta2)
    return C

def f(x, u, dt):
    """
    Dynamics for torque controlled arms
    (TODO) Look up how to implement a pendulum system
    """
    B = np.array([[0.05, 0.025], [0.025, 0.05]])
    m1 = 1.4
    m2 = 1.0
    l1 = 0.30 # using m or cm?
    l2 = 0.33 # using m or cm?
    s1 = 0.11 # using m or cm?
    s2 = 0.16 # using m or cm?
    I1 = 0.025
    I2 = 0.045
    a1 = I1 + I2 + m2 * l1 * l1
    a2 = m2 * l1 * s2
    a3 = I2

    DT = 0.1
    t = 0

    assert len(x) == 4 # state is four dimensional
    assert len(u) == 2 # control is two dimensional

    theta1 = x[0]
    theta2 = x[1]
    dtheta1 = x[2]
    dtheta2 = x[3]

    while t < dt:
        current_dt = min(DT, dt - t)
        dtheta = np.array([dtheta1, dtheta2])

        # check when there's too big of a number detected
        if np.isnan(dtheta1) or np.isnan(dtheta2):
            raw_input("!! nan detected")
            import pdb; pdb.set_trace()

        ddtheta = np.linalg.inv(getM(a1, a2, a3, theta2)).dot(u - getC(a2, theta2, dtheta1, dtheta2) - B.dot(dtheta))
        ddtheta1 = ddtheta[0]
        ddtheta2 = ddtheta[1]

        theta1 = theta1 + dtheta1 * current_dt
        theta2 = theta2 + dtheta2 * current_dt
        dtheta1 = dtheta1 + ddtheta1 * current_dt
        dtheta2 = dtheta2 + ddtheta2 * current_dt

        t = t + current_dt
    
    return np.array([theta1, theta2, dtheta1, dtheta2])
