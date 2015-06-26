import numpy as np
from math import sin

def f(x, u, dt):
    """
    Dynamics for pendulum
    dx1/dt = x2
    dx2/dt = g/l sinx1 - mu/(ml^2) x2 + 1/(ml^2) mu
     
    x1 = theta
    x2 = d(theta)/ dt

    (TODO) Look up how to implement a pendulum system
    """
    m = 1.0
    l = 1.0
    g = 9.8
    mu = 0.01
    r = pow(10, -5)
    
    DT = 0.1
    t = 0

    assert len(u) == 1 # control is one dimensional
    u = u[0]

    x1 = x[0]
    x2 = x[1]

    # while t < dt:
    current_dt = min(DT, dt - t)

    dx1 = x2
    dx2 = g / l * sin(x1) - mu / (m * l * l) * x2 + 1 / (m * l * l) * u

    # x1 = x1 + dx1 * current_dt
    # x2 = x2 + dx2 * current_dt
    x1 = x1 + dx1 * dt
    x2 = x2 + dx2 * dt

    # t = t + current_dt
    
    return np.array([x1, x2])
