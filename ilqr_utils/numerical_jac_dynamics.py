import numpy as np

def numerical_jac_dynamics(f, x, u, dt, eps=pow(10, -5)):
    """
    Compute the numerical jacobian for dynamics f using
    current trajectory x and control u
    (TODO) test that this function is correct
    """
    y = f(x, u, dt) 
    grad_x = np.zeros((len(y), len(x)));
    grad_u = np.zeros((len(y), len(u)));
    
    xp = x.copy()

    for i in range(len(x)):
        xp[i] = x[i] + eps / 2
        yhi = f(xp, u, dt)
        xp[i] = x[i] - eps / 2
        ylow = f(xp, u, dt)
        xp[i] = x[i]
        grad_x[:,i] = (yhi - ylow) / eps
    
    up = u.copy()

    for i in range(len(u)):
        up[i] = u[i] + eps / 2
        yhi = f(x, up, dt)
        up[i] = u[i] - eps / 2
        ylow = f(x, up, dt)
        up[i] = u[i]
        grad_u[:,i] = (yhi - ylow) / eps

    return (grad_x, grad_u)
