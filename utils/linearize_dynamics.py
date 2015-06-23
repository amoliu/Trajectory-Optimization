import numpy as np

def linearize_dynamics(f, x_ref, u_ref, dt, my_eps, x_ref_tplus1=None):
    """
    This function linearize the dynamics as follows
    x_{t+1} - x_ref_tplus1 to-first-order-equal-to A*(x_t - x_ref) + 
    B * (u_t - u_ref) + c. And c == 0 and x_ref_tplus1 = x_ref

    Adapted from CS287
    (TODO): Make sure this is a correct implementation
    """
    if x_ref_tplus1 == None:
        x_ref_next = x_ref_tplus1
    else:
        x_ref_next = x_ref

    c = 0
    f_xref_uref = f(x_ref, u_ref, dt)
    A = np.zeros((length(f_xref_uref), length(x_ref)))
    B = np.zeros((length(f_xref_uref), length(u_ref)))
    
    xp = x_ref.copy()

    for i in range(x_ref):
        xp[i] = x_ref[i] + my_eps / 2
        f_hi = f(xp, u_ref, dt)
        xp[i] = x_ref[i] - my_eps / 2
        f_lo = f(xp, u_ref, dt)
        A[:,i] = (f_hi - f_lo ) / my_eps
    
    up = u_ref.copy()
    
    for i in range(u_ref):
        up[i] = u_ref[i] + my_eps / 2
        f_hi = f(x_ref, up, dt)
        up[i] = u_ref[i] - my_eps / 2
        f_lo = f(x_ref, up, dt)
        B[:,i] = (f_hi - f_lo ) / my_eps

    return (A, B, c)
        

