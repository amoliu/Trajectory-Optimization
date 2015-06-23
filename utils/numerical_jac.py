import numpy as np

def numerical_jac(f, x, eps=pow(10, -5)):
    y = f(x) 
    grad = np.zeros((len(y), len(x)));
    
    xp = x.copy()

    for i in range(len(x)):
        xp[i] = x[i] + eps / 2
        yhi = f(xp)
        xp[i] = x[i] - eps / 2
        ylow = f(xp)
        xp[i] = x[i]
        grad[:,i] = (yhi - ylow) / eps
    
    return grad
