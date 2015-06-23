import numpy as np
from numerical_jac import numerical_jac

def numerical_grad_hess(f, x, full_hessian):
    """
    Computes the numerical gradient and diagonal hessian
    full_hessian - if True, returns the full hessian and returns the 
                    only the diagonal hessian
    """
    y = f(x)
    assert len(y) == 1

    grad = np.zeros((1, len(x)))
    hess = np.zeros((len(x), len(x)))

    eps = pow(10, -5)
    xp = x.copy()
     
    if not full_hessian:
        for i in range(len(x)):
            xp[i] = x[i] + eps / 2
            yhi = f(xp)
            xp[i] = x[i] - eps / 2
            ylo = f(xp)
            xp[i] = x[i]
            hess[i, i] = (yhi + ylo - 2 * y) / (pow(eps, 2) / 4)
            grad[i] = (yhi - ylo) / eps
    else:
        grad = numerical_jac(f, x)
        hess = numerical_jac(lambda x: np.asarray(numerical_jac(f, x)).reshape(-1), x) 
        hess = (hess + hess.T) / 2

    return (grad, hess)
