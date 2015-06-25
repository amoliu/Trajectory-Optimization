from numerical_grad_hess import numerical_grad_hess
import numpy as np

"""
Test 1
f([x]) = [2x + 1]
numerical(f, 1) == 2
numerical(f, 2) == 2
numerical(f, 3) == 2

BE REALLY CAREFUL ABOUT THE DATATYPE OF THE ARRAY
"""
k = 10.0
f = lambda x: k * x + 1
correct_grad = np.zeros((1, 1))
correct_grad.fill(k)
correct_hess = np.zeros((1, 1))
grad, hess = numerical_grad_hess(f, np.array([1.0]), True)
assert np.allclose(grad, correct_grad)
assert np.allclose(hess, correct_hess)

grad, hess = numerical_grad_hess(f, np.array([2.0]), True)
assert np.allclose(grad, correct_grad)
assert np.allclose(hess, correct_hess)

grad, hess = numerical_grad_hess(f, np.array([3.0]), True)
assert np.allclose(grad, correct_grad)
assert np.allclose(hess, correct_hess)

print("=== Passed Test 1 ===")

"""
Test 2
f([x y]) = [x^2y^3]
"""
f = lambda x: np.array([pow(x[0], 2) * pow(x[1], 3)])
x = np.array([1.0, 2.0])
correct_hess = np.zeros((2, 2))
correct_hess[0, 0] = 2.0 * pow(x[1], 3)
correct_hess[1, 0] = 6.0 * x[0] * pow(x[1], 2)
correct_hess[0, 1] = 6.0 * x[0] * pow(x[1], 2)
correct_hess[1, 1] = 3.0 * pow(x[0], 2) * pow(x[1], 2)

correct_grad = np.zeros((1, 2))
correct_grad[0][0] = 2 * x[0] * pow(x[1], 3)
correct_grad[0][1] = 3 * pow(x[0], 2) * pow(x[1], 2)

grad, hess = numerical_grad_hess(f, x, True)
assert np.allclose(grad, correct_grad) 
assert np.allclose(hess, correct_hess)

print("=== Passed Test 2 ===")
