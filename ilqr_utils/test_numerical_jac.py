from numerical_jac import numerical_jac
import numpy as np

"""
Test 1
f([x]) = [2x + 1]
numerical(f, 1) == 2
numerical(f, 2) == 2
numerical(f, 3) == 2

BE REALLY CAREFUL ABOUT THE DATATYPE OF THE ARRAY
"""
f = lambda x: 2.0 * x + 1.0
correct_ans = np.zeros((1, 1))
correct_ans.fill(2.0)
ans = numerical_jac(f, np.array([1.0]))
assert np.allclose(ans, correct_ans)
ans = numerical_jac(f, np.array([2.0]))
assert np.allclose(ans, correct_ans)
ans = numerical_jac(f, np.array([3.0]))
assert np.allclose(ans, correct_ans)

print("=== Passed Test 1 ===")

"""
Test 2
f([x y]) = [x^2+y^2 y^3]
"""
f = lambda x: np.array([(x[0] * x[0] + x[1] * x[1]), pow(x[1], 3)])
x = np.array([1.0, 2.0])
correct_ans = np.zeros((2, 2))
correct_ans[0, 0] = 2.0 * x[0]
correct_ans[1, 0] = 0.0
correct_ans[0, 1] = 2.0 * x[1]
correct_ans[1, 1] = 3.0 * pow(x[1], 2)
ans = numerical_jac(f, x)
assert np.allclose(ans, correct_ans)

print("=== Passed Test 2 ===")
