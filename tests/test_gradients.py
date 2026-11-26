import numpy as np
from ridge_regression import gradients, finite_difference_w

# Comparing the gradient (analytical gradient) and finite_difference (numerical gradient) functions
X = np.array([
    [1.0, 2.0],
    [2.0, 1.0],
    [3.0, 4.0]
])

y = np.array([2.0, 3.0, 5.0])

w = np.array([0.5, -0.5])
b = 1.0
lam = 0.1

grad_w, grad_b = gradients(X, y, w, b, lam)

for j in range(len(w)):
    numerical = finite_difference_w(X, y, w, b, lam, j)

    print("Coefficient:", j)
    print("Analytical:", grad_w[j])
    print("Numerical:", numerical)