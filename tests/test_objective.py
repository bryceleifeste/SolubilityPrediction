import numpy as np
from ridge_regression import objective

# Testing the objective function with a simple example
X = np.array([[1], [2]])
y = np.array([3, 5])

w = np.array([2])
b = 1

print(objective(X, y, w, b, 0)) # The prediction is [ 3, 5], so expected output should be zero
print(objective(X, y, w, b, 1)) # A nonzero regularization value should increase the objective function since the penalty is positive