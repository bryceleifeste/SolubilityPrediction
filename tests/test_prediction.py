import numpy as np
from ridge_regression import predict

# Testing the prediction function with a simple example
X_test = np.array([
    [1, 2],
    [3, 4]
])

w = np.array([2, 3])
b = 1

print(predict(X_test, w, b)) # Expected output is [ 9 19]