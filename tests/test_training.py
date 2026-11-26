import numpy as np
import matplotlib.pyplot as plt
from ridge_regression import train

# Test dataset
X = np.array([
    [1, 2],
    [2, 1],
    [3, 4],
    [4, 3],
    [5, 2],
    [2, 5]
], dtype=float)

y = 2 * X[:, 0] - 3 * X[:, 1] + 1 # Generate the target values using the equation y = 2x_1 - 3x_2 + 1

w, b, history = train(X, y, lam=0, learning_rate=0.01) # Training without regularization

print("Calculated coefficients:", w) # Expected output [2, -3]
print("Calculated intercept:", b) # Expected output 1

print("Final objective:", history[-1]) # I also checked the full history and saw that the objective calculations decreased, which is expected

# Plot the convergence
plt.plot(history)
plt.xlabel("Iteration")
plt.ylabel("Objective function")
plt.title("Gradient Descent Convergence")
plt.grid()
plt.show()