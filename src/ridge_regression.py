import numpy as np

def predict(X, w, b):
    """This function implements the prediction function"""
    return X @ w + b

def objective(X, y, w, b, lam):
    """This function implements the objective function"""
    predictions = predict(X, w, b)
    mse = np.mean((predictions - y) ** 2) # The first term of the objective function is the mean squared error
    penalty = lam * np.sum(w ** 2) # The second term of the objective function penalizes large coefficients
    return mse + penalty

def gradients(X, y, w, b, lam):
    """This function implements the gradient calculations"""
    n = len(y) # Number of training molecules
    predictions = predict(X, w, b) # The model's current predictions
    errors = predictions - y # Prediction - measured solubility
    grad_w = (2 / n) * (X.T @ errors) + 2 * lam * w # Gradient for every descriptor coefficient
    grad_b = (2 / n) * np.sum(errors) # Gradient for the intercept
    return grad_w, grad_b # Returns the coefficient and intercept gradients

def finite_difference_w(X, y, w, b, lam, j, eps=1e-6):
    """This function verifies the gradient calculations using finite differences"""
    w_plus = w.copy()
    w_minus = w.copy()

    w_plus[j] += eps
    w_minus[j] -= eps

    J_plus = objective(X, y, w_plus, b, lam)
    J_minus = objective(X, y, w_minus, b, lam)

    return (J_plus - J_minus) / (2 * eps)

def train(X, y, lam, learning_rate=0.01, max_iterations=100000, tolerance=1e-10):
    """This function implements the gradient descent algorithm"""
    n_features = X.shape[1]

    # Initial parameters
    w = np.zeros(n_features) # Initialized the coefficients to zero
    b = np.mean(y) # Initialize the intercept to the mean training solubility

    history = [] # An empty list to store the objective values to monitor convergence

    previous_objective = objective(X, y, w, b, lam)

    for iteration in range(max_iterations):
        grad_w, grad_b = gradients(X, y, w, b, lam) # Calculate the gradients

        # Update parameters
        w -= learning_rate * grad_w
        b -= learning_rate * grad_b

        current_objective = objective(X, y, w, b, lam) # Calculate the new objective

        history.append(current_objective) # Records the objective

        # Stops when the objective changes by less than the specified tolerance or if max iteration count is reached
        if abs(previous_objective - current_objective) < tolerance:
            break

        previous_objective = current_objective # Updates the previous objective with the current objective

    return w, b, history

def mae(y_true, y_pred):
    """This function calculates the mean absolute error"""
    return np.mean(np.abs(y_true - y_pred))

def rmse(y_true, y_pred):
    """This function calculates the root mean squared error"""
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

# Testing the prediction function with a simple example
X_test = np.array([
    [1, 2],
    [3, 4]
])

w = np.array([2, 3])
b = 1

print(predict(X_test, w, b)) # Expected output is [ 9 19]

# Testing the objective function with a simple example
X = np.array([[1], [2]])
y = np.array([3, 5])

w = np.array([2])
b = 1

print(objective(X, y, w, b, 0)) # The prediction is [ 3, 5], so expected output should be zero
print(objective(X, y, w, b, 1)) # A nonzero regularization value should increase the objective function since the penalty is positive

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