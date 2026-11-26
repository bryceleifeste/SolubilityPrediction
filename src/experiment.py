from ridge_regression import train, predict, objective, mae, rmse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#############################################################################################################
#############################COPIED FROM "prepare_data.py"####################################################
###############################################################################################################
df = pd.read_csv("delaney-processed.csv") # Read in the ESOL csv file

# These are the molecular descriptors we will use in this project
feature_columns = ["Minimum Degree",
                   "Molecular Weight",
                   "Number of H-Bond Donors",
                   "Number of Rings",
                   "Number of Rotatable Bonds",
                   "Polar Surface Area"]

X = df[feature_columns].to_numpy(dtype=float) # Convert the feature columns into a NumPy array
y = df["measured log solubility in mols per litre"].to_numpy(dtype=float) # Convert the target into a Numpy array

esol_predictions = df["ESOL predicted log solubility in mols per litre"].to_numpy(dtype=float) # Keep the existing ESOL predictions for later comparison

rng = np.random.default_rng(42) # Reproducible random split

indices = rng.permutation(len(y)) # Shuffles the array

n = len(y) # Number of compounds
n_train = int(0.70 * n) # 70% train
n_valid = int(0.15 * n) # 15% validation

train_indices = indices[:n_train] # Training group
valid_indices = indices[n_train:n_train + n_valid] # Validation group
test_indices = indices[n_train + n_valid:] # Test group

# Unstandardized splits
X_train_raw = X[train_indices]
X_valid_raw = X[valid_indices]
X_test_raw = X[test_indices]

# Split the target
y_train = y[train_indices]
y_valid = y[valid_indices]
y_test = y[test_indices]

# Split the existing ESOL predictions
esol_valid = esol_predictions[valid_indices]
esol_test = esol_predictions[test_indices]

# Calculating mean and standard deviation using the training data
mean = X_train_raw.mean(axis=0)
std = X_train_raw.std(axis=0)

std[std == 0] = 1.0 # Prevents division by zero error

# Standardize the three datasets using x' = (x - mu) / sigma
X_train = (X_train_raw - mean) / std
X_valid = (X_valid_raw - mean) / std
X_test = (X_test_raw - mean) / std
##################################################################################################################
##################################################################################################################
##################################################################################################################

# Train a ridge regression model
lam = 0.01

w, b, history = train(X_train, y_train, lam=lam)

print("Coefficients:", w)
print("Intercept:", b)
print("Final objective:", history[-1])

y_pred = predict(X_valid, w, b) # Predict validation solubilities

print("Validation MAE:", mae(y_valid, y_pred))
print("Validation RMSE:", rmse(y_valid, y_pred))

baseline_predictions = np.full(len(y_valid), np.mean(y_train)) # Our model needs to outperform this baseline prediction

# Expected output for both baseline errors is higher than our model
print("Baseline MAE:", mae(y_valid, baseline_predictions))
print("Baseline RMSE:", rmse(y_valid, baseline_predictions))

#####################INVESTIGATE REGULARIZATION STRENGTH##################################################
lambdas = [0, 0.0001, 0.001, 0.01, 0.1, 1, 10] # Testing different regularization strengths

results = []

for lam in lambdas:
    w, b, history = train(X_train, y_train, lam=lam) # Train model

    train_predictions = predict(X_train, w, b) # Training predictions
    valid_predictions = predict(X_valid, w, b) # Validation predictions

    # MAE
    train_mae = mae(y_train, train_predictions)
    valid_mae = mae(y_valid, valid_predictions)

    # RMSE
    train_rmse = rmse(y_train, train_predictions)
    valid_rmse = rmse(y_valid, valid_predictions)

    results.append({
        "lambda": lam,
        "training MAE": train_mae,
        "validation MAE": valid_mae,
        "training RMSE": train_rmse,
        "validation RMSE": valid_rmse,
        "coefficient norm": np.linalg.norm(w),
        "final objective": history[-1]
    })

results_df = pd.DataFrame(results) # Convert the results into a pd df

print(results_df) # This table shows how regularization strength affects our model

results_df.to_csv("regularization_results.csv", index=False) # Creates a csv file with our experiment results
########################################################################################################################

# Plot validation error vs regularization strength
plt.plot(results_df["lambda"], results_df["validation MAE"], marker = "o")
plt.xscale("symlog", linthresh=0.0001)
plt.xlabel(r"Regularization strength ($\lambda$)")
plt.ylabel("Validation MAE")
plt.title("Effect of Regularization on Prediction Error")
plt.grid()
plt.show()

# Finding the best regularization strength
best_index = results_df["validation MAE"].idxmin()
best_lambda = results_df.loc[best_index, "lambda"]
print("Best lambda:", best_lambda) # We will use this regularization strength for the final model

# Combine the training and validation sets for final evaluation
X_combined_raw = np.vstack((X_train_raw, X_valid_raw))
y_combined = np.concatenate((y_train, y_valid))

# Recalculating the mean and std using the combined data
combined_mean = X_combined_raw.mean(axis=0)
combined_std = X_combined_raw.std(axis=0)

combined_std[combined_std == 0] = 1.0

X_combined_scaled = (X_combined_raw - combined_mean) / combined_std

X_test_scaled = (X_test_raw - combined_mean) / combined_std

w_final, b_final, history_final = train(X_combined_scaled, y_combined, lam=best_lambda) # Training the final model

final_predictions = predict(X_test_scaled, w_final, b_final) # Predict the test-set solubilities

# Final errors
print("Final test MAE:", mae(y_test, final_predictions))
print("Final test RMSE:", rmse(y_test, final_predictions))

# Comparing our model against the ESOL predictions and mean-prediction baseline
print("Our ridge regression:")
print("MAE:", mae(y_test, final_predictions))
print("RMSE:", rmse(y_test, final_predictions))

print("\nExisting ESOL predictions:")
print("MAE:", mae(y_test, esol_test))
print("RMSE:", rmse(y_test, esol_test))

baseline_test = np.full(len(y_test), np.mean(y_combined))

print("\nMean baseline:")
print("MAE:", mae(y_test, baseline_test))
print("RMSE:", rmse(y_test, baseline_test))

# Predicted vs. measured plot
plt.scatter(y_test, final_predictions, alpha=0.6)

minimum = min(y_test.min(), final_predictions.min())
maximum = max(y_test.max(), final_predictions.max())

plt.plot([minimum, maximum], [minimum, maximum], linestyle="--") # Line of best fit

plt.xlabel("Measured log solubility")
plt.ylabel("Predicted log solubility")
plt.title("Predicted vs. Measured Solubility")

plt.grid()
plt.show()

comparison = pd.DataFrame({"model": ["Ridge Regression", "Existing ESOL", "Mean Baseline"],
                           "MAE": [mae(y_test, final_predictions), mae(y_test, esol_test), mae(y_test, baseline_test)],
                           "RMSE": [rmse(y_test, final_predictions), rmse(y_test, esol_test), rmse(y_test, baseline_test)]})

print(comparison)

comparison.to_csv("final_model_comparison.csv", index=False)

predictions_df = pd.DataFrame({"Measured Solubility": y_test,
                               "Ridge Prediction": final_predictions,
                               "ESOL Prediction": esol_test,
                               "Absolute Ridge Error": np.abs(y_test - final_predictions)})

predictions_df.to_csv("test_predictions.csv", index=False)