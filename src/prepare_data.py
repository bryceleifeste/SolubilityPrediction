import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

print("Feature matrix shape: ", X.shape)
print("Target vector shape: ", y.shape)
print("ESOL prediction shape: ", esol_predictions.shape)

# Display the descriptive statistics
print(df[feature_columns].describe())
print("\nMeasured solubility:")
print(df["measured log solubility in mols per litre"].describe())

# Plot the distribution of measured solubilities
plt.hist(y, bins=30, edgecolor='black')
plt.xlabel("Measured log solubility (mol/L)")
plt.ylabel("Number of molecules")
plt.title("Distribution of Aqueous Solubility")
plt.show()

# Plot the distribution of the molecular descriptors
plt.hist(df["Minimum Degree"].to_numpy(dtype=float), bins=30, edgecolor='black')
plt.xlabel("Minimum Degree")
plt.ylabel("Number of molecules")
plt.title("Distribution of Minimum Degree")
plt.show()

plt.hist(df["Molecular Weight"].to_numpy(dtype=float), bins=30, edgecolor='black')
plt.xlabel("Molecular Weight (g/mol)")
plt.ylabel("Number of molecules")
plt.title("Distribution of Molecular Weight")
plt.show()

plt.hist(df["Number of H-Bond Donors"].to_numpy(dtype=float), bins=30, edgecolor='black')
plt.xlabel("Number of H-Bond Donors")
plt.ylabel("Number of molecules")
plt.title("Distribution of Number of H-Bond Donors")
plt.show()

plt.hist(df["Number of Rings"].to_numpy(dtype=float), bins=30, edgecolor='black')
plt.xlabel("Number of Rings")
plt.ylabel("Number of molecules")
plt.title("Distribution of Number of Rings")
plt.show()

plt.hist(df["Number of Rotatable Bonds"].to_numpy(dtype=float), bins=30, edgecolor='black')
plt.xlabel("Number of Rotatable Bonds")
plt.ylabel("Number of molecules")
plt.title("Distribution of Number of Rotatable Bonds")
plt.show()

plt.hist(df["Polar Surface Area"].to_numpy(dtype=float), bins=30, edgecolor='black')
plt.xlabel(r"Polar Surface Area ($\AA^2$)")
plt.ylabel("Number of molecules")
plt.title("Distribution of Polar Surface Area")
plt.show()

rng = np.random.default_rng(42) # Reproducible random split

indices = rng.permutation(len(y)) # Shuffles the array

n = len(y) # Number of compounds
n_train = int(0.70 * n) # 70% train
n_valid = int(0.15 * n) # 15% validation

train_indices = indices[:n_train] # Training group
valid_indices = indices[n_train:n_train + n_valid] # Validation group
test_indices = indices[n_train + n_valid:] # Test group

# Split the features
X_train = X[train_indices]
X_valid = X[valid_indices]
X_test = X[test_indices]

# Split the target
y_train = y[train_indices]
y_valid = y[valid_indices]
y_test = y[test_indices]

# Split the existing ESOL predictions
esol_valid = esol_predictions[valid_indices]
esol_test = esol_predictions[test_indices]

print("Training:", X_train.shape) # Number of compounds in the training group is 789
print("Validation:", X_valid.shape) # Number of compounds in the validation group is 169
print("Testing:", X_test.shape) # Number of compounds in the test group is 170

# Calculating mean and standard deviation using the training data
mean = X_train.mean(axis=0)
std = X_train.std(axis=0)

std[std == 0] = 1.0 # Prevents division by zero error

# Standardize the three datasets using x' = (x - mu) / sigma
X_train = (X_train - mean) / std
X_valid = (X_valid - mean) / std
X_test = (X_test - mean) / std

print("Training feature means:")
print(X_train.mean(axis=0))

print("Training feature standard deviations:")
print(X_train.std(axis=0))