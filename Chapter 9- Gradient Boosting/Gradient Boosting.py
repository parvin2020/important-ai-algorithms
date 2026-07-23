import numpy as np

# Actual house prices
y = np.array([100, 120, 140])

# Step 1: Initial prediction (mean)
F0 = np.mean(y)

print("Initial Prediction (F0):", F0)

# Residuals
residual1 = y - F0

print("\nResiduals after first model:")
print(residual1)

# Learning rate
lr = 0.5

# Step 2
F1 = F0 + lr * residual1

print("\nPredictions after Tree 1:")
print(F1)

# New residuals
residual2 = y - F1

print("\nResiduals after Tree 1:")
print(residual2)

# Step 3
F2 = F1 + lr * residual2

print("\nPredictions after Tree 2:")
print(F2)

print("\nActual Values:")
print(y)