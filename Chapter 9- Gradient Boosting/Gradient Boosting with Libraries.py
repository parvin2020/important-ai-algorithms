from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
import numpy as np

# Features (Area)
X = np.array([
    [50],
    [60],
    [70]
])

# Target (Price)
y = np.array([
    100,
    120,
    140
])

# Build model
gbr = GradientBoostingRegressor(
    n_estimators=2,
    learning_rate=0.5,
    max_depth=1,
    random_state=42
)

# Train
gbr.fit(X, y)

# Predict
y_pred = gbr.predict(X)

print("Actual Prices:")
print(y)

print("\nPredicted Prices:")
print(np.round(y_pred, 2))

# MSE
mse = mean_squared_error(y, y_pred)

print("\nMean Squared Error:")
print(round(mse, 2))

# New sample
x_new = np.array([[65]])

prediction = gbr.predict(x_new)

print("\nPrediction for Area=65:")
print(round(prediction[0], 2))