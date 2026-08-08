# ============================================================
# GRU - Jena Climate
# Predicting the next-hour temperature
# ============================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense


# 1. Load dataset
# Put jena_climate_2009_2016.csv next to this Python file.
data = pd.read_csv("jena_climate_2009_2016.csv")

print("Dataset shape:", data.shape)


# 2. Select the features used by the model
features = [
    "T (degC)",   # Temperature
    "p (mbar)",   # Air pressure
    "rh (%)",     # Relative humidity
    "wv (m/s)"    # Wind speed
]

data = data[features]


# 3. Check the data
print("\nFirst 5 rows:")
print(data.head())

print("\nMissing values:")
print(data.isnull().sum())


# 4. Remove incomplete rows
data = data.dropna()


# 5. Show temperature over time
plt.figure(figsize=(10, 4))
plt.plot(data["T (degC)"])
plt.xlabel("Time")
plt.ylabel("Temperature (°C)")
plt.title("Jena Climate Temperature")
plt.show()


# 6. Split the time series into Train and Test
# The first 80% is used for training.
# The last 20% is used for testing.
split = int(len(data) * 0.8)

train_data = data.iloc[:split].copy()
test_data = data.iloc[split:].copy()

print("\nTraining samples:", len(train_data))
print("Test samples:", len(test_data))


# 7. Normalize the data
# IMPORTANT: fit is performed only on Training data.
scaler = MinMaxScaler()

scaler.fit(train_data)

train_scaled = scaler.transform(train_data)
test_scaled = scaler.transform(test_data)


# 8. Create time windows
# The previous 24 time steps are used to predict the next temperature.
window = 24


def create_sequences(data, window):

    X = []
    y = []

    for i in range(window, len(data)):

        # Previous 24 time steps = input
        X.append(data[i-window:i])

        # Current temperature = target
        y.append(data[i, 0])

    return np.array(X), np.array(y)


# 9. Create Training and Test sequences
X_train, y_train = create_sequences(
    train_scaled,
    window
)

X_test, y_test = create_sequences(
    test_scaled,
    window
)

print("\nX_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)

print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)


# 10. Build the GRU model
model = Sequential()

# GRU learns temporal patterns in the previous 24 observations.
model.add(
    GRU(
        32,
        input_shape=(window, len(features))
    )
)

# One output: predicted temperature
model.add(Dense(1))


# 11. Show model structure
model.summary()


# 12. Compile the model
model.compile(
    optimizer="adam",
    loss="mse"
)


# 13. Train the model
print("\nTraining started...")

history = model.fit(
    X_train,
    y_train,
    epochs=10,
    batch_size=32,
    validation_split=0.1,
    verbose=1
)


# 14. Predict the Test data
prediction = model.predict(
    X_test,
    verbose=0
)


# 15. Convert predictions back to Celsius
prediction_real = (
    prediction[:, 0]
    * scaler.data_range_[0]
    + scaler.data_min_[0]
)

y_test_real = (
    y_test
    * scaler.data_range_[0]
    + scaler.data_min_[0]
)


# 16. Calculate MAE
mae = mean_absolute_error(
    y_test_real,
    prediction_real
)


# 17. Calculate RMSE
rmse = np.sqrt(
    mean_squared_error(
        y_test_real,
        prediction_real
    )
)


# 18. Show evaluation results
print("\n================================")
print("Model Evaluation")
print("================================")

print("MAE  =", round(mae, 2), "°C")
print("RMSE =", round(rmse, 2), "°C")


# 19. Compare actual and predicted temperature
plt.figure(figsize=(10, 4))

plt.plot(
    y_test_real,
    label="Actual"
)

plt.plot(
    prediction_real,
    label="Predicted"
)

plt.xlabel("Time")
plt.ylabel("Temperature (°C)")
plt.title("Actual vs Predicted Temperature")
plt.legend()
plt.show()


# 20. Show training loss
plt.figure(figsize=(8, 4))

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("GRU Training Loss")
plt.legend()
plt.show()


# 21. Predict the next temperature
# Use the latest 24 observations.
last_24_hours = test_scaled[-window:]

last_24_hours = last_24_hours.reshape(
    1,
    window,
    len(features)
)

next_temperature = model.predict(
    last_24_hours,
    verbose=0
)


# Convert the prediction to Celsius
next_temperature = (
    next_temperature[0, 0]
    * scaler.data_range_[0]
    + scaler.data_min_[0]
)


# 22. Show final prediction
print("\n================================")
print("Next Time-Step Prediction")
print("================================")

print(
    "Predicted Temperature:",
    round(float(next_temperature), 2),
    "°C"
)

# ============================================================
# End of project
# ============================================================
