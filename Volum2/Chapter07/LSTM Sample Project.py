# ==========================================================
# Mini Project
# Energy Consumption Forecasting using LSTM
# Author : Dr. Bahram Parvin
# ==========================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# ==========================================================
# 1- تولید دیتاست آموزشی
# ==========================================================
# برای اینکه در هر بار اجرای برنامه، داده‌های تصادفی یکسان تولید شوند.
np.random.seed(42)

# ایجاد شماره روزها از 0 تا 364
days = np.arange(365)

# تولید داده‌های مصنوعی مصرف انرژی
# شامل سه بخش:
# 1- مقدار پایه مصرف (450)
# 2- روند افزایشی تدریجی مصرف
# 3- نوسان فصلی به همراه نویز تصادفی

consumption = (
    450
    + 0.25 * days
    + 30 * np.sin(2 * np.pi * days / 30)
    + np.random.normal(0, 5, len(days))
)

# ایجاد DataFrame شامل شماره روز و میزان مصرف انرژی
data = pd.DataFrame({
    "Day": days + 1,
    "Consumption": consumption
})
# ==========================================================
# 2- نمایش چند رکورد اول
# ==========================================================
print(data.head())

# ==========================================================
# 3- رسم داده اولیه
# ==========================================================
plt.figure(figsize=(12,4))
plt.plot(data["Consumption"])
plt.title("Daily Energy Consumption")
plt.xlabel("Day")
plt.ylabel("MWh")
plt.grid(True)
plt.show()

# ==========================================================
# 4- نرمال سازی
# ==========================================================
scaler = MinMaxScaler()
values = scaler.fit_transform(
    data["Consumption"].values.reshape(-1,1)
)

# ==========================================================
# 5- ساخت داده های زمانی
# ==========================================================
window = 7
X = []
y = []
for i in range(len(values)-window):
    X.append(values[i:i+window])
    y.append(values[i+window])
X = np.array(X)
y = np.array(y)

# ==========================================================
# 6- تقسیم داده
# ==========================================================
train_size = int(len(X)*0.80)
X_train = X[:train_size]
X_test = X[train_size:]
y_train = y[:train_size]
y_test = y[train_size:]

# ==========================================================
# 7- ساخت مدل
# ==========================================================
model = Sequential()
model.add(LSTM(units=32, input_shape=(window,1)))
model.add(Dense(1))
model.compile(optimizer="adam", loss="mse")

# ==========================================================
# 8- آموزش مدل
# ==========================================================
history = model.fit(
    X_train,
    y_train,
    epochs=30,
    batch_size=16,
    validation_data=(X_test,y_test),
    verbose=1
)

# ==========================================================
# 9- پیش بینی
# ==========================================================
prediction = model.predict(X_test)
prediction = scaler.inverse_transform(prediction)
actual = scaler.inverse_transform(y_test)
actual = actual.flatten()
prediction = prediction.flatten()

# ==========================================================
# 10- رسم نمودار
# ==========================================================
plt.figure(figsize=(12,5))
plt.plot(
    actual,
    color="blue",
    linewidth=2,
    label="Actual"
)

plt.plot(
    prediction,
    "--",
    color="red",
    linewidth=2,
    label="Prediction"
)
plt.title("Energy Consumption Forecast using LSTM")
plt.xlabel("Test Samples")
plt.ylabel("Consumption (MWh)")
plt.grid(True)
plt.legend()
plt.show()

# ==========================================================
# 11- پیش بینی روز آینده
# ==========================================================
last_week = values[-window:]
last_week = last_week.reshape(1,window,1)
next_day = model.predict(last_week)
next_day = scaler.inverse_transform(next_day)
print()
print("=======================================")
print("Predicted Consumption for Next Day")
print(round(next_day[0][0],2),"MWh")
print("=======================================")