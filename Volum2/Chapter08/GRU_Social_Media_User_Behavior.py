# ============================================================
# پروژه عملی: تحلیل رفتار کاربران شبکه اجتماعی با استفاده از GRU
# هدف: پیش‌بینی میزان تعامل کاربران در روز بعد
# ============================================================

import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense


# 1) تولید داده نمونه شبیه رفتار کاربران شبکه اجتماعی
np.random.seed(42)
days = np.arange(200)

engagement = (
    100
    + 20 * np.sin(days / 8)
    + np.random.normal(0, 5, 200)
)


# 2) نمایش رفتار کاربران در طول زمان
plt.plot(engagement)
plt.xlabel("Day")
plt.ylabel("User Engagement")
plt.title("Social Media User Engagement")
plt.show()


# 3) نرمال‌سازی داده‌ها برای آموزش بهتر GRU
scaler = MinMaxScaler()
engagement_scaled = scaler.fit_transform(
    engagement.reshape(-1, 1)
)


# 4) ساخت پنجره زمانی
# هفت روز گذشته ورودی و روز بعد خروجی مدل است.
window = 7

X = []
y = []

for i in range(window, len(engagement_scaled)):
    X.append(engagement_scaled[i-window:i])
    y.append(engagement_scaled[i])

X = np.array(X)
y = np.array(y)


# 5) تقسیم داده به آموزش و آزمون
split = int(len(X) * 0.8)

X_train = X[:split]
y_train = y[:split]

X_test = X[split:]
y_test = y[split:]


# 6) ساخت مدل GRU
model = Sequential()

# GRU الگوهای زمانی رفتار کاربران را یاد می‌گیرد.
model.add(
    GRU(32, input_shape=(window, 1))
)

# خروجی مدل میزان تعامل روز بعد است.
model.add(Dense(1))


# 7) آماده‌سازی مدل
model.compile(
    optimizer="adam",
    loss="mse"
)


# 8) آموزش مدل
print("Training started...")

history = model.fit(
    X_train,
    y_train,
    epochs=30,
    batch_size=16,
    validation_data=(X_test, y_test),
    verbose=1
)


# 9) پیش‌بینی داده‌های آزمون
y_pred = model.predict(X_test, verbose=0)


# 10) بازگرداندن داده‌ها به مقیاس اصلی
y_test_real = scaler.inverse_transform(y_test)
y_pred_real = scaler.inverse_transform(y_pred)


# 11) محاسبه خطای مدل
mae = mean_absolute_error(y_test_real, y_pred_real)

rmse = np.sqrt(
    mean_squared_error(y_test_real, y_pred_real)
)

print("\n===== Model Evaluation =====")
print("MAE  :", round(mae, 2))
print("RMSE :", round(rmse, 2))


# 12) مقایسه رفتار واقعی و پیش‌بینی‌شده
plt.plot(
    y_test_real,
    label="Real Engagement"
)

plt.plot(
    y_pred_real,
    label="Predicted Engagement"
)

plt.xlabel("Day")
plt.ylabel("User Engagement")
plt.title("Real vs Predicted User Engagement")
plt.legend()
plt.show()


# 13) نمایش روند کاهش خطا هنگام آموزش
plt.plot(history.history["loss"])

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("GRU Training Loss")
plt.show()


# 14) پیش‌بینی تعامل کاربران در روز آینده
last_week = engagement_scaled[-window:]

last_week = last_week.reshape(
    1,
    window,
    1
)

next_day = model.predict(
    last_week,
    verbose=0
)

next_day = scaler.inverse_transform(next_day)

print(
    "\nPredicted user engagement for next day:",
    round(float(next_day[0][0]), 2)
)
