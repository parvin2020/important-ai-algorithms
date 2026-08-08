# ============================================================
# پروژه عملی GRU: پیش‌بینی ریزش مشتری بر اساس رفتار 6 هفته گذشته
# 0 = عدم ریزش | 1 = ریزش
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense

# 1) تولید داده نمونه؛ در پروژه واقعی این داده از پایگاه داده خوانده می‌شود.
np.random.seed(42)
n_customers, weeks, features = 1000, 6, 4
X = np.zeros((n_customers, weeks, features), dtype="float32")
y = np.zeros(n_customers, dtype="int32")

# 2) ساخت دو الگوی رفتاری: مشتری فعال و مشتری در حال ریزش.
for i in range(n_customers):
    if i < n_customers // 2:
        base = np.random.uniform(0.7, 1.0)
        for w in range(weeks):
            X[i, w] = [
                base * 10 + np.random.normal(0, .5),
                base * 4 + np.random.normal(0, .3),
                base * 120 + np.random.normal(0, 5),
                base * 15 + np.random.normal(0, 1)
            ]
        y[i] = 0
    else:
        start = np.random.uniform(.8, 1.0)
        for w in range(weeks):
            activity = max(0, start - w * .12)
            X[i, w] = [
                max(0, activity * 10 + np.random.normal(0, .5)),
                max(0, activity * 4 + np.random.normal(0, .3)),
                max(0, activity * 120 + np.random.normal(0, 5)),
                max(0, activity * 15 + np.random.normal(0, 1))
            ]
        y[i] = 1

# 3) نمایش روند ورود دو مشتری نمونه.
plt.plot(X[10, :, 0], marker="o", label="Active Customer")
plt.plot(X[700, :, 0], marker="o", label="Churning Customer")
plt.xlabel("Week")
plt.ylabel("Number of Logins")
plt.title("Customer Activity During 6 Weeks")
plt.legend()
plt.show()

# 4) تقسیم استاندارد: 80٪ آموزش و 20٪ آزمون؛ stratify نسبت کلاس‌ها را حفظ می‌کند.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=.20, random_state=42, stratify=y
)

# 5) نرمال‌سازی بدون Data Leakage؛ Scaler فقط روی Training fit می‌شود.
scaler = MinMaxScaler()
X_train_2d = X_train.reshape(-1, features)
X_test_2d = X_test.reshape(-1, features)
scaler.fit(X_train_2d)
X_train = scaler.transform(X_train_2d).reshape(X_train.shape)
X_test = scaler.transform(X_test_2d).reshape(X_test.shape)

# 6) ساخت مدل؛ GRU الگوی تغییر رفتار مشتری را در طول زمان یاد می‌گیرد.
model = Sequential([
    GRU(32, input_shape=(weeks, features)),
    Dense(1, activation="sigmoid")
])

# 7) آماده‌سازی برای مسئله طبقه‌بندی دودویی.
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# 8) آموزش مدل.
history = model.fit(
    X_train, y_train, epochs=20, batch_size=32,
    validation_split=.20, verbose=1
)

# 9) ارزیابی روی داده‌هایی که مدل در آموزش ندیده است.
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print("\n==============================")
print("Model Evaluation")
print("==============================")
print("Test Accuracy:", round(float(accuracy), 3))

# 10) تبدیل احتمال به کلاس؛ احتمال 0.5 یا بیشتر یعنی ریزش.
probabilities = model.predict(X_test, verbose=0)
y_pred = (probabilities >= .5).astype(int).ravel()
print("Accuracy:", round(accuracy_score(y_test, y_pred), 3))

# 11) نمایش ماتریس خطا.
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# 12) نمایش روند یادگیری مدل.
plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("GRU Training")
plt.legend()
plt.show()

# 13) تعریف یک مشتری جدید با کاهش تدریجی فعالیت در 6 هفته.
new_customer = np.array([
    [9, 4, 115, 14],
    [8, 3, 100, 12],
    [7, 3, 85, 10],
    [5, 2, 60, 7],
    [3, 1, 35, 4],
    [1, 0, 15, 1]
], dtype="float32")

# 14) استفاده از همان Scaler آموزش برای مشتری جدید.
new_customer_scaled = scaler.transform(new_customer).reshape(1, weeks, features)

# 15) پیش‌بینی احتمال ریزش مشتری جدید.
churn_probability = float(model.predict(new_customer_scaled, verbose=0)[0][0])
print("\n==============================")
print("New Customer Prediction")
print("==============================")
print("Churn Probability:", round(churn_probability, 3))

if churn_probability >= .5:
    print("Prediction: Customer is likely to churn.")
else:
    print("Prediction: Customer is likely to stay.")
