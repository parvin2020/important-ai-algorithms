import numpy as np
from sklearn.neighbors import KNeighborsClassifier

# داده‌های آموزشی
X = np.array([
    [2,20,15],
    [3,25,20],
    [10,80,60],
    [12,90,65],
    [5,40,30],
    [9,70,55]
])

# برچسب‌ها
y = np.array([
    "Normal",
    "Normal",
    "Special",
    "Special",
    "Normal",
    "Special"
])

# ساخت مدل
model = KNeighborsClassifier(n_neighbors=3)

# آموزش
model.fit(X, y)

# مشتری جدید
new_customer = np.array([[8,65,50]])

# پیش‌بینی
prediction = model.predict(new_customer)

print(" New Customer Class:")
print(prediction[0])
