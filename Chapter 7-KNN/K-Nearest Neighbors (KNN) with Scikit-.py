# K-Nearest Neighbors (KNN) with Scikit-learn
import numpy as np
import pandas as pd
# کتابخانه‌های مورد نیاز
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
# شاخص‌های ارزیابی
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    precision_score,
    recall_score,
    f1_score
)

# ----------------------------------------------------
# 1. Create data frame
# ----------------------------------------------------
data = {
    "Age":    [25, 30, 28, 40, 45, 50],
    "Income": [30, 35, 40, 60, 65, 70],
    "Class":  ["A", "A", "A", "B", "B", "B"]
}
df = pd.DataFrame(data)
X = df[["Age", "Income"]]
y = df["Class"]

# ----------------------------------------------------
# 2. Train-test split
# ----------------------------------------------------
# تقسیم داده‌ها به آموزش و آزمون
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.33,          # 33 درصد داده‌ها برای آزمون
    random_state=42,          # جهت تکرارپذیری نتایج
    stratify=y                # حفظ نسبت کلاس‌ها
)

# ----------------------------------------------------
# 3. Feature Scaling
# ----------------------------------------------------
# استانداردسازی داده‌ها
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ----------------------------------------------------
# 4. Train model
# ----------------------------------------------------
knn = KNeighborsClassifier(
    n_neighbors=3,          # تعداد نزدیک‌ترین همسایه‌ها
    weights='uniform',      # همه همسایه‌ها وزن یکسان دارند
    # اگر distance انتخاب شود، همسایه‌های نزدیک‌تر وزن بیشتری می‌گیرند
    algorithm='auto',       # انتخاب خودکار بهترین الگوریتم جستجو
    leaf_size=30,           # اندازه برگ برای KDTree و BallTree
    metric='minkowski',     # معیار محاسبه فاصله
    p=2,                    # p=2 یعنی فاصله اقلیدسی
    n_jobs=-1               # استفاده از تمام هسته‌های پردازنده
)

knn.fit(X_train_scaled, y_train)

# ----------------------------------------------------
# 5. Predict
# ----------------------------------------------------

y_pred = knn.predict(X_test_scaled)
# پیش‌بینی مشتری جدید
new_customer = np.array([[35, 45]])
new_customer_scaled = scaler.transform(new_customer)
prediction = knn.predict(new_customer_scaled)

# ----------------------------------------------------
# 6. Evaluate
# ----------------------------------------------------
acc = accuracy_score(y_test, y_pred)
pre = precision_score(
    y_test,
    y_pred,
    pos_label="A"
)

rec = recall_score(
    y_test,
    y_pred,
    pos_label="A"
)

f1 = f1_score(
    y_test,
    y_pred,
    pos_label="A"
)

cm = confusion_matrix(y_test, y_pred)
print("Accuracy :", acc)
print("Precision:", pre)
print("Recall   :", rec)
print("F1 Score :", f1)
print("\nConfusion Matrix")
print(cm)
print("\nClassification Report")
print(classification_report(y_test, y_pred))
print("\nPrediction for New Customer:")
print(prediction[0])