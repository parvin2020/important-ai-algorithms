# Isolation Forest with Scikit-learn

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# ---------------------------------------------------------
# 1. Create data frame
# ---------------------------------------------------------

# ایجاد مجموعه داده

temperature = [29,30,31,31,32,32,33,60]
# 0 = Normal
# 1 = Anomaly
true_label = [0,0,0,0,0,0,0,1]

df = pd.DataFrame({
    "Temperature": temperature,
    "TrueLabel": true_label
})
print("Original Data")
print(df)

# ---------------------------------------------------------
# 2. Train-test split
# ---------------------------------------------------------
# تقسیم داده ها
X = df[["Temperature"]]
y = df["TrueLabel"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42
)

# ---------------------------------------------------------
# 3. Train model
# ---------------------------------------------------------

# ایجاد مدل Isolation Forest

model = IsolationForest(
    n_estimators=200,          # تعداد درخت ها
    max_samples="auto",        # تعداد نمونه هر درخت
    contamination=0.125,       # نسبت داده پرت
    max_features=1.0,          # درصد ویژگی ها
    bootstrap=False,           # نمونه گیری بدون جایگذاری
    random_state=42,           # تکرارپذیری نتایج
    n_jobs=-1,                 # استفاده از تمام هسته های CPU
    warm_start=False           # آموزش مجدد کامل
)
model.fit(X_train)

# ---------------------------------------------------------
# 4. Predict
# ---------------------------------------------------------
# پیش بینی
prediction = model.predict(X_test)

# تبدیل خروجی sklearn
# Normal = 1
# Anomaly = -1
prediction = np.where(prediction == -1, 1, 0)

# امتیاز ناهنجاری
scores = model.decision_function(X_test)
print("\nDecision Scores")
print(scores)
print("\nPrediction")
print(prediction)

# ---------------------------------------------------------
# 5. Evaluate
# ---------------------------------------------------------
accuracy = accuracy_score(y_test, prediction)
precision = precision_score(
    y_test,
    prediction,
    zero_division=0
)

recall = recall_score(
    y_test,
    prediction,
    zero_division=0
)

f1 = f1_score(
    y_test,
    prediction,
    zero_division=0
)
cm = confusion_matrix(y_test, prediction)
print("\nAccuracy =", accuracy)
print("Precision =", precision)
print("Recall =", recall)
print("F1 Score =", f1)
print("\nConfusion Matrix")
print(cm)
print("\nClassification Report")
print(classification_report( y_test, prediction, zero_division=0))