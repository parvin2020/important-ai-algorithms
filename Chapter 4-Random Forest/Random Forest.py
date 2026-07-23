import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
# داده‌ها
data = {
    'study_hours': [5, 12, 8, 3, 15, 7, 10, 9, 11, 6],
    'attendance': [60, 85, 75, 40, 90, 65, 80, 70, 88, 55],
    'passed': [0, 1, 1, 0, 1, 0, 1, 1, 1, 0]
}
df = pd.DataFrame(data)
X = df[['study_hours', 'attendance']]
y = df['passed']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
# مدل جنگل تصادفی
model = RandomForestClassifier(
    n_estimators=200, 
    max_depth=6,
    max_features='sqrt',
    random_state=42,
    oob_score=True  # استفاده از Out-of-Bag
)
model.fit(X_train, y_train)
# پیش‌بینی
pred = model.predict([[10, 80]])
print(f"Prediction for  (10 Hours 80% Attendance): {'Accept' if pred[0] == 1 else 'Decline'}")
print(f"Accuracy of Model: {accuracy_score(y_test, model.predict(X_test)):.3f}")
print(f"OOB Score: {model.oob_score_:.3f}")
# اهمیت ویژگی‌ها
importances = pd.Series(model.feature_importances_, index=X.columns)
print("\nImportance of Features:")
print(importances.sort_values(ascending=False))