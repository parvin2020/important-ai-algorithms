# Random Forest with Scikit-learn

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    roc_auc_score
)

# 1. Load data
data = pd.read_csv('d:\\data.csv')
X = data[data.columns[:-1]]
y = data[data.columns[-1]]

# 2. Train-test split
X_train, X_test, y_train, y_test = \
    train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
)

# 3. Train model
model = RandomForestClassifier(
    n_estimators=200,      # تعداد درخت ها
    criterion='gini',      # معيار تقسيم
    max_depth=10,          # حداکثر عمق
    min_samples_split=5,   # حداقل نمونه برای تقسيم
    min_samples_leaf=2,    # حداقل نمونه در برگ
    max_features='sqrt',   # تعداد ويژگی تصادفی
    bootstrap=True,        # نمونه برداری بوت استرپ
    oob_score=True,        # محاسبه OOB
    random_state=42,
    n_jobs=-1              # استفاده از همه هسته ها
)
model.fit(X_train, y_train)

# 4. Predict
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:,1] # فقط ستون مربوط به کلاس مثبت (1) را از ماتریس احتمالات کلاس ها جدا می‌کند

# 5. Evaluate
acc = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)
print(f"Accuracy: {acc:.4f}")
print(f"AUC-ROC: {auc:.4f}")
print(f"OOB Score: {model.oob_score_:.4f}")
print(classification_report(y_test, y_pred))