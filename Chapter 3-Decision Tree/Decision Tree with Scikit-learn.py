# Decision Tree with Scikit-learn

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    roc_auc_score
)

# --------------------------------------------------
# 1. Load Data
# --------------------------------------------------

data = pd.read_csv("d:\\data.csv")

X = data[['feature1', 'feature2', 'feature3']]
y = data['target']

# --------------------------------------------------
# 2. Train-Test Split
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.20,      # نسبت داده تست

    random_state=42,     # تکرارپذیری نتایج

    stratify=y           # حفظ نسبت کلاس ها
)

# --------------------------------------------------
# 3. Create Model
# --------------------------------------------------

model = DecisionTreeClassifier(

    criterion='gini',        # معيار تقسيم گره
    max_depth=5,             # حداکثر عمق درخت
    min_samples_split=10,    # حداقل نمونه برای تقسيم
    min_samples_leaf=5,      # حداقل نمونه در برگ
    max_features=None,       # تعداد ويژگی های قابل بررسی
    max_leaf_nodes=20,       # حداکثر تعداد برگ
    splitter='best',         # روش انتخاب بهترين تقسيم
    ccp_alpha=0.0,           # هرس درخت
    random_state=42
)

# --------------------------------------------------
# 4. Train
# --------------------------------------------------

model.fit(X_train, y_train)

# --------------------------------------------------
# 5. Predict
# --------------------------------------------------

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:,1]

# --------------------------------------------------
# 6. Evaluate
# --------------------------------------------------

acc = accuracy_score(y_test, y_pred)

auc = roc_auc_score(y_test, y_prob)

print("Decision Tree Results")
print("---------------------")

print(f"Accuracy : {acc:.4f}")

print(f"AUC-ROC  : {auc:.4f}")

print("\nClassification Report")

print(classification_report(y_test, y_pred))

# --------------------------------------------------
# 7. Tree Information
# --------------------------------------------------

print("\nTree Depth =", model.get_depth())

print("Number of Leaves =", model.get_n_leaves())