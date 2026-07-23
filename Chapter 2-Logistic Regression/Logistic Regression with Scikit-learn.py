# Logistic Regression with Scikit-learn

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, \
    classification_report, roc_auc_score

# 1. Load data
data = pd.read_csv('d:\\data.csv')

X = data[['feature1', 'feature2', 'feature3']]
y = data['target']   # 0 or 1

# 2. Train-test split
X_train, X_test, y_train, y_test = \
    train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

# 3. Train model
model = LogisticRegression()

model.fit(X_train, y_train)

# 4. Predict
y_pred = model.predict(X_test)

y_prob = model.predict_proba(X_test)[:, 1]  # Prob of class 1

# 5. Evaluate
acc = accuracy_score(y_test, y_pred)

auc = roc_auc_score(y_test, y_prob)

print(f"Accuracy: {acc:.4f}")
print(f"AUC-ROC: {auc:.4f}")

print(classification_report(y_test, y_pred))