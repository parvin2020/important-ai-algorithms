# ============================================================================
# پروژه ۱: تشخیص تومور سرطانی پستان با الگوریتم SVM
# دیتاست: Breast Cancer Wisconsin (Diagnostic)
# ============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, confusion_matrix, 
                             classification_report)

# =============================================================================
# گام ۱: بارگذاری دیتاست
# =============================================================================
cancer = load_breast_cancer()
X = cancer.data
y = cancer.target

print("=" * 60)
print("Project 1: Breast Cancer Detection with SVM")
print("=" * 60)
print(f"Number of samples: {X.shape[0]}")
print(f"Number of features: {X.shape[1]}")
print(f"Class distribution - Benign: {sum(y==0)}, Malignant: {sum(y==1)}")

# =============================================================================
# گام : تقسیم داده‌ها و مقیاس‌بندی
# =============================================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# مقیاس‌بندی ویژگی‌ها (حیاتی برای SVM)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\nTrain set: {X_train.shape[0]} samples")
print(f"Test set:  {X_test.shape[0]} samples")

# =============================================================================
# گام ۳: آموزش مدل SVM
# =============================================================================
# پارامترهای کلیدی:
# kernel='rbf'     : کرنل شعاعی برای مدل‌سازی روابط غیرخطی
# C=1.0            : پارامتر منظم‌سازی (کنترل trade-off بین حاشیه و خطا)
# gamma='scale'    : ضریب تأثیر هر نمونه (پیش‌فرض Scikit-learn)
# probability=True : برای محاسبه ROC-AUC الزامی است
svm_model = SVC(kernel='rbf', C=1.0, gamma='scale', 
                random_state=42, probability=True)
svm_model.fit(X_train_scaled, y_train)

print("\nModel trained successfully.")
print(f"Support vectors: {svm_model.n_support_}")

# =============================================================================
# گام ۴: پیش‌بینی
# =============================================================================
y_pred = svm_model.predict(X_test_scaled)
y_pred_proba = svm_model.predict_proba(X_test_scaled)[:, 1]

# =============================================================================
# گام ۵: ارزیابی مدل با شاخص‌های کلیدی
# =============================================================================
print("\n" + "=" * 60)
print("Model Evaluation Metrics")
print("=" * 60)

accuracy  = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall    = recall_score(y_test, y_pred)      # Sensitivity
f1        = f1_score(y_test, y_pred)
roc_auc   = roc_auc_score(y_test, y_pred_proba)

print(f"\n{'Metric':<20} {'Value':>8}")
print("-" * 30)
print(f"{'Accuracy':<20} {accuracy:>8.4f}")
print(f"{'Precision':<20} {precision:>8.4f}")
print(f"{'Recall (Sens.)':<20} {recall:>8.4f}  <- Key metric")
print(f"{'F1-Score':<20} {f1:>8.4f}")
print(f"{'ROC-AUC':<20} {roc_auc:>8.4f}")

# گزارش طبقه‌بندی
print("\nClassification Report:")
print(classification_report(y_test, y_pred, 
                            target_names=['Benign', 'Malignant']))

# ماتریس درهم‌ریختگی
cm = confusion_matrix(y_test, y_pred)
print(f"\nConfusion Matrix:")
print(f"  TN={cm[0,0]}  FP={cm[0,1]}")
print(f"  FN={cm[1,0]}  TP={cm[1,1]}")

# =============================================================================
# گام ۶: مصورسازی ماتریس درهم‌ریختگی
# =============================================================================
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Benign', 'Malignant'],
            yticklabels=['Benign', 'Malignant'])
plt.title('Confusion Matrix - Breast Cancer SVM')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('breast_cancer_cm.png', dpi=300)
plt.show()

print("\n" + "=" * 60)
print("Key Insight: In medical diagnosis, Recall is the most")
print("important metric because False Negative (missing a")
print("malignant tumor) can be life-threatening.")
print("=" * 60)