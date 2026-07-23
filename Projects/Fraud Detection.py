# ============================================================================
# پروژه ۳: تشخیص تقلب در تراکنش‌های کارت اعتباری با Isolation Forest
# دیتاست: Credit Card Fraud Detection (شبیه‌سازی شده برای اجرای سریع)
# ============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (precision_score, recall_score, f1_score, 
                             average_precision_score, roc_auc_score, 
                             confusion_matrix, classification_report)

# =============================================================================
# گام ۱: ایجاد دیتاست شبیه‌سازی شده (جایگزین فایل حجیم Kaggle)
# =============================================================================
# تولید ۵۰,۰۰۰ نمونه با نرخ تقلب ۰.۱۷٪ (دقیقاً مشابه دیتاست واقعی)
X, y = make_classification(
    n_samples=50000, n_features=28, n_informative=20, 
    weights=[0.9983, 0.0017], random_state=42, flip_y=0
)

print("=" * 60)
print("Project 3: Credit Card Fraud Detection with Isolation Forest")
print("=" * 60)
print(f"Total samples: {X.shape[0]}")
print(f"Features: {X.shape[1]}")
print(f"Normal transactions: {sum(y==0)} ({sum(y==0)/len(y)*100:.2f}%)")
print(f"Fraud transactions:  {sum(y==1)} ({sum(y==1)/len(y)*100:.2f}%)")

# =============================================================================
# گام ۲: تقسیم داده‌ها به آموزش و تست
# =============================================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTrain set: {X_train.shape[0]} samples")
print(f"Test set:  {X_test.shape[0]} samples")

# =============================================================================
# گام ۳: آموزش مدل Isolation Forest
# =============================================================================
# پارامترهای کلیدی:
# n_estimators=100 : تعداد درخت‌های ایزوله
# contamination=0.0017 : نسبت مورد انتظار ناهنجاری‌ها در دیتاست (۰.۱۷٪)
# max_samples='auto' : استفاده از حداکثر ۲۵ نمونه برای هر درخت (افزایش سرعت)
iso_forest = IsolationForest(
    n_estimators=100, 
    contamination=0.0017, 
    max_samples='auto', 
    random_state=42, 
    n_jobs=-1
)
iso_forest.fit(X_train)

print("\nModel trained successfully.")

# =============================================================================
# گام ۴: پیش‌بینی و محاسبه امتیازات
# =============================================================================
# متد predict خروجی ۱ (عادی) و -۱ (تقلب) می‌دهد. آن را به ۰ و ۱ تبدیل می‌کنیم
y_pred_raw = iso_forest.predict(X_test)
y_pred = (y_pred_raw == -1).astype(int)  # تبدیل -۱ به ۱ (تقلب) و ۱ به ۰ (عادی)

# متد decision_function امتیاز خام می‌دهد (مقادیر منفی‌تر = احتمال تقلب بیشتر)
# برای منحنی ROC و PR، مقادیر را معکوس می‌کنیم تا امتیاز بالاتر = تقلب بیشتر
y_scores = -iso_forest.decision_function(X_test)

# =============================================================================
# گام : ارزیابی با معیارهای حیاتی برای داده‌های نامتوازن
# =============================================================================
print("\n" + "=" * 60)
print("Model Evaluation Metrics (Crucial for Imbalanced Data)")
print("=" * 60)

precision = precision_score(y_test, y_pred, zero_division=0)
recall    = recall_score(y_test, y_pred, zero_division=0)
f1        = f1_score(y_test, y_pred, zero_division=0)
ap_score  = average_precision_score(y_test, y_scores)
roc_auc   = roc_auc_score(y_test, y_scores)

print(f"\n{'Metric':<25} {'Value':>8}")
print("-" * 35)
print(f"{'Precision':<25} {precision:>8.4f}")
print(f"{'Recall (Sensitivity)':<25} {recall:>8.4f}  <- Key metric")
print(f"{'F1-Score':<25} {f1:>8.4f}")
print(f"{'Average Precision (AP)':<25} {ap_score:>8.4f}  <- Best for imbalance")
print(f"{'ROC-AUC':<25} {roc_auc:>8.4f}")

# ماتریس درهم‌ریختگی
cm = confusion_matrix(y_test, y_pred)
print(f"\nConfusion Matrix:")
print(f"  True Negatives  (TN): {cm[0,0]}")
print(f"  False Positives (FP): {cm[0,1]}")
print(f"  False Negatives (FN): {cm[1,0]}  <- Missed Frauds (Critical!)")
print(f"  True Positives  (TP): {cm[1,1]}")

# =============================================================================
# گام ۶: مصورسازی منحنی Precision-Recall (مهم‌تر از ROC برای داده نامتوازن)
# =============================================================================
from sklearn.metrics import precision_recall_curve

precisions, recalls, thresholds = precision_recall_curve(y_test, y_scores)

plt.figure(figsize=(7, 5))
plt.plot(recalls, precisions, marker='.', label='Isolation Forest')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('credit_card_pr_curve.png', dpi=300)
plt.show()

print("\n" + "=" * 60)
print("Key Insight: Accuracy is useless here (99.83% is trivial).")
print("We focus on Recall (catching frauds) and Average Precision.")
print("=" * 60)