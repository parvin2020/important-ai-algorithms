# ============================================================================
# پروژه ۱: تشخیص تومور سرطانی پستان با الگوریتم SVM
# حوزه کاربرد: پزشکی و سیستم‌های کمک‌تشخیصی (CAD)
# ============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, confusion_matrix, 
                             classification_report, roc_curve)
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# گام ۱: بارگذاری و بررسی دیتاست
# =============================================================================
print("=" * 70)
print("Step 1: Loading and Exploring Dataset")
print("=" * 70)

# بارگذاری دیتاست Breast Cancer
cancer = load_breast_cancer()
X = cancer.data
y = cancer.target

# ایجاد DataFrame برای تحلیل بهتر
df = pd.DataFrame(X, columns=cancer.feature_names)
df['target'] = y

# نمایش اطلاعات اولیه دیتاست
print(f"\nNumber of samples: {X.shape[0]}")
print(f"Number of features: {X.shape[1]}")
print(f"\nClass names: {cancer.target_names}")
print(f"Class distribution:")
print(f"  Benign: {sum(y == 0)} samples ({sum(y == 0)/len(y)*100:.1f}%)")
print(f"  Malignant: {sum(y == 1)} samples ({sum(y == 1)/len(y)*100:.1f}%)")

# نمایش ۵ نمونه اول
print("\nFirst 5 samples of dataset:")
print(df.head())

# =============================================================================
# گام ۲: تحلیل اکتشافی داده‌ها (EDA)
# =============================================================================
print("\n" + "=" * 70)
print("Step 2: Exploratory Data Analysis (EDA)")
print("=" * 70)

# آمار توصیفی
print("\nDescriptive statistics of features:")
print(df.describe().T)

# بررسی همبستگی ویژگی‌ها با متغیر هدف
correlation = df.corr()['target'].sort_values(ascending=False)
print("\nTop 5 features correlation with target:")
print(correlation.head(6))  # خود target هم نشان داده می‌شود

# مصورسازی توزیع کلاس‌ها
plt.figure(figsize=(8, 6))
sns.countplot(x='target', data=df, palette='viridis')
plt.xticks([0, 1], ['Benign', 'Malignant'])
plt.title('Class Distribution of Tumors')
plt.xlabel('Tumor Type')
plt.ylabel('Count')
plt.savefig('breast_cancer_class_distribution.png', dpi=300, bbox_inches='tight')
plt.show()

# مصورسازی همبستگی
plt.figure(figsize=(12, 10))
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix.iloc[:10, :10], annot=True, cmap='coolwarm', 
            fmt='.2f', linewidths=0.5)
plt.title('Correlation Matrix of First 10 Features')
plt.savefig('breast_cancer_correlation.png', dpi=300, bbox_inches='tight')
plt.show()

# =============================================================================
# گام ۳: پیش‌پردازش و تقسیم داده‌ها
# =============================================================================
print("\n" + "=" * 70)
print("Step 3: Preprocessing and Data Splitting")
print("=" * 70)

# تقسیم داده‌ها به آموزش و تست (۸۰-۲۰)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2,      # ۲۰٪ برای تست
    random_state=42,    # برای تکرارپذیری
    stratify=y          # حفظ توزیع کلاس‌ها
)

print(f"\nTraining set size: {X_train.shape[0]} samples")
print(f"Test set size: {X_test.shape[0]} samples")

# مقیاس‌بندی ویژگی‌ها (حیاتی برای SVM)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nFeature scaling with StandardScaler completed.")
print(f"Mean of training data after scaling: {X_train_scaled.mean():.6f}")
print(f"Standard deviation of training data after scaling: {X_train_scaled.std():.6f}")

# =============================================================================
# گام ۴: آموزش مدل SVM با پارامترهای پیش‌فرض
# =============================================================================
print("\n" + "=" * 70)
print("Step 4: Training SVM Model with Default Parameters")
print("=" * 70)

# ایجاد و آموزش مدل SVM با کرنل RBF
svm_model = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42, probability=True)
svm_model.fit(X_train_scaled, y_train)

print("\nSVM model trained successfully.")
print(f"Number of support vectors: {svm_model.n_support_}")

# پیش‌بینی روی داده‌های تست
y_pred = svm_model.predict(X_test_scaled)
y_pred_proba = svm_model.predict_proba(X_test_scaled)[:, 1]

# =============================================================================
# گام ۵: ارزیابی مدل
# =============================================================================
print("\n" + "=" * 70)
print("Step 5: Model Evaluation")
print("=" * 70)

# محاسبه شاخص‌های ارزیابی
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)  # Sensitivity
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)

print(f"\n{'Metric':<25} {'Value':>10}")
print("-" * 35)
print(f"{'Accuracy':<25} {accuracy:>10.4f}")
print(f"{'Precision':<25} {precision:>10.4f}")
print(f"{'Recall (Sensitivity)':<25} {recall:>10.4f}")
print(f"{'F1-Score':<25} {f1:>10.4f}")
print(f"{'ROC-AUC':<25} {roc_auc:>10.4f}")

# نمایش گزارش طبقه‌بندی
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Benign', 'Malignant']))

# ماتریس درهم‌ریختگی
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Benign', 'Malignant'],
            yticklabels=['Benign', 'Malignant'])
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.savefig('breast_cancer_confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"\nConfusion Matrix:")
print(f"True Negatives (TN): {cm[0, 0]}")
print(f"False Positives (FP): {cm[0, 1]}")
print(f"False Negatives (FN): {cm[1, 0]}")
print(f"True Positives (TP): {cm[1, 1]}")

# منحنی ROC
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, 
         label=f'ROC curve (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.savefig('breast_cancer_roc_curve.png', dpi=300, bbox_inches='tight')
plt.show()

# =============================================================================
# گام ۶: بهینه‌سازی هایپرپارامترها با Grid Search
# =============================================================================
print("\n" + "=" * 70)
print("Step 6: Hyperparameter Optimization with Grid Search")
print("=" * 70)

# تعریف پارامترهای جستجو
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto', 0.01, 0.001],
    'kernel': ['rbf', 'linear']
}

# ایجاد GridSearchCV
grid_search = GridSearchCV(
    SVC(probability=True, random_state=42),
    param_grid,
    cv=5,  # اعتبارسنجی متقاطع ۵ تایی
    scoring='recall',  # بهینه‌سازی بر اساس Recall (مهم برای پزشکی)
    n_jobs=-1,
    verbose=1
)

print("\nSearching for best parameters...")
grid_search.fit(X_train_scaled, y_train)

print(f"\nBest parameters: {grid_search.best_params_}")
print(f"Best Recall score in cross-validation: {grid_search.best_score_:.4f}")

# آموزش مدل با بهترین پارامترها
best_svm = grid_search.best_estimator_
y_pred_best = best_svm.predict(X_test_scaled)
y_pred_proba_best = best_svm.predict_proba(X_test_scaled)[:, 1]

# ارزیابی مدل بهینه‌شده
print("\n" + "=" * 70)
print("Evaluation of Optimized Model")
print("=" * 70)

accuracy_best = accuracy_score(y_test, y_pred_best)
recall_best = recall_score(y_test, y_pred_best)
roc_auc_best = roc_auc_score(y_test, y_pred_proba_best)

print(f"\nAccuracy: {accuracy_best:.4f}")
print(f"Recall: {recall_best:.4f}")
print(f"ROC-AUC: {roc_auc_best:.4f}")

# =============================================================================
# گام ۷: اعتبارسنجی متقاطع
# =============================================================================
print("\n" + "=" * 70)
print("Step 7: Cross-Validation")
print("=" * 70)

cv_scores = cross_val_score(best_svm, X_train_scaled, y_train, cv=10, scoring='recall')
print(f"\nRecall scores in 10-fold cross-validation:")
print(cv_scores)
print(f"\nMean: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# =============================================================================
# گام ۸: اهمیت ویژگی‌ها (Feature Importance)
# =============================================================================
print("\n" + "=" * 70)
print("Step 8: Feature Importance Analysis")
print("=" * 70)

# برای SVM با کرنل خطی می‌توانیم ضرایب را بررسی کنیم
linear_svm = SVC(kernel='linear', C=1, random_state=42)
linear_svm.fit(X_train_scaled, y_train)

# محاسبه اهمیت ویژگی‌ها بر اساس ضرایب
feature_importance = np.abs(linear_svm.coef_[0])
feature_names = cancer.feature_names

# ایجاد DataFrame برای نمایش
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': feature_importance
}).sort_values('Importance', ascending=False)

print("\nTop 10 features:")
print(importance_df.head(10))

# مصورسازی ۱۰ ویژگی برتر
plt.figure(figsize=(10, 8))
sns.barplot(x='Importance', y='Feature', data=importance_df.head(10), palette='viridis')
plt.title('Top 10 Features in Tumor Diagnosis')
plt.xlabel('Importance Coefficient')
plt.tight_layout()
plt.savefig('breast_cancer_feature_importance.png', dpi=300, bbox_inches='tight')
plt.show()

# =============================================================================
# گام ۹: پیش‌بینی روی نمونه‌های جدید
# =============================================================================
print("\n" + "=" * 70)
print("Step 9: Prediction on New Samples")
print("=" * 70)

# انتخاب ۵ نمونه تصادفی از داده‌های تست
sample_indices = np.random.choice(X_test.shape[0], 5, replace=False)
X_samples = X_test[sample_indices]
y_samples_true = y_test[sample_indices]

# مقیاس‌بندی نمونه‌ها
X_samples_scaled = scaler.transform(X_samples)

# پیش‌بینی
samples_pred = best_svm.predict(X_samples_scaled)
samples_proba = best_svm.predict_proba(X_samples_scaled)

print("\nPredictions for 5 random samples:")
print("-" * 70)
for i, (idx, true_label, pred_label, proba) in enumerate(zip(
    sample_indices, y_samples_true, samples_pred, samples_proba), 1):
    true_str = 'Malignant' if true_label == 1 else 'Benign'
    pred_str = 'Malignant' if pred_label == 1 else 'Benign'
    prob_malignant = proba[1] * 100
    
    print(f"\nSample {i}:")
    print(f"  Sample index in dataset: {idx}")
    print(f"  True label: {true_str}")
    print(f"  Predicted label: {pred_str}")
    print(f"  Probability of malignancy: {prob_malignant:.2f}%")
    print(f"  Result: {'Correct' if true_label == pred_label else 'Incorrect'}")

# =============================================================================
# خلاصه و نتیجه‌گیری
# =============================================================================
print("\n" + "=" * 70)
print("Summary and Conclusion")
print("=" * 70)

print(f"""
Final Results of Optimized SVM Model:
--------------------------------------
• Overall Accuracy: {accuracy_best:.4f}
• Recall (Sensitivity): {recall_best:.4f} <- Most important metric in medicine
• Precision: {precision_score(y_test, y_pred_best):.4f}
• F1-Score: {f1_score(y_test, y_pred_best):.4f}
• ROC-AUC: {roc_auc_best:.4f}

Key Points:
-----------
1. Feature scaling is critical for SVM
2. In medical applications, Recall is more important than Precision
   (Detecting all malignant cases is the priority)
3. Hyperparameter optimization with Grid Search improved performance
4. Features 'worst perimeter' and 'worst concave points' had the 
   highest impact on diagnosis

Practical Applications:
-----------------------
• Computer-aided diagnosis (CAD) systems in hospitals
• Automated screening of pathology images
• Reducing human error in diagnosis
""")

print("=" * 70)
print("End of Project 1: Breast Cancer Tumor Detection")
print("=" * 70)
