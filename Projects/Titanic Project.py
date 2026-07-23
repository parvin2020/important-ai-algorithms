# ============================================================================
# پروژه ۶: پیش‌بینی بقای مسافران تایتانیک با الگوریتم Decision Tree
# دیتاست: Titanic (از طریق کتابخانه Seaborn)
# ============================================================================

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, confusion_matrix, classification_report)

# =============================================================================
# گام ۱: بارگذاری و پیش‌پردازش داده‌ها
# =============================================================================
# بارگذاری دیتاست
df = sns.load_dataset('titanic')

# حذف ستون‌های غیرضروری و دارای نویز زیاد
df = df.drop(['deck', 'embark_town', 'class', 'alive', 'who', 'adult_male'], axis=1)

# مدیریت مقادیر گمشده (Imputation)
df['age'].fillna(df['age'].median(), inplace=True)
df['embarked'].fillna(df['embarked'].mode()[0], inplace=True)

# تبدیل متغیرهای دسته‌ای به عددی (Encoding)
df['sex'] = LabelEncoder().fit_transform(df['sex'])
df['embarked'] = LabelEncoder().fit_transform(df['embarked'])

# جدا کردن ویژگی‌ها و متغیر هدف
X = df.drop('survived', axis=1)
y = df['survived']

print("=" * 60)
print("Project 6: Titanic Survival Prediction with Decision Tree")
print("=" * 60)
print(f"Total samples: {X.shape[0]} | Features: {X.shape[1]}")
print(f"Survived: {sum(y==1)} | Died: {sum(y==0)}")

# =============================================================================
# گام : تقسیم داده‌ها
# =============================================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTrain set: {X_train.shape[0]} samples")
print(f"Test set:  {X_test.shape[0]} samples")

# =============================================================================
# گام ۳: آموزش مدل Decision Tree
# =============================================================================
# کنترل Overfitting با محدود کردن عمق درخت
tree_model = DecisionTreeClassifier(
    max_depth=3, min_samples_split=10, criterion='gini', random_state=42
)
tree_model.fit(X_train, y_train)

print("\nModel trained successfully.")

# =============================================================================
# گام ۴: پیش‌بینی و ارزیابی استاندارد
# =============================================================================
y_pred = tree_model.predict(X_test)

print("\n" + "=" * 60)
print("Model Evaluation Metrics")
print("=" * 60)

accuracy  = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall    = recall_score(y_test, y_pred)
f1        = f1_score(y_test, y_pred)

print(f"\n{'Metric':<20} {'Value':>8}")
print("-" * 30)
print(f"{'Accuracy':<20} {accuracy:>8.4f}")
print(f"{'Precision':<20} {precision:>8.4f}")
print(f"{'Recall':<20} {recall:>8.4f}")
print(f"{'F1-Score':<20} {f1:>8.4f}")

# ماتریس درهم‌ریختگی
cm = confusion_matrix(y_test, y_pred)
print(f"\nConfusion Matrix:")
print(f"  TN={cm[0,0]}  FP={cm[0,1]}")
print(f"  FN={cm[1,0]}  TP={cm[1,1]}")

# گزارش کامل طبقه‌بندی
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Died', 'Survived']))

# نمایش اهمیت ویژگی‌ها به صورت مرتب‌شده
print("Feature Importance:")
for feature, importance in sorted(
    zip(X.columns, tree_model.feature_importances_), 
    key=lambda x: x[1], reverse=True
):
    print(f"  {feature:<15}: {importance:.4f}")