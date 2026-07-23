# [CatBoost Algorithm] with Scikit-learn API
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

# 1. Load data set or Create data frame
# ایجاد یک دیتاست نمونه برای شبیه‌سازی مساله عددی
data = {
    'Color': ['Red', 'Blue', 'Red', 'Blue', 'Red', 'Blue', 'Red', 'Blue', 'Red', 'Blue'],
    'Size': [10, 20, 15, 25, 12, 22, 18, 28, 11, 21],
    'Target': [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
}
df = pd.DataFrame(data)
X = df[['Color', 'Size']]
y = df['Target']

# 2. Train-test split
# تقسیم داده‌ها به بخش آموزش و آزمون با حفظ ترتیب برای جلوگیری از نشت داده
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)

# 3. Train model
# تعریف و آموزش مدل CatBoost با تنظیم دقیق هایپرپارامترها
model = CatBoostClassifier(
    iterations=100,          # تعداد کل درخت‌ها یا تکرارهای بوستینگ
    learning_rate=0.1,       # نرخ یادگیری برای کنترل سرعت همگرایی
    depth=4,                 # حداکثر عمق درختان متقارن
    cat_features=['Color'],  # معرفی ستون‌های دسته‌ای برای کدگذاری خودکار
    l2_leaf_reg=3.0,         # ضریب تنظیم‌کننده L2 برای جلوگیری از بیش‌برازش
    random_seed=42,          # دانه تصادفی برای تکرارپذیری نتایج
    verbose=0                # عدم نمایش لاگ‌های حین آموزش برای تمیزی خروجی
)
model.fit(X_train, y_train)

# 4. Predict
# پیش‌بینی کلاس‌ها و احتمال‌های متناظر برای داده‌های آزمون
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

# 5. Evaluate with suitable items
# ارزیابی مدل با استفاده از شاخص‌های دقت و سطح زیر منحنی ROC
acc = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_proba)
print("Model Evaluation Results:")
print(f"Accuracy: {acc:.4f}")
print(f"AUC-ROC: {auc:.4f}")
print("Classification Report:")
print(classification_report(y_test, y_pred, zero_division=0))