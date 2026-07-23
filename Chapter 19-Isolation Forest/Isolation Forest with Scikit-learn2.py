# Isolation Forest with Scikit-learn
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report

# 1. Load data set or Create data frame
# ایجاد یک دیتاست مصنوعی شامل داده‌های عادی و ناهنجار برای شبیه‌سازی مساله عددی در ابعادی بزرگتر
np.random.seed(42)
X_normal = np.random.normal(loc=0, scale=1, size=(200, 2))
X_anomaly = np.random.uniform(low=-4, high=4, size=(20, 2))
X = np.vstack([X_normal, X_anomaly])
# برچسب‌ها: 1 برای عادی و -1 برای ناهنجار (طبق استاندارد کتابخانه sklearn)
y = np.array([1]*200 + [-1]*20)

# 2. Train-test split
# تقسیم داده‌ها به بخش آموزش و تست برای ارزیابی دقیق‌تر و جلوگیری از بیش‌برازش
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 3. Train model
# آموزش مدل با تنظیم هایپرپارامترهای کلیدی
# n_estimators: تعداد درخت‌های ایزوله (افزایش آن پایداری مدل را بالا می‌برد اما هزینه محاسباتی را زیاد می‌کند)
# contamination: نسبت ناهنجارها در دیتاست (0.1 یعنی فرض بر این است که 10 درصد داده‌ها ناهنجارند)
# max_samples: تعداد نمونه‌ها برای آموزش هر درخت (auto یعنی min(256, n_samples) که سرعت را به شدت افزایش می‌دهد)
# max_features: تعداد ویژگی‌ها برای آموزش هر درخت (1.0 یعنی استفاده از همه ویژگی‌ها در هر گره)
# bootstrap: استفاده از نمونه‌گیری با جایگذاری (False بودن آن استاندارد جنگل ایزوله است)
# random_state: تضمین تکرارپذیری نتایج در اجراهای مختلف
# n_jobs: تعداد هسته‌های درگیر پردازنده (-1 یعنی استفاده از تمام هسته‌ها برای موازی‌سازی)
iso_forest = IsolationForest(
    n_estimators=100,
    contamination=0.1,
    max_samples='auto',
    max_features=1.0,
    bootstrap=False,
    random_state=42,
    n_jobs=-1
)
iso_forest.fit(X_train)

# 4. Predict
# پیش‌بینی برچسب‌ها (1 برای عادی، -1 برای ناهنجار) و محاسبه امتیاز ناهنجاری
y_pred = iso_forest.predict(X_test)
# امتیاز ناهنجاری: هرچه مقدار منفی‌تر باشد، احتمال ناهنجاری نمونه بیشتر است
y_scores = iso_forest.decision_function(X_test)

# 5. Evaluate with suitable items
# ارزیابی مدل با استفاده از شاخص‌های مناسب برای داده‌های نامتوازن (Imbalanced)
# ROC-AUC: توانایی مدل در تفکیک کلاس‌ها در آستانه‌های مختلف تصمیم‌گیری
# Average Precision: دقت مدل در شناسایی اقلیت کلاس ناهنجار (بسیار مهم‌تر از Accuracy در این مساله)
print("Model Evaluation Metrics:")
print(f"ROC-AUC Score: {roc_auc_score(y_test, y_scores):.4f}")
print(f"Average Precision Score: {average_precision_score(y_test, y_scores):.4f}")
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Normal', 'Anomaly']))