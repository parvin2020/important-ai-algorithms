# الگوریتم بیز ساده با استفاده از Scikit-learn
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import CategoricalNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1. ایجاد دیتافریم یا بارگذاری مجموعه داده
# ایجاد داده‌های مثال عددی بالا برای استفاده در پایتون
data = {
    'X1': [0, 0, 1, 1, 0, 1, 0, 1, 0],
    'X2': [0, 1, 0, 1, 0, 0, 1, 1, 0],
    'Y':  [0, 0, 0, 1, 1, 1, 1, 1, 1]
}
df = pd.DataFrame(data)
X = df[['X1', 'X2']]
y = df['Y']

# 2. تقسیم داده‌ها به آموزش و آزمون
# تقسیم 80 درصد برای آموزش و 20 درصد برای آزمون با ثابت بودن seed برای تکرارپذیری
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. آموزش مدل
# ایجاد مدل بیز ساده دسته‌ای با تنظیم هایپرپارامترهای مهم
# alpha: پارامتر هموارسازی لاپلاس برای جلوگیری از صفر شدن احتمالات
# fit_prior: آیا احتمالات پیشین کلاس‌ها از داده‌ها یاد گرفته شود یا خیر
# class_prior: احتمالات پیشین کلاس‌ها (در اینجا None است تا از داده‌ها محاسبه شود)
# min_categories: حداقل تعداد دسته‌های مجاز برای هر ویژگی
model = CategoricalNB(alpha=1.0, fit_prior=True, class_prior=None, min_categories=None)
model.fit(X_train, y_train)

# 4. پیش‌بینی
# پیش‌بینی کلاس‌ها و همچنین احتمالات تعلق به هر کلاس برای داده‌های آزمون
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)
print("Predicted Classes:", y_pred)
print("Predicted Probabilities:\n", y_prob)

# 5. ارزیابی با شاخص‌های مناسب
# محاسبه دقت کلی، ماتریس درهم‌ریختگی و گزارش کامل شامل Precision، Recall و F1-Score
acc = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print("Accuracy:", acc)
print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n", class_report)