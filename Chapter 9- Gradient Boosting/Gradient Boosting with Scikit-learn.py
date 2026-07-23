# -*- coding: utf-8 -*-
"""
مثال حل‌شده الگوریتم Gradient Boosting با استفاده از کتابخانه scikit-learn
مسئله: رگرسیون روی ۴ نمونه داده با ویژگی x و مقدار هدف y
"""

# وارد کردن کتابخانه numpy برای کار با آرایه‌های عددی
import numpy as np

# وارد کردن مدل GradientBoostingRegressor از کتابخانه scikit-learn
from sklearn.ensemble import GradientBoostingRegressor

# تعریف داده‌های ویژگی (x) به صورت آرایه دو بعدی، چون sklearn ورودی دو بعدی می‌خواهد
X = np.array([[1], [2], [3], [4]])

# تعریف مقادیر هدف (y) که مدل باید آن‌ها را پیش‌بینی کند
y = np.array([30, 20, 40, 50])

# ساخت مدل Gradient Boosting با تنظیمات دقیقاً مطابق حل دستی مسئله
model = GradientBoostingRegressor(
    n_estimators=2,        # تعداد تکرارها (درخت‌ها) برابر با M=2 مطابق حل دستی
    learning_rate=0.1,     # نرخ یادگیری برابر با 0.1 مطابق حل دستی
    max_depth=1,           # عمق درخت برابر با 1 یعنی استفاده از Stump (شاخه ساده)
    loss='squared_error',  # تابع زیان خطای مربعی مطابق فرمول‌بندی مسئله
    criterion='squared_error'  # معیار شکاف درخت نیز خطای مربعی است
)

# آموزش مدل روی داده‌های X و y با استفاده از تابع fit
model.fit(X, y)

# محاسبه مقدار اولیه مدل (F0) که میانگین y است، برای مقایسه با حل دستی
F0 = np.mean(y)

# چاپ مقدار اولیه مدل به انگلیسی طبق درخواست کاربر
print(f"Initial prediction F0 (mean of y): {F0}")

# گرفتن پیش‌بینی نهایی مدل پس از دو تکرار (F2) با استفاده از تابع predict
final_predictions = model.predict(X)

# چاپ عنوان جدول نتایج به انگلیسی
print("\nFinal predictions after 2 boosting iterations:")

# حلقه برای چاپ مقدار x، y واقعی و پیش‌بینی نهایی هر نمونه
for xi, yi, pred in zip(X.flatten(), y, final_predictions):
    # چاپ هر سطر شامل x، مقدار واقعی y و مقدار پیش‌بینی‌شده F2
    print(f"x = {xi}, actual y = {yi}, predicted F2 = {pred:.2f}")

# محاسبه باقیمانده نهایی (خطا) بین مقدار واقعی و پیش‌بینی مدل
residuals = y - final_predictions

# چاپ باقیمانده‌های نهایی برای مقایسه با محاسبات دستی
print("\nFinal residuals (actual - predicted):")
print(np.round(residuals, 2))

# استخراج پیش‌بینی مرحله به مرحله مدل با استفاده از staged_predict
# این تابع خروجی مدل را بعد از هر تکرار (هر درخت اضافه‌شده) برمی‌گرداند
staged_preds = list(model.staged_predict(X))

# چاپ عنوان بخش مربوط به مراحل میانی
print("\nStep-by-step predictions (F1 after iteration 1, F2 after iteration 2):")

# حلقه برای چاپ پیش‌بینی مدل بعد از هر مرحله از بوستینگ
for step, pred in enumerate(staged_preds, start=1):
    # چاپ شماره مرحله و مقادیر پیش‌بینی‌شده گرد شده تا دو رقم اعشار
    print(f"After iteration {step} (F{step}): {np.round(pred, 2)}")

# محاسبه اهمیت ویژگی‌ها با استفاده از خاصیت feature_importances_
print("\nFeature importance:")
print(model.feature_importances_)