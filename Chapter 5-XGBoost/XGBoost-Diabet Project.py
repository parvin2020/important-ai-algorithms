# ============================================
# مرحله 1: وارد کردن کتابخانه‌های مورد نیاز
# ============================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb

# ============================================
# مرحله 2: بارگذاری دیتاست
# ============================================
# دیتاست Diabetes شامل 442 نمونه و 10 ویژگی عددی است
diabetes = load_diabetes()
X = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
y = pd.Series(diabetes.target, name='Disease_Progression')

print("=" * 60)
print("Dataset Information:")
print("=" * 60)
print(f"Number of samples: {X.shape[0]}")
print(f"Number of features: {X.shape[1]}")
print(f"Feature names: {list(X.columns)}")
print(f"\nFirst 5 rows of data:")
print(X.head())
print(f"\nTarget variable statistics:")
print(y.describe())

# ============================================
# مرحله 3: تقسیم داده‌ها به آموزش و آزمون
# ============================================
# 80 درصد برای آموزش و 20 درصد برای تست
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\n" + "=" * 60)
print("Data Split Summary:")
print("=" * 60)
print(f"Training set size: {X_train.shape[0]} samples")
print(f"Testing set size:  {X_test.shape[0]} samples")

# ============================================
# مرحله 4: آموزش مدل XGBoost
# ============================================
# ایجاد و تنظیم مدل با پارامترهای پیش‌فرض
model = xgb.XGBRegressor(
    n_estimators=100,       # تعداد درخت‌ها
    learning_rate=0.1,      # نرخ یادگیری
    max_depth=3,            # حداکثر عمق درخت
    random_state=42,        # برای تکرارپذیری
    verbosity=0             # عدم نمایش پیام‌های اضافی
)

print("\n" + "=" * 60)
print("Training XGBoost Model...")
print("=" * 60)

# آموزش مدل روی داده‌های آموزشی
model.fit(X_train, y_train)
print("Model training completed successfully!")

# ============================================
# مرحله 5: پیش‌بینی روی داده‌های تست
# ============================================
y_pred = model.predict(X_test)

print("\n" + "=" * 60)
print("Sample Predictions (First 10):")
print("=" * 60)
comparison = pd.DataFrame({
    'Actual': y_test.values[:10],
    'Predicted': np.round(y_pred[:10], 2)
})
print(comparison)

# ============================================
# مرحله 6: ارزیابی مدل
# ============================================
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\n" + "=" * 60)
print("Model Evaluation Metrics:")
print("=" * 60)
print(f"MSE  (Mean Squared Error):     {mse:.2f}")
print(f"RMSE (Root Mean Squared Error): {rmse:.2f}")
print(f"MAE  (Mean Absolute Error):    {mae:.2f}")
print(f"R²   (R-squared Score):        {r2:.4f}")
print("=" * 60)

# ============================================
# مرحله 7: نمایش اهمیت ویژگی‌ها
# ============================================
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
}).sort_values(by='Importance', ascending=False)

print("\n" + "=" * 60)
print("Feature Importance Ranking:")
print("=" * 60)
for i, row in feature_importance.iterrows():
    bar = "█" * int(row['Importance'] * 100)
    print(f"{row['Feature']:8s} | {bar} ({row['Importance']:.4f})")

# رسم نمودار اهمیت ویژگی‌ها
plt.figure(figsize=(10, 6))
sns.barplot(data=feature_importance, x='Importance', y='Feature', palette='viridis')
plt.title('Feature Importance in XGBoost Model')
plt.xlabel('Importance Score')
plt.ylabel('Feature')
plt.tight_layout()
plt.show()

# ============================================
# مرحله 8: مقایسه مقادیر واقعی و پیش‌بینی شده
# ============================================
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.6, edgecolors='k', s=60)
plt.plot([y_test.min(), y_test.max()], 
         [y_test.min(), y_test.max()], 
         'r--', lw=2, label='Perfect Prediction')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted Disease Progression')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# ============================================
# مرحله 9: پیش‌بینی برای یک نمونه جدید
# ============================================
# ایجاد یک نمونه فرضی با مقادیر میانگین
new_sample = pd.DataFrame([X.mean()], columns=X.columns)
new_prediction = model.predict(new_sample)[0]

print("\n" + "=" * 60)
print("Prediction for a New Sample:")
print("=" * 60)
print(f"Input features (mean values):")
print(new_sample.round(2).to_string(index=False))
print(f"\nPredicted Disease Progression: {new_prediction:.2f}")
print("=" * 60)