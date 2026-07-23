import lightgbm as lgb
import numpy as np
import pandas as pd

# ============================================
# داده‌های دقیقاً مطابق مثال عددی
# ============================================
data = {
    'Area': [50, 60, 70, 80, 90, 100, 110, 120],
    'Rooms': [1, 1, 2, 2, 2, 3, 3, 3],
    'Price': [500, 600, 700, 800, 850, 950, 1000, 1100]
}

df = pd.DataFrame(data)

print("Dataset:")
print(df)
print(f"\nTotal samples: {len(df)}\n")

# ============================================
# آماده‌سازی داده‌ها
# ============================================
X = df[['Area', 'Rooms']]
y = df['Price']

train_data = lgb.Dataset(X, label=y)

# ============================================
# پارامترهای مدل (ساده‌شده برای مثال)
# ============================================
params = {
    'objective': 'regression',
    'metric': 'mse',
    'boosting_type': 'gbdt',
    'num_leaves': 4,
    'learning_rate': 0.1,
    'min_data_in_leaf': 2,
    'min_gain_to_split': 0.0,
    'verbose': -1
}

# ============================================
# آموزش مدل
# ============================================
print("Training LightGBM model...")
model = lgb.train(
    params,
    train_data,
    num_boost_round=10
)
print("Training completed.\n")

# ============================================
# پیش‌بینی
# ============================================
predictions = model.predict(X)

print("Results:")
print("-" * 50)
for i in range(len(df)):
    print(f"Sample {i+1}: Area={df.iloc[i]['Area']}, "
          f"Rooms={df.iloc[i]['Rooms']}, "
          f"Actual={df.iloc[i]['Price']}, "
          f"Predicted={predictions[i]:.2f}")
print("-" * 50)

# محاسبه خطا
mse = np.mean((y - predictions)**2)
print(f"\nMSE: {mse:.2f}")

# ============================================
# پیش‌بینی داده‌های جدید
# ============================================
new_data = pd.DataFrame({
    'Area': [65, 85, 105],
    'Rooms': [1, 2, 3]
})

new_predictions = model.predict(new_data)

print("\nPredictions for new data:")
for i in range(len(new_data)):
    print(f"Area={new_data.iloc[i]['Area']}, "
          f"Rooms={new_data.iloc[i]['Rooms']} -> "
          f"Predicted Price: {new_predictions[i]:.2f}")