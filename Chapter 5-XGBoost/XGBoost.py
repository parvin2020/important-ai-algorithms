import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Dataset
data = {
    'area': [60, 80, 100, 120, 90, 110, 95, 130, 105],
    'rooms': [2, 3, 3, 4, 2, 3, 3, 4, 3],
    'price': [250, 350, 420, 550, 380, 480, 410, 620, 465]
}

df = pd.DataFrame(data)
X = df[['area', 'rooms']]
y = df['price']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Train XGBoost Regressor
model = xgb.XGBRegressor(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric='rmse'
)

model.fit(X_train, y_train)

# Prediction
new_house = [[105, 3]]
predicted_price = model.predict(new_house)[0]

print(f"Predicted price for the new house (105m², 3 rooms): {predicted_price:.1f} Million Tomans")

# Evaluation
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"Model RMSE: {rmse:.2f}")

# Feature Importance
print("\nFeature Importance:")
importance = pd.Series(model.feature_importances_, index=X.columns)
print(importance.sort_values(ascending=False))