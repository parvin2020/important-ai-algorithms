import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Dataset
data = {
    'income': [12, 8, 15, 7, 18, 9, 11, 14, 10, 16],
    'age': [30, 25, 35, 23, 40, 28, 32, 34, 29, 38],
    'loan': [1, 0, 1, 0, 1, 0, 1, 1, 0, 1]
}

df = pd.DataFrame(data)
X = df[['income', 'age']]
y = df['loan']

# Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train Logistic Regression Model
model = LogisticRegression(
    solver='liblinear', 
    C=1.0, 
    random_state=42
)
model.fit(X_scaled, y)

# Prediction for new customer
new_customer = scaler.transform([[13, 33]])
probability = model.predict_proba(new_customer)[0][1]

print(f"Probability of getting loan: {probability:.3f} ({probability*100:.1f}%)")
print(f"Final Decision: {'Yes' if probability >= 0.5 else 'No'}")

# Model Coefficients
print("\nModel Coefficients:")
print(f"Income Weight: {model.coef_[0][0]:.4f}")
print(f"Age Weight: {model.coef_[0][1]:.4f}")
print(f"Intercept: {model.intercept_[0]:.4f}")