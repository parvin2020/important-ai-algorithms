import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

# Dataset
data = {
    'income': [8, 12, 15, 7, 18, 9, 13, 11, 14, 6],
    'age': [25, 30, 35, 22, 40, 28, 32, 29, 33, 24],
    'buyer': [0, 1, 1, 0, 1, 0, 1, 1, 1, 0]
}

df = pd.DataFrame(data)
X = df[['income', 'age']]
y = df['buyer']

# Feature Scaling (Very important for SVM)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train SVM Model
model = SVC(
    kernel='rbf', 
    C=1.0, 
    gamma='scale', 
    random_state=42,
    probability=True
)
model.fit(X_scaled, y)

# Prediction for new customer
new_customer = scaler.transform([[13, 32]])
probability = model.predict_proba(new_customer)[0][1]

print(f"Probability of being a buyer: {probability:.3f} ({probability*100:.1f}%)")
print(f"Final Prediction: {'Buyer' if probability >= 0.5 else 'Non-Buyer'}")

# Model parameters
print(f"\nSupport Vectors Count: {model.n_support_}")