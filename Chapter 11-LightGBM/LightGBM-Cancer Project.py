#importing Libraries
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load the breast cancer dataset
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a LightGBM dataset
train_data = lgb.Dataset(X_train, label=y_train)

# Define parameters for GBDT
params = {
    'objective': 'binary',
    'boosting_type': 'gbdt',
    'metric': 'binary_logloss',
    'num_leaves': 11,
    'learning_rate': 0.05,
    'feature_fraction': 0.9
}

# Train the GBDT model
gbm = lgb.train(params, train_data, num_boost_round=100)

# Make predictions on the test set
y_pred = gbm.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, (y_pred > 0.5).astype(int))
print("Accuracy:", accuracy)