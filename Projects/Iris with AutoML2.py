# ======================================================
# AutoML Example using FLAML
# Dataset : Iris
# ======================================================

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from flaml import AutoML
from tabulate import tabulate
import logging
import time
#--------------------------------------------------------
# Hide FLAML Logs
#--------------------------------------------------------
logging.getLogger("flaml").setLevel(logging.CRITICAL)
#--------------------------------------------------------
# Load Dataset
#--------------------------------------------------------
X, y = load_iris(return_X_y=True)

#--------------------------------------------------------
# Split Dataset
#--------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=42)

#--------------------------------------------------------
# Create AutoML Object
#--------------------------------------------------------
automl = AutoML()

#--------------------------------------------------------
# Train Model
#--------------------------------------------------------
start = time.time()

automl.fit(
    X_train=X_train,
    y_train=y_train,
    task="classification",
    time_budget=10,
    verbose=0
)
end = time.time()

#--------------------------------------------------------
# Evaluate Model
#--------------------------------------------------------
accuracy = automl.score(X_test, y_test)

#--------------------------------------------------------
# Display Results
#--------------------------------------------------------
summary = [
    ["Dataset", "Iris"],
    ["Task", "Classification"],
    ["Training Samples", len(X_train)],
    ["Test Samples", len(X_test)],
    ["Best Algorithm", automl.best_estimator],
    ["Accuracy", f"{accuracy:.4f}"],
    ["Training Time", f"{end-start:.2f} sec"]
]
print("\nAutoML Summary\n")
print(tabulate(
    summary,
    headers=["Item", "Value"],
    tablefmt="grid"
))
#--------------------------------------------------------
# Best Model Parameters
#--------------------------------------------------------
print("\nBest Model Parameters\n")
model = automl.model.estimator
parameters = [
    ["Learning Rate", model.learning_rate],
    ["Number of Trees", model.n_estimators],
    ["Number of Leaves", model.num_leaves]
]

print(tabulate(
    parameters,
    headers=["Parameter", "Value"],
    tablefmt="grid"
))