import numpy as np

# -------------------------
# Training Data
# -------------------------

customers = ["A", "B", "C", "D", "E"]

# True labels
y = np.array([-1, -1, 1, 1, 1])

# Initial weights
w = np.ones(len(y)) / len(y)

print("Initial Weights:")
print(w)

# -------------------------
# Weak Learner 1
# Rule: If income is high => Repay Loan
# -------------------------

h1 = np.array([-1, -1, 1, 1, -1])

# Calculate error
error1 = np.sum(w[y != h1])

print("\nError of Weak Learner 1:")
print(round(error1, 3))

# Calculate alpha
alpha1 = 0.5 * np.log((1 - error1) / error1)

print("\nAlpha 1:")
print(round(alpha1, 3))

# Update weights
w = w * np.exp(-alpha1 * y * h1)

# Normalize weights
w = w / np.sum(w)

print("\nUpdated Weights:")
for customer, weight in zip(customers, w):
    print(f"Customer {customer}: {weight:.3f}")

# -------------------------
# Weak Learner 2
# Rule: If age is high => No Repayment
# -------------------------

h2 = np.array([1, -1, 1, -1, 1])

# Calculate error
error2 = np.sum(w[y != h2])

print("\nError of Weak Learner 2:")
print(round(error2, 3))

# Calculate alpha
alpha2 = 0.5 * np.log((1 - error2) / error2)

print("\nAlpha 2:")
print(round(alpha2, 3))

# -------------------------
# Prediction for a New Customer
# High Income = Yes
# High Age = No
# -------------------------

new_h1 = 1
new_h2 = 1

final_score = alpha1 * new_h1 + alpha2 * new_h2

print("\nFinal Score:")
print(round(final_score, 3))

if final_score > 0:
    print("\nPrediction: Customer WILL repay the loan.")
else:
    print("\nPrediction: Customer WILL NOT repay the loan.")