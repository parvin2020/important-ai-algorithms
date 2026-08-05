#=========================================================
# Example : Loan Default Prediction using Simple RNN
# Author : Dr. Bahram Parvin
#=========================================================
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import SimpleRNN
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import MinMaxScaler

#=========================================================
# Training Data
# Each row represents one customer
# Columns represent transaction amounts in the last 6 months
#=========================================================
X = np.array([
    [120,118,121,119,122,120],   # Normal
    [140,125,100,80,55,30],      # Risk
    [80,82,81,79,80,83],         # Normal
    [160,145,120,90,70,40],      # Risk
    [100,102,101,99,100,98],     # Normal
    [200,190,170,140,110,80],    # Risk
    [90,91,92,91,90,92],         # Normal
    [180,170,150,120,90,60]      # Risk
], dtype="float32")

#=========================================================
# Labels
# 0 = Low Risk
# 1 = High Risk
#=========================================================
y = np.array([0,1,0,1,0,1,0,1], dtype="float32")

#=========================================================
# Normalize Data
#=========================================================
maximum = np.max(X)
scaler = MinMaxScaler()
X = scaler.fit_transform(X)

#=========================================================
# Reshape for RNN
# (samples, time_steps, features)
#=========================================================
X = X.reshape((8,6,1))
print("Shape of X =", X.shape)

#=========================================================
# Build RNN Model
#=========================================================
model = Sequential([
    Input(shape=(6,1)),
    SimpleRNN(units=8, activation="tanh"),
    Dense(1,activation="sigmoid" )
])
#=========================================================
# Compile Model
#=========================================================
model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

#=========================================================
# Model Summary
#=========================================================
print("\nNetwork Structure")
model.summary()

#=========================================================
# Train Model
#=========================================================
history = model.fit( X, y,epochs=300, verbose=0)
print("\nTraining Finished Successfully.")

#=========================================================
# Evaluate Model
#=========================================================
loss, accuracy = model.evaluate( X, y, verbose=0)
print("\nTraining Accuracy =", round(accuracy*100,2), "%")

#=========================================================
# Predict New Customer
#=========================================================
new_customer = np.array([
170,
150,
120,
95,
70,
45
], dtype="float32")
maximum = np.max(X)
minimum= np.min(X)
new_customer = (new_customer - minimum)/(maximum-minimum)
new_customer = new_customer.reshape((1,6,1))
prediction = model.predict(new_customer, verbose=0)
print("\nProbability of Default =", round(float(prediction[0][0]),4))
if prediction >= 0.5:
    print("Prediction : High Risk Customer")
else:
    print("Prediction : Low Risk Customer")