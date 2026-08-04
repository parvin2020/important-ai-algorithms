import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense

#====================================================
# بارگذاری داده ها
#====================================================
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

print("Training Images :", X_train.shape)
print("Training Labels :", y_train.shape)
print("Test Images :", X_test.shape)
print("Test Labels :", y_test.shape)

#====================================================
# نرمال سازی تصاویر
#====================================================
X_train = X_train.astype("float32") / 255.0
X_test  = X_test.astype("float32") / 255.0

#====================================================
# نمایش یک تصویر
#====================================================
class_names = [
    "Airplane",
    "Automobile",
    "Bird",
    "Cat",
    "Deer",
    "Dog",
    "Frog",
    "Horse",
    "Ship",
    "Truck"
]

plt.imshow(X_train[0])
plt.title(class_names[y_train[0][0]])
plt.axis("off")
plt.show()

#====================================================
# ساخت مدل CNN
#====================================================
model = Sequential([
    Input(shape=(32,32,3)),
    Conv2D(
        filters=32,
        kernel_size=(3,3),
        activation="relu"
    ),
    MaxPooling2D(pool_size=(2,2)),
    Conv2D(
        filters=64,
        kernel_size=(3,3),
        activation="relu"
    ),
    MaxPooling2D(pool_size=(2,2)),
    Flatten(),
    Dense(
        128,
        activation="relu"
    ),
    Dense(
        10,
        activation="softmax"
    )
])

#====================================================
# کامپایل مدل
#====================================================
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

#====================================================
# آموزش مدل
#====================================================
history = model.fit(
    X_train,
    y_train,
    epochs=10,
    batch_size=64,
    validation_split=0.2,
    verbose=1
)

#====================================================
# ارزیابی مدل
#====================================================
loss, accuracy = model.evaluate(
    X_test,
    y_test,
    verbose=1
)
print("\nTest Accuracy =", accuracy)

#====================================================
# پیش بینی
#====================================================
prediction = model.predict(X_test)
predicted_class = np.argmax(prediction, axis=1)
print("\nPredicted Class :", predicted_class[0])
print("Actual Class    :", y_test[0][0])
print("Predicted Label :", class_names[predicted_class[0]])
print("Actual Label    :", class_names[y_test[0][0]])

#====================================================
# نمایش تصویر پیش بینی شده
#====================================================
plt.imshow(X_test[0])
plt.title(
    "Prediction : " +
    class_names[predicted_class[0]]
)
plt.axis("off")
plt.show()

#====================================================
# خلاصه مدل
#====================================================
model.summary()