# ============================================
# Image Denoising with Autoencoder
# ============================================
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense

# ------------------------------------------------
# 1. Load MNIST dataset
# ------------------------------------------------
# مسیر فایل MNIST در ریشه درایو D
mnist_path = r"D:\mnist.npz"
# باز کردن فایل
data = np.load(mnist_path)
# دریافت تصاویر آموزشی
X_train = data["training_images"]
# دریافت تصاویر آزمایشی
X_test = data["test_images"]

print("Dataset loaded successfully.")
print("Training images:", X_train.shape)
print("Test images:", X_test.shape)
print("Dataset loaded successfully.")
print("Training images:", X_train.shape)
print("Test images:", X_test.shape)

# ------------------------------------------------
# 2. Normalize the images
# ------------------------------------------------
# تبدیل مقدار پیکسل‌ها از 0-255 به بازه 0-1
X_train = X_train.astype("float32") / 255.0
X_test = X_test.astype("float32") / 255.0


# ------------------------------------------------
# 3. Flatten the images
# ------------------------------------------------
# هر تصویر 28×28 را به یک بردار 784تایی تبدیل می‌کنیم
X_train = X_train.reshape((len(X_train), 784))
X_test = X_test.reshape((len(X_test), 784))
print("Images normalized and flattened.")

# ------------------------------------------------
# 4. Add noise to the images
# ------------------------------------------------
# ایجاد نویز تصادفی
noise_factor = 0.4
X_train_noisy = X_train + noise_factor * np.random.normal(
    loc=0.0,
    scale=1.0,
    size=X_train.shape )

X_test_noisy = X_test + noise_factor * np.random.normal(
    loc=0.0,
    scale=1.0,
    size=X_test.shape )

# محدود کردن مقادیر پیکسل‌ها به بازه 0 تا 1
X_train_noisy = np.clip(X_train_noisy, 0.0, 1.0)
X_test_noisy = np.clip(X_test_noisy, 0.0, 1.0)
print("Noise added to images.")


# ------------------------------------------------
# 5. Build the Autoencoder
# ------------------------------------------------
# ورودی شبکه شامل 784 مقدار پیکسل است
input_image = Input(shape=(784,))

# Encoder
# کاهش تعداد ویژگی‌ها و استخراج اطلاعات مهم تصویر
encoded = Dense(128, activation="relu")(input_image)
encoded = Dense(64, activation="relu")(encoded)

# Decoder
# بازسازی تصویر اصلی از اطلاعات فشرده‌شده
decoded = Dense(128, activation="relu")(encoded)
decoded = Dense(784, activation="sigmoid")(decoded)

# ساخت مدل Autoencoder
autoencoder = Model(input_image, decoded)

# ------------------------------------------------
# 6. Compile the model
# ------------------------------------------------

# استفاده از Adam برای بهینه‌سازی
# loss مشخص می‌کند بازسازی تصویر چقدر با تصویر اصلی تفاوت دارد
autoencoder.compile(
    optimizer="adam",
    loss="binary_crossentropy")

print("Autoencoder created successfully.")

# ------------------------------------------------
# 7. Train the Autoencoder
# ------------------------------------------------

# ورودی: تصاویر نویزی
# خروجی مورد انتظار: تصاویر اصلی بدون نویز
history = autoencoder.fit(
    X_train_noisy,
    X_train,
    epochs=10,
    batch_size=256,
    validation_data=(X_test_noisy, X_test),
    verbose=1 )

print("Training completed.")

# ------------------------------------------------
# 8. Evaluate the model
# ------------------------------------------------
# محاسبه مقدار خطا روی تصاویر تست
test_loss = autoencoder.evaluate(
    X_test_noisy,
    X_test,
    verbose=0 )

print("Test Loss:", test_loss)

# ------------------------------------------------
# 9. Reconstruct noisy images
# ------------------------------------------------

# استفاده از Autoencoder برای حذف نویز
decoded_images = autoencoder.predict(
    X_test_noisy,
    verbose=0
)

print("Images reconstructed successfully.")

# ------------------------------------------------
# 10. Display the results
# ------------------------------------------------
# تعداد تصاویری که می‌خواهیم نمایش دهیم
n = 8
plt.figure(figsize=(20, 6))

# نمایش تصاویر نویزی
for i in range(n):
    ax = plt.subplot(3, n, i + 1)
    # تبدیل بردار 784تایی به تصویر 28×28
    plt.imshow(
        X_test_noisy[i].reshape(28, 28),
        cmap="gray" )

    plt.title("Noisy")
    plt.axis("off")

# نمایش تصاویر بازسازی‌شده
for i in range(n):
    ax = plt.subplot(3, n, i + 1 + n)
    plt.imshow(
        decoded_images[i].reshape(28, 28),
        cmap="gray" )
    plt.title("Denoised")
    plt.axis("off")


# نمایش تصاویر اصلی
for i in range(n):
    ax = plt.subplot(3, n, i + 1 + 2 * n)
    plt.imshow(
        X_test[i].reshape(28, 28),
        cmap="gray" )

    plt.title("Original")
    plt.axis("off")

plt.tight_layout()
plt.show()