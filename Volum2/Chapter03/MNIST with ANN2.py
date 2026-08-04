"""
main.py
--------
نقطه ورود اصلی پروژه. اجرای این فایل تمام مراحل زیر را به‌ترتیب انجام می‌دهد:
  1. بارگذاری دیتاست از روی دیسک
  2. نمایش چند نمونه برای آشنایی با داده
  3. پیش‌پردازش داده
  4. ساخت مدل (MLP یا CNN)
  5. آموزش مدل
  6. ارزیابی نهایی روی داده تست
"""

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.utils import to_categorical

from data_loader import load_mnist
from model import build_mlp_model, build_cnn_model
from train import train_model
from evaluate import evaluate_model

# ============== تنظیمات قابل تغییر توسط دانشجو ==============
DATASET_ROOT = "D:\AI Algorithms-Volum2-0509\Python Codes\datasets"      # مسیر ریشه‌ای که دیتاست در آن ذخیره شده است
MODEL_TYPE = "cnn"        # "cnn" یا "mlp"
EPOCHS = 10
BATCH_SIZE = 128
OUTPUT_DIR = "outputs"
# ================================================================


def show_sample_images(x, y, n=10):
    """نمایش چند نمونه تصادفی از دیتاست برای آشنایی اولیه با داده."""
    idxs = np.random.choice(len(x), n, replace=False)
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    for ax, idx in zip(axes.flat, idxs):
        ax.imshow(x[idx], cmap="gray")
        ax.set_title(f"Label: {y[idx]}")
        ax.axis("off")
    plt.tight_layout()
    plt.show()


def preprocess(x, y, num_classes=10):
    """
    پیش‌پردازش داده:
      - نرمال‌سازی مقادیر پیکسل به بازه [0, 1]
      - افزودن بعد کانال برای ورودی CNN
      - تبدیل برچسب‌ها به بردار one-hot
    """
    x_norm = x.astype("float32") / 255.0
    x_norm = np.expand_dims(x_norm, axis=-1)  # (N, 28, 28) -> (N, 28, 28, 1)
    y_onehot = to_categorical(y, num_classes=num_classes)
    return x_norm, y_onehot


def main():
    print("مرحله ۱: بارگذاری دیتاست از", DATASET_ROOT)
    (x_train_raw, y_train_raw), (x_test_raw, y_test_raw) = load_mnist(DATASET_ROOT)
    print("تعداد نمونه آموزش:", len(x_train_raw))
    print("تعداد نمونه تست:", len(x_test_raw))

    print("\nمرحله ۲: نمایش چند نمونه از داده (برای بستن پنجره، آن را ببندید)")
    show_sample_images(x_train_raw, y_train_raw)

    print("\nمرحله ۳: پیش‌پردازش داده")
    x_train, y_train = preprocess(x_train_raw, y_train_raw)
    x_test, y_test = preprocess(x_test_raw, y_test_raw)

    # جدا کردن بخشی از داده آموزش برای اعتبارسنجی (validation)
    val_split = int(0.9 * len(x_train))
    x_val, y_val = x_train[val_split:], y_train[val_split:]
    x_train, y_train = x_train[:val_split], y_train[:val_split]

    if MODEL_TYPE == "mlp":
        # مدل MLP نیاز به ورودی ۲۸x۲۸ (بدون بعد کانال) دارد
        x_train_model = x_train.squeeze(-1)
        x_val_model = x_val.squeeze(-1)
        x_test_model = x_test.squeeze(-1)
        model = build_mlp_model()
    else:
        x_train_model, x_val_model, x_test_model = x_train, x_val, x_test
        model = build_cnn_model()

    print(f"\nمرحله ۴: ساخت مدل ({MODEL_TYPE.upper()})")
    model.summary()

    print("\nمرحله ۵: آموزش مدل")
    train_model(model, x_train_model, y_train, x_val_model, y_val,
                epochs=EPOCHS, batch_size=BATCH_SIZE, output_dir=OUTPUT_DIR)

    print("\nمرحله ۶: ارزیابی روی داده تست")
    evaluate_model(model, x_test_model, y_test, y_test_raw, output_dir=OUTPUT_DIR)

    model_path = f"{OUTPUT_DIR}/model.h5"
    model.save(model_path)
    print(f"\nمدل نهایی ذخیره شد در: {model_path}")


if __name__ == "__main__":
    main()