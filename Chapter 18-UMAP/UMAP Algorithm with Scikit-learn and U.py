# UMAP Algorithm with Scikit-learn and UMAP-Learn

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import trustworthiness

import umap.umap_ as umap

# ----------------------------------------------------
# 1. Load data set
# ----------------------------------------------------
# بارگذاری مجموعه داده Wine
wine = load_wine()
X = wine.data
y = wine.target
print("Original data shape:", X.shape)

# ----------------------------------------------------
# 2. Train-test split
# ----------------------------------------------------
# تقسیم داده ها
X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,
    test_size=0.30,          # درصد داده آزمون
    random_state=42,          # قابلیت تکرار نتایج
    stratify=y                # حفظ نسبت کلاس ها
)

# ----------------------------------------------------
# 3. Feature Scaling
# ----------------------------------------------------
# نرمال سازی ویژگی ها
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# ----------------------------------------------------
# 4. Create UMAP Model
# ----------------------------------------------------
# ایجاد مدل UMAP
umap_model = umap.UMAP(
    n_neighbors=15,          # تعداد همسایه ها
    n_components=2,          # تعداد ابعاد خروجی
    metric='euclidean',      # معیار فاصله
    min_dist=0.10,           # حداقل فاصله بین نقاط
    spread=1.0,              # میزان پراکندگی خروجی
    learning_rate=1.0,       # نرخ یادگیری
    n_epochs=500,            # تعداد تکرار آموزش
    init='spectral',         # روش مقداردهی اولیه
    random_state=42,         # تکرارپذیری
    verbose=True             # نمایش روند اجرا
)

# ----------------------------------------------------
# 5. Fit Model
# ----------------------------------------------------
# آموزش مدل
X_train_umap = umap_model.fit_transform(X_train)
print("Reduced training shape:", X_train_umap.shape)

# ----------------------------------------------------
# 6. Transform Test Data
# ----------------------------------------------------
# کاهش ابعاد داده آزمون
X_test_umap = umap_model.transform(X_test)
print("Reduced test shape:", X_test_umap.shape)

# ----------------------------------------------------
# 7. Evaluate
# ----------------------------------------------------
# محاسبه شاخص Trustworthiness
tw = trustworthiness(
    X_train,
    X_train_umap,
    n_neighbors=15
)
print("Trustworthiness:", round(tw,4))

# ----------------------------------------------------
# 8. Save Results
# ----------------------------------------------------
# ذخیره نتایج در DataFrame

df = pd.DataFrame({
    "UMAP_1": X_train_umap[:,0],
    "UMAP_2": X_train_umap[:,1],
    "Class": y_train
})
print(df.head())

# ----------------------------------------------------
# 9. Visualization
# ----------------------------------------------------
# رسم نمودار دوبعدی
plt.figure(figsize=(8,6))
scatter = plt.scatter(
    X_train_umap[:,0],
    X_train_umap[:,1],
    c=y_train,
    s=50
)
plt.title("UMAP Projection")
plt.xlabel("UMAP Dimension 1")
plt.ylabel("UMAP Dimension 2")
plt.colorbar(scatter)
plt.grid(True)
plt.show()