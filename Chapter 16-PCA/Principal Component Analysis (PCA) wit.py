# Principal Component Analysis (PCA) with Scikit-learn

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error

#---------------------------------------------------
# 1. Load data set or Create data frame
# ایجاد مجموعه داده فرضی
#---------------------------------------------------
data = {
    "Math":        [2,4,6,8,10],
    "Programming": [1,3,5,7,9]
}

df = pd.DataFrame(data)
print("Original Dataset")
print(df)

#---------------------------------------------------
# استانداردسازی داده ها
#---------------------------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

#---------------------------------------------------
# 2. Train-test split
# در PCA تقسیم آموزش و آزمون معمولاً وجود ندارد.
#---------------------------------------------------
X = X_scaled

#---------------------------------------------------
# 3. Train model
#---------------------------------------------------

pca = PCA(
    n_components=1,      # تعداد مؤلفه های اصلی
    svd_solver='auto',   # روش محاسبه SVD
    whiten=False,        # سفیدسازی مؤلفه ها
    copy=True,           # کپی داده ها
    random_state=42      # فقط برای برخی Solverها کاربرد دارد
)

X_pca = pca.fit_transform(X)

print("\nReduced Dataset")
print(X_pca)

#---------------------------------------------------
# 4. Predict
# در PCA مرحله Predict وجود ندارد.
#---------------------------------------------------

#---------------------------------------------------
# بازسازی داده ها
#---------------------------------------------------

X_inverse = pca.inverse_transform(X_pca)

#---------------------------------------------------
# 5. Evaluate
#---------------------------------------------------

print("\nExplained Variance")
print(pca.explained_variance_)

print("\nExplained Variance Ratio")
print(pca.explained_variance_ratio_)

print("\nCumulative Explained Variance")
print(np.cumsum(pca.explained_variance_ratio_))

print("\nPrincipal Components")
print(pca.components_)

print("\nSingular Values")
print(pca.singular_values_)

#---------------------------------------------------
# محاسبه خطای بازسازی
#---------------------------------------------------

mse = mean_squared_error(X, X_inverse)

print("\nReconstruction Error")
print(mse)

#---------------------------------------------------
# Scree Plot
#---------------------------------------------------

plt.figure(figsize=(6,4))

plt.plot(
    range(1, len(pca.explained_variance_)+1),
    pca.explained_variance_,
    marker='o'
)

plt.title("Scree Plot")
plt.xlabel("Principal Component")
plt.ylabel("Eigenvalue")

plt.grid(True)

plt.show()

#---------------------------------------------------
# Cumulative Explained Variance
#---------------------------------------------------

plt.figure(figsize=(6,4))

plt.plot(
    np.cumsum(pca.explained_variance_ratio_),
    marker='o'
)

plt.title("Cumulative Explained Variance")

plt.xlabel("Number of Components")

plt.ylabel("Explained Variance Ratio")

plt.grid(True)

plt.show()

#---------------------------------------------------
# نمایش داده های کاهش یافته
#---------------------------------------------------

plt.figure(figsize=(6,4))

plt.scatter(
    X_pca,
    np.zeros(len(X_pca)),
    s=120
)

plt.title("Projection onto First Principal Component")

plt.xlabel("PC1")

plt.yticks([])

plt.grid(True)

plt.show()