# UMAP Algorithm with Python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import umap.umap_ as umap

from sklearn.preprocessing import StandardScaler
from sklearn.manifold import trustworthiness
from sklearn.metrics import pairwise_distances

# -------------------------------------------------------
# 1. Create data frame
# -------------------------------------------------------
# ایجاد مجموعه داده مثال عددی

data = {
    "Sample": ["A", "B", "C", "D", "E"],
    "X1": [2, 3, 3, 8, 9],
    "X2": [3, 4, 2, 8, 9]
}

df = pd.DataFrame(data)
print("Original Dataset")
print(df)

# -------------------------------------------------------
# 2. Feature Scaling
# -------------------------------------------------------
# نرمال سازی داده ها
scaler = StandardScaler()
X = scaler.fit_transform(df[["X1", "X2"]])

# -------------------------------------------------------
# 3. Distance Matrix
# -------------------------------------------------------
# محاسبه ماتریس فاصله

distance_matrix = pairwise_distances(X, metric="euclidean")
print("\nDistance Matrix")
print(np.round(distance_matrix,3))

# -------------------------------------------------------
# 4. Train UMAP Model
# -------------------------------------------------------
# ایجاد مدل UMAP

model = umap.UMAP(
    n_neighbors=2,          # تعداد همسایه ها
    n_components=2,         # کاهش به دو بعد
    metric="euclidean",     # معیار فاصله
    min_dist=0.10,          # حداقل فاصله
    spread=1.0,             # میزان پراکندگی
    learning_rate=1.0,      # نرخ یادگیری
    n_epochs=300,           # تعداد تکرار
    init="spectral",        # مقداردهی اولیه
    random_state=42,        # قابلیت تکرار
    verbose=True            # نمایش روند اجرا

)
# -------------------------------------------------------
# 5. Fit Model
# -------------------------------------------------------
# آموزش مدل
embedding = model.fit_transform(X)
print("\nReduced Coordinates")
print(np.round(embedding,3))

# -------------------------------------------------------
# 6. Evaluate
# -------------------------------------------------------
# محاسبه شاخص Trustworthiness
tw = trustworthiness(X, embedding, n_neighbors=2)
print("\nTrustworthiness =", round(tw,4))

# -------------------------------------------------------
# 7. Save Results
# -------------------------------------------------------
# ذخیره مختصات دوبعدی

result = pd.DataFrame({
    "Sample": df["Sample"],
    "UMAP_1": embedding[:,0],
    "UMAP_2": embedding[:,1]
})
print("\nReduced Dataset")
print(result)

# -------------------------------------------------------
# 8. Visualization
# -------------------------------------------------------

# رسم نمودار دوبعدی
plt.figure(figsize=(7,6))
plt.scatter(
    embedding[:,0],
    embedding[:,1],
    s=120
)

# نمایش نام نمونه ها
for i, txt in enumerate(df["Sample"]):
    plt.text(
        embedding[i,0]+0.03,
        embedding[i,1]+0.03,
        txt,
        fontsize=12
    )

plt.title("UMAP Projection of Numerical Example")
plt.xlabel("UMAP Dimension 1")
plt.ylabel("UMAP Dimension 2")
plt.grid(True)
plt.show()