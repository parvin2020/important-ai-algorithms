# K-Means Clustering with Scikit-learn
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score
)

# ============================================================
# 1. Create Data Frame
# ============================================================
# داده های مثال کتاب
data = {
    "Purchase_Count": [2,3,4,8,9,10],
    "Purchase_Amount": [3,4,2,7,8,7]
}

df = pd.DataFrame(data)
X = df.values

# ============================================================
# 2. Train-Test Split
# ============================================================

# در الگوریتم K-Means یادگیری بدون ناظر است
# بنابراین نیازی به Train/Test Split وجود ندارد.

# ============================================================
# 3. Train Model
# ============================================================

model = KMeans(
    n_clusters=2,          # تعداد خوشه ها
    init='k-means++',      # انتخاب هوشمند مراکز اولیه
    n_init='auto',         # تعداد اجرای الگوریتم با مراکز اولیه مختلف
    max_iter=300,          # حداکثر تعداد تکرار
    tol=1e-4,              # شرط توقف
    algorithm='lloyd',     # الگوریتم استاندارد KMeans
    random_state=42,       # تکرارپذیری نتایج
    copy_x=True,           # جلوگیری از تغییر داده های اصلی
    verbose=0              # عدم نمایش مراحل اجرا
)

model.fit(X)

# ============================================================
# 4. Predict
# ============================================================
labels = model.predict(X)

# ============================================================
# 5. Evaluate
# ============================================================

silhouette = silhouette_score(X, labels)
dbi = davies_bouldin_score(X, labels)
chi = calinski_harabasz_score(X, labels)
inertia = model.inertia_
centers = model.cluster_centers_
iterations = model.n_iter_
print("========== K-Means Results ==========")
print("Cluster Labels:")
print(labels)
print("------------------------------------")
print("Cluster Centers:")
print(centers)
print("------------------------------------")
print("Inertia :", inertia)
print("Silhouette Score :", silhouette)
print("Davies-Bouldin Index :", dbi)
print("Calinski-Harabasz Index :", chi)
print("Iterations :", iterations)