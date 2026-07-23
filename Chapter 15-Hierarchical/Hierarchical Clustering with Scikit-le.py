# Hierarchical Clustering with Scikit-learn

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score
)

from scipy.cluster.hierarchy import dendrogram, linkage

#---------------------------------------------------
# 1. Load data set or Create data frame
# ایجاد مجموعه داده فرضی
#---------------------------------------------------
data = {
    "Feature1": [2,3,8,9,8],
    "Feature2": [3,4,7,8,9]
}
df = pd.DataFrame(data)
print("Dataset")
print(df)

#---------------------------------------------------
# 2. Train-test split
# در الگوریتم های خوشه بندی تقسیم داده به آموزش و آزمون وجود ندارد
# زیرا یادگیری بدون ناظر است.
#---------------------------------------------------
X = df.values

#---------------------------------------------------
# 3. Train model
#---------------------------------------------------
model = AgglomerativeClustering(
    n_clusters=2,              # تعداد خوشه های نهایی
    metric='euclidean',        # معیار محاسبه فاصله
    linkage='ward',            # روش اتصال خوشه ها
                               # Ward فقط با Euclidean سازگار است
    compute_distances=True     # ذخیره فاصله ها برای رسم دندروگرام
)

labels = model.fit_predict(X)
print("\nCluster Labels")
print(labels)

#---------------------------------------------------
# 4. Predict
# در AgglomerativeClustering تابع predict وجود ندارد.
# خروجی fit_predict همان برچسب خوشه ها است.
#---------------------------------------------------

df["Cluster"] = labels
print("\nFinal Result")
print(df)

#---------------------------------------------------
# 5. Evaluate with suitable items
#---------------------------------------------------
sil = silhouette_score(X, labels)
db = davies_bouldin_score(X, labels)
ch = calinski_harabasz_score(X, labels)
print("\nSilhouette Score =", sil)
print("Davies-Bouldin Index =", db)
print("Calinski-Harabasz Index =", ch)

#---------------------------------------------------
# Scatter Plot
#---------------------------------------------------
plt.figure(figsize=(6,5))

plt.scatter(
    df["Feature1"],
    df["Feature2"],
    c=df["Cluster"],
    s=150
)

plt.xlabel("Feature1")
plt.ylabel("Feature2")
plt.title("Hierarchical Clustering")
plt.grid(True)
plt.show()

#---------------------------------------------------
# Dendrogram
#---------------------------------------------------
Z = linkage( X, method='ward')
plt.figure(figsize=(8,5))
dendrogram(Z)
plt.title("Hierarchical Clustering Dendrogram")
plt.xlabel("Samples")
plt.ylabel("Distance")
plt.show()