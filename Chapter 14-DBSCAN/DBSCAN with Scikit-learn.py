# DBSCAN with Scikit-learn

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import DBSCAN
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score
from sklearn.metrics import davies_bouldin_score
from sklearn.metrics import calinski_harabasz_score

# ---------------------------------------------------
# 1. Create data frame
# ایجاد مجموعه داده
# ---------------------------------------------------

data = np.array([
    [1,1],
    [2,1],
    [2,2],
    [8,8],
    [8,9],
    [25,25]
])

df = pd.DataFrame(data, columns=["X","Y"])

print("Input Data")
print(df)

# ---------------------------------------------------
# 2. Train-test split
# تقسیم داده (فقط برای رعایت قالب استاندارد)
# در DBSCAN نیازی به Train/Test نیست.
# ---------------------------------------------------

X_train, X_test = train_test_split(
    df,
    test_size=0.20,
    random_state=42
)

# ---------------------------------------------------
# 3. Train model
# آموزش مدل
# ---------------------------------------------------

model = DBSCAN(

    eps=1.5,                 # شعاع همسایگی

    min_samples=3,           # حداقل تعداد همسایه

    metric='euclidean',      # معیار فاصله

    algorithm='auto',        # روش جستجوی همسایه

    leaf_size=30,            # اندازه برگ در KDTree/BallTree

    p=None,                  # توان فاصله Minkowski

    n_jobs=-1                # استفاده از تمام هسته‌های CPU
)

model.fit(df)

# ---------------------------------------------------
# 4. Predict
# پیش‌بینی خوشه‌ها
# ---------------------------------------------------

labels = model.labels_

df["Cluster"] = labels

print("\nCluster Labels")
print(df)

# ---------------------------------------------------
# نمایش تعداد خوشه‌ها
# ---------------------------------------------------

number_of_clusters = len(set(labels)) - (1 if -1 in labels else 0)

print("\nNumber of clusters :", number_of_clusters)

# ---------------------------------------------------
# نمایش تعداد نویزها
# ---------------------------------------------------

noise_points = list(labels).count(-1)

print("Number of noise points :", noise_points)

# ---------------------------------------------------
# نمایش Core Samples
# ---------------------------------------------------

core_index = model.core_sample_indices_

print("Core sample indices :")
print(core_index)

# ---------------------------------------------------
# 5. Evaluate
# ارزیابی مدل
# ---------------------------------------------------

# این شاخص‌ها فقط زمانی قابل محاسبه هستند
# که حداقل دو خوشه تشکیل شده باشد.

if number_of_clusters >= 2:

    sil = silhouette_score(df[["X","Y"]], labels)

    db = davies_bouldin_score(df[["X","Y"]], labels)

    ch = calinski_harabasz_score(df[["X","Y"]], labels)

    print("\nSilhouette Score :", sil)

    print("Davies-Bouldin Index :", db)

    print("Calinski-Harabasz Index :", ch)

else:

    print("\nSilhouette Score : Not Available")

    print("Davies-Bouldin Index : Not Available")

    print("Calinski-Harabasz Index : Not Available")

# ---------------------------------------------------
# رسم خوشه‌ها
# ---------------------------------------------------

plt.figure(figsize=(7,6))

plt.scatter(
    df["X"],
    df["Y"],
    c=labels,
    s=120
)

for i in range(len(df)):
    plt.text(
        df.iloc[i,0]+0.2,
        df.iloc[i,1]+0.2,
        str(i)
    )

plt.xlabel("X")

plt.ylabel("Y")

plt.title("DBSCAN Clustering")

plt.grid(True)

plt.show()