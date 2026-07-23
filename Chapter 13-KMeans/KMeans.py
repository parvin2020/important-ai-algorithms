import numpy as np
from sklearn.cluster import KMeans

# داده‌ها
X = np.array([
    [12,1],
    [11,2],
    [10,3],
    [4,20],
    [3,25],
    [2,30]
])

# مدل KMeans
kmeans = KMeans(
    n_clusters=2,
    random_state=42
)

# آموزش مدل
kmeans.fit(X)

# برچسب خوشه‌ها
labels = kmeans.labels_

# centroidها
centroids = kmeans.cluster_centers_

print("Every Customer Label:")
print(labels)

print("\nCentroids :")
print(centroids)
