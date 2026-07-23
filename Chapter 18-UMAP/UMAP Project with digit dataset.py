import numpy as np
import umap
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

digits = load_digits()
data = digits.data
scaler = StandardScaler()
data = scaler.fit_transform(data)

# Apply UMAP
reducer = umap.UMAP()
embedding = reducer.fit_transform(data)
plt.scatter(embedding[:, 0], embedding[:, 1], c=digits.target, cmap='Spectral', s=5)
plt.colorbar(boundaries=np.arange(11)-0.5).set_ticks(np.arange(10))
plt.show()