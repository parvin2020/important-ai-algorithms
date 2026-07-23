# t-SNE with Scikit-learn

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.datasets import load_iris
from sklearn.metrics import silhouette_score

#---------------------------------------------------
# 1. Load data set or Create data frame
# ایجاد و بارگذاری مجموعه داده
#---------------------------------------------------
data = load_iris()
X = data.data
y = data.target
df = pd.DataFrame(
    X,
    columns=data.feature_names
)
print("Original Dataset")
print(df.head())


#---------------------------------------------------
# استانداردسازی داده ها
# آماده سازی داده قبل از اجرای t-SNE
#---------------------------------------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

#---------------------------------------------------
# 2. Train-test split
# در t-SNE تقسیم آموزش و آزمون وجود ندارد
# زیرا الگوریتم برای تجسم داده استفاده می شود
#---------------------------------------------------

X_train = X_scaled

#---------------------------------------------------
# 3. Train model
# اجرای الگوریتم t-SNE
#---------------------------------------------------

tsne = TSNE(
    n_components=2,        
    # تعداد ابعاد خروجی (معمولاً 2 یا 3)
    perplexity=30,
    # تعداد تقریبی همسایه های موثر هر نقطه
    learning_rate="auto",
    # نرخ یادگیری الگوریتم
    n_iter=1000,
    # تعداد تکرارهای بهینه سازی
    init="pca",
    # مقداردهی اولیه با PCA

    metric="euclidean",
    # معیار فاصله بین نمونه ها
    random_state=42,
    # برای تکرارپذیری نتایج
    method="barnes_hut",
    # روش سریع محاسبه برای داده های بزرگ
    angle=0.5
    # پارامتر سرعت/دقت در Barnes-Hut
)

X_tsne = tsne.fit_transform(X_train)
print("\nEmbedding Result")

print(X_tsne[:5])

#---------------------------------------------------
# 4. Predict
# t-SNE متد Predict ندارد
# برای داده جدید باید دوباره الگوریتم اجرا شود
#---------------------------------------------------


#---------------------------------------------------
# 5. Evaluate with suitable items
# ارزیابی کیفیت embedding
#---------------------------------------------------
sil_score = silhouette_score(X_tsne,y)
print("\nSilhouette Score")
print(sil_score)

#---------------------------------------------------
# رسم نتیجه کاهش ابعاد
#---------------------------------------------------
plt.figure(figsize=(8,6))
scatter = plt.scatter(
    X_tsne[:,0],
    X_tsne[:,1],
    c=y,
    s=60
)

plt.title( "t-SNE Visualization")
plt.xlabel( "Component 1")
plt.ylabel( "Component 2")
plt.grid(True)
plt.show()