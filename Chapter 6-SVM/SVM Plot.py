# ============================================================
# پیاده‌سازی ماشین بردار پشتیبان (SVM) با scikit-learn
# ============================================================

import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC

# تعریف داده‌ها و برچسب‌ها
X = np.array([[1, 2], [2, 2], [1, 0], [2, 0]])
y = np.array([1, 1, -1, -1])

# ساخت و آموزش مدل با حاشیه سخت (C بزرگ)
svm_model = SVC(kernel='linear', C=1e6).fit(X, y)

# استخراج پارامترهای مدل
w = svm_model.coef_[0]
b = svm_model.intercept_[0]

# پیش‌بینی نقطه جدید
Z = np.array([[1.5, 1.8]])
pred = svm_model.predict(Z)[0]
dec_val = svm_model.decision_function(Z)[0]

# رسم نمودار
fig, ax = plt.subplots(figsize=(8, 6))

# رسم نقاط داده
ax.scatter(X[y == 1, 0], X[y == 1, 1], s=150, c='green', marker='o', 
           edgecolors='k', label='Class +1')
ax.scatter(X[y == -1, 0], X[y == -1, 1], s=150, c='red', marker='s', 
           edgecolors='k', label='Class -1')

# برجسته‌سازی بردارهای پشتیبان
sv = svm_model.support_vectors_
ax.scatter(sv[:, 0], sv[:, 1], s=300, facecolors='none', 
           edgecolors='k', linewidths=2, label='Support Vectors')

# رسم خطوط مرزی و حاشیه‌ها
x1_grid = np.linspace(0, 3, 100)
x2_decision = -(w[0] * x1_grid + b) / w[1]
x2_margin_pos = (1 - w[0] * x1_grid - b) / w[1]
x2_margin_neg = (-1 - w[0] * x1_grid - b) / w[1]

ax.plot(x1_grid, x2_decision, 'k-', linewidth=2, label='Decision Boundary')
ax.plot(x1_grid, x2_margin_pos, 'k--', linewidth=1.5, label='Positive Margin')
ax.plot(x1_grid, x2_margin_neg, 'k--', linewidth=1.5, label='Negative Margin')

# رسم نقطه جدید
ax.scatter(Z[0, 0], Z[0, 1], s=200, c='blue', marker='*', 
           edgecolors='k', label=f'Point Z (pred: {pred})')

# تنظیمات نمودار
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_title('SVM Linear Classifier')
ax.legend(loc='best')
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 3)
ax.set_ylim(-1, 3)

# نمایش پارامترها روی نمودار
textstr = f'w = [{w[0]:.2f}, {w[1]:.2f}]\nb = {b:.2f}\nMargin = {2/np.linalg.norm(w):.2f}'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=11,
        verticalalignment='top', bbox=props)

plt.tight_layout()
plt.show()
