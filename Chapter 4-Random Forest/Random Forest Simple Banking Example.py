# Random Forest Simple Banking Example
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)
# -------------------------------------
# داده های آموزشی
# -------------------------------------

data = pd.DataFrame({
    'Income':[1,1,1,0,0,0],
    'Credit':[1,1,0,1,0,0],
    'CardApproval':[1,1,1,1,0,0]
})

# -------------------------------------
# ویژگی ها و هدف
# -------------------------------------
X = data[['Income','Credit']]
y = data['CardApproval']

# -------------------------------------
# مدل جنگل تصادفی
# -------------------------------------
model = RandomForestClassifier(
    # تعداد درخت ها
    n_estimators=3,
    # عمق درخت
    max_depth=2,
    # معیار جینی
    criterion='gini',
    # ویژگی های تصادفی
    max_features=1,
    random_state=42
)
# -------------------------------------
# آموزش مدل
# -------------------------------------
model.fit(X,y)

# -------------------------------------
# مشتری جدید
# -------------------------------------
new_customer = [[1,0]]
prediction = model.predict(new_customer)
probability = model.predict_proba(new_customer)

# -------------------------------------
# نمایش نتایج
# -------------------------------------
print("Prediction =", prediction[0])
print("Probability =")
print(probability)

# -------------------------------------
# ارزیابی روی داده آموزشی
# -------------------------------------
y_pred = model.predict(X)

acc = accuracy_score(y,y_pred)
print("\nAccuracy =", acc)
print("\nConfusion Matrix")
print(confusion_matrix(y,y_pred))
print("\nClassification Report")
print(classification_report( y,y_pred))