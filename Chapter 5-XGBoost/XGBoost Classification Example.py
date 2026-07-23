# XGBoost Banking Example
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)
from xgboost import XGBClassifier
# --------------------------------------------------
# 1. Create Banking Dataset
# --------------------------------------------------
# Income : Monthly income (thousand dollars)
# Age    : Customer age
# LoanPaid : 1=Paid Loan , 0=Default
data = pd.DataFrame({
    'Income':[20,25,30,35,40,45,50,55,60,65,
              70,75,80,85,90,95],
    'Age':[22,25,27,30,32,35,38,40,
           42,45,48,50,52,55,58,60],
    'LoanPaid':[0,0,0,0,0,1,1,1,
                1,1,1,1,1,1,1,1]
})
# --------------------------------------------------
# 2. Define Features and Target
# --------------------------------------------------
X = data[['Income','Age']]
y = data['LoanPaid']

# --------------------------------------------------
# 3. Train-Test Split
# --------------------------------------------------
X_train, X_test, y_train, y_test = \
    train_test_split(
        X,
        y,
        test_size=0.25,      # نسبت داده تست
        random_state=42,     # تکرارپذیری
        stratify=y           # حفظ نسبت کلاس ها
)

# --------------------------------------------------
# 4. Create XGBoost Model
# --------------------------------------------------
model = XGBClassifier(
    # تعداد درخت ها
    n_estimators=100,
    # نرخ یادگیری
    learning_rate=0.1,
    # حداکثر عمق هر درخت
    max_depth=3,
    # حداقل وزن گره فرزند
    min_child_weight=1,
    # حداقل کاهش خطا برای تقسیم
    gamma=0,
    # درصد نمونه های آموزشی
    subsample=0.8,
    # درصد ویژگی های مورد استفاده
    colsample_bytree=0.8,
    # منظم سازی L1
    reg_alpha=0,
    # منظم سازی L2
    reg_lambda=1,
    # تابع هدف طبقه بندی دودویی
    objective='binary:logistic',
    # معیار ارزیابی
    eval_metric='logloss',
    random_state=42
)
# --------------------------------------------------
# 5. Train Model
# --------------------------------------------------
model.fit(X_train, y_train)
# --------------------------------------------------
# 6. Prediction on Test Data
# --------------------------------------------------
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:,1]
# --------------------------------------------------
# 7. Evaluation Metrics
# --------------------------------------------------
acc = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)
cm = confusion_matrix(y_test, y_pred)

# --------------------------------------------------
# 8. Display Results
# --------------------------------------------------
print("\nXGBoost Banking Results")
print("-------------------------")
print(f"Accuracy  : {acc:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")
print(f"AUC-ROC   : {auc:.4f}")
print("\nConfusion Matrix")
print(cm)
print("\nClassification Report")
print(classification_report(y_test, y_pred))

# --------------------------------------------------
# 9. Feature Importance
# --------------------------------------------------
print("\nFeature Importance")
for feature, importance in zip(
        X.columns,
        model.feature_importances_
    ):
    print(
        f"{feature} : {importance:.4f}"
    )

# --------------------------------------------------
# 10. New Customer Analysis
# --------------------------------------------------
# مشتری جدید:
# Income = 50
# Age = 40
new_customer = pd.DataFrame({
    'Income':[50],
    'Age':[40]
})
# پیش بینی کلاس

prediction = model.predict(new_customer)
# احتمال کلاس ها
probability = model.predict_proba(new_customer)

# --------------------------------------------------
# 11. Interpret Results
# --------------------------------------------------
repayment_probability = probability[0][1]

default_probability =  probability[0][0]

print("\nNew Customer Analysis")
print("-------------------------")
print(f"Repayment Probability : " f"{repayment_probability*100:.2f}%")
print(f"Default Probability   : " f"{default_probability*100:.2f}%")
if prediction[0] == 1:
    print("Decision: Customer is expected to repay the loan" )
else:
    print( "Decision: Customer is expected to default on the loan" )