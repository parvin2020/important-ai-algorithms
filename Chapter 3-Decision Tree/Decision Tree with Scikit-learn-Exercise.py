# Decision Tree with Scikit-learn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    roc_auc_score,
    confusion_matrix
)

# --------------------------------------------------
# 1. Create Dataset
# --------------------------------------------------
data = pd.DataFrame({

    'Age': [
        'Young','Young','Young','Young','Young',
        'Old','Old','Old','Old','Old'
    ],

    'Income': [
        'Low','Low','Low','High','High',
        'Low','Low','High','High','High'
    ],

    'Target': [
        'No','No','Yes','Yes','Yes',
        'No','No','Yes','No','Yes'
    ]
})

# --------------------------------------------------
# 2. Convert Categorical Data to Numeric
# --------------------------------------------------
# جوان=0 ، مسن=1
data['Age'] = data['Age'].map({
    'Young':0,
    'Old':1
})

# کم=0 ، زیاد=1
data['Income'] = data['Income'].map({
    'Low':0,
    'High':1
})

# خیر=0 ، بله=1
data['Target'] = data['Target'].map({
    'No':0,
    'Yes':1
})

# --------------------------------------------------
# 3. Define Features and Target
# --------------------------------------------------
X = data[['Age', 'Income']]
y = data['Target']

# --------------------------------------------------
# 4. Train-Test Split
# --------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,
    test_size=0.30,      # 30 درصد داده برای تست
    random_state=42,     # تکرارپذیری نتایج
    stratify=y           # حفظ نسبت کلاس ها
)

# --------------------------------------------------
# 5. Create Decision Tree Model
# --------------------------------------------------
model = DecisionTreeClassifier(
    criterion='entropy',     # استفاده از Information Gain
    max_depth=3,             # حداکثر عمق درخت
    min_samples_split=2,     # حداقل نمونه برای تقسیم
    min_samples_leaf=1,      # حداقل نمونه در برگ
    max_features=None,       # استفاده از همه ویژگی ها
    max_leaf_nodes=10,       # حداکثر تعداد برگ ها
    splitter='best',         # بهترین تقسیم
    ccp_alpha=0.0,           # بدون هرس
    random_state=42
)

# --------------------------------------------------
# 6. Train Model
# --------------------------------------------------
model.fit(X_train, y_train)

# --------------------------------------------------
# 7. Prediction
# --------------------------------------------------
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:,1]

# --------------------------------------------------
# 8. Evaluation
# --------------------------------------------------
acc = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_prob)
cm = confusion_matrix(y_test, y_pred)
print("Decision Tree Results")
print("----------------------")
print(f"Accuracy : {acc:.4f}")
print(f"AUC-ROC  : {auc:.4f}")
print("\nConfusion Matrix")
print(cm)
print("\nClassification Report")
print(classification_report(y_test, y_pred))

# --------------------------------------------------
# 9. Tree Information
# --------------------------------------------------
print("\nTree Depth =", model.get_depth())
print("Number of Leaves =", model.get_n_leaves())

# --------------------------------------------------
# 10. Show Decision Rules
# --------------------------------------------------
from sklearn.tree import export_text
tree_rules = export_text(
    model,
    feature_names=['Age','Income']
)

print("\nDecision Rules")
print(tree_rules)