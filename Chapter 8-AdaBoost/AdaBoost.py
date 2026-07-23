from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

# داده های آموزشی
X = [
    [15, 2, 35],  # خوب=2
    [8, 0, 45],   # ضعیف=0
    [20, 2, 30],
    [10, 0, 50],
    [18, 2, 32],
    [9, 1, 40]    # متوسط=1
]

# خروجی ها
y = [1, 0, 1, 0, 1, 0]

# ایجاد مدل AdaBoost
model = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1),
    n_estimators=5,
    random_state=42
)

# آموزش مدل
model.fit(X, y)

# مشتری جدید
new_customer = [[12, 2, 38]]

# پیش بینی
prediction = model.predict(new_customer)

# نمایش نتیجه
if prediction[0] == 1:
    print("The customer pays the loan.")
else:
   print("The customer does not pay the loan.")
