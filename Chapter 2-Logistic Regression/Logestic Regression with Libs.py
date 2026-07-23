import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import log_loss
from sklearn.metrics import confusion_matrix

# داده های آموزشی
X=np.array([2,3,4,5,6,7]).reshape(-1,1)
y=np.array([0,0,0,1,1,1])

# تعریف مدل
model=LogisticRegression(
    penalty='l2', # نوع منظم سازی
    C=1.0, # شدت منظم سازی
    solver='lbfgs', # الگوریتم بهینه سازی
    max_iter=1000, # حداکثر تکرار
    fit_intercept=True, # محاسبه بایاس
    class_weight=None, # وزن کلاس ها
    random_state=42, # تکرارپذیری
    warm_start=False, # شروع مجدد آموزش
    multi_class='auto' # انتخاب خودکار حالت چندکلاسه
)

# آموزش مدل
model.fit(X,y)

# احتمال ها
probabilities=model.predict_proba(X)

# پیش بینی کلاس
predictions=model.predict(X)

# ضرایب
print("Intercept =",model.intercept_)
print("Coefficients =",model.coef_)

# ماتریس درهم ریختگی
cm=confusion_matrix(y,predictions)

# شاخص ها
accuracy=accuracy_score(y,predictions)
precision=precision_score(y,predictions)
recall=recall_score(y,predictions)
f1=f1_score(y,predictions)
loss=log_loss(y,probabilities)

# نمایش نتایج
print("Confusion Matrix =")
print(cm)

print("Accuracy =",accuracy)
print("Precision =",precision)
print("Recall =",recall)
print("F1 Score =",f1)
print("Log Loss =",loss)