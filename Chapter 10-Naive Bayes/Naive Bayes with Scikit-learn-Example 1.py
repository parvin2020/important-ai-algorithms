# Naive Bayes with Scikit-learn

import numpy as np
import pandas as pd

# کتابخانه‌های مورد نیاز
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

#-------------------------------------------------------
# 1. Create data frame
#-------------------------------------------------------

# مطالعه:
# زیاد = 1
# کم = 0

# حضور:
# بله = 1
# خیر = 0

# نتیجه:
# قبول = 1
# رد = 0

data = pd.DataFrame({

    "Study":[1,1,1,0,0,0,0,1,0,0],

    "Attendance":[1,1,0,1,1,0,0,0,0,1],

    "Result":[1,1,1,1,1,0,0,0,0,0]

})

X = data[["Study","Attendance"]]
y = data["Result"]

#-------------------------------------------------------
# 2. Train-test split
#-------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.30,      # 30 درصد داده برای آزمون

    random_state=42,      # تکرارپذیری نتایج

    stratify=y            # حفظ نسبت کلاس‌ها

)

#-------------------------------------------------------
# 3. Train model
#-------------------------------------------------------

model = GaussianNB(

    var_smoothing=1e-9    # جلوگیری از تقسیم بر صفر

)

model.fit(X_train,y_train)

#-------------------------------------------------------
# 4. Predict
#-------------------------------------------------------

y_pred = model.predict(X_test)

# پیش‌بینی یک دانشجوی جدید

new_student = [[1,1]]

prediction = model.predict(new_student)

#-------------------------------------------------------
# 5. Evaluate
#-------------------------------------------------------

acc = accuracy_score(y_test,y_pred)

pre = precision_score(y_test,y_pred)

rec = recall_score(y_test,y_pred)

f1 = f1_score(y_test,y_pred)

cm = confusion_matrix(y_test,y_pred)

report = classification_report(y_test,y_pred)

print("Accuracy :",acc)

print("Precision :",pre)

print("Recall :",rec)

print("F1 Score :",f1)

print("\nConfusion Matrix")

print(cm)

print("\nClassification Report")

print(report)

if prediction[0]==1:

    print("\nPrediction : Pass")

else:

    print("\nPrediction : Fail")