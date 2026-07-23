# کتابخانه های موردنیاز
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score

# داده های آموزشی
X=np.array([1,2,3,4,5]).reshape(-1,1)
y=np.array([52,55,61,66,72])

# تعریف مدل
model=LinearRegression(
    fit_intercept=True, # محاسبه عرض از مبدا
    copy_X=True, # حفظ داده های اصلی
    n_jobs=-1, # استفاده از همه هسته ها
    positive=False # اجازه ضرایب منفی
)

# آموزش مدل
model.fit(X,y)

# پیش بینی
y_pred=model.predict(X)

# استخراج پارامترهای مدل
slope=model.coef_[0]
intercept=model.intercept_

# محاسبه شاخص های ارزیابی
mse=mean_squared_error(y,y_pred)
rmse=np.sqrt(mse)
mae=mean_absolute_error(y,y_pred)
r2=r2_score(y,y_pred)

print(f"Slope = {slope:.2f}")
print(f"Intercept = {intercept:.2f}")
print("Prediction=",[f"{x:.2f}" for x in y_pred])
print(f"MSE = {mse:.2f}")
print(f"RMSE = {rmse:.2f}")
print(f"MAE = {mae:.2f}")
print(f"R2 = {r2:.2f}")