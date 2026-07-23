# داده های آموزشی
x=[1,2,3,4,5]
y=[52,55,61,66,72]

# محاسبه تعداد نمونه ها
n=len(x)

# محاسبه میانگین ها
x_mean=sum(x)/n
y_mean=sum(y)/n

# محاسبه صورت و مخرج فرمول شیب
numerator=0
denominator=0
for i in range(n):
    numerator+=(x[i]-x_mean)*(y[i]-y_mean)
    denominator+=(x[i]-x_mean)**2

# محاسبه شیب و عرض از مبدا
b1=numerator/denominator
b0=y_mean-b1*x_mean

print(f"Slope = {b1:.2f}")
print(f"Intercept = {b0:.2f}")

# محاسبه مقادیر پیش بینی شده
y_pred=[]
for i in range(n):
    pred=b0+b1*x[i]
    y_pred.append(pred)

print([f"{x:.2f}" for x in y_pred])
# محاسبه SSE
sse=0
for i in range(n):
    sse+=(y[i]-y_pred[i])**2

# محاسبه MSE
mse=sse/n

# محاسبه RMSE
rmse=mse**0.5

# محاسبه MAE
mae=0
for i in range(n):
    mae+=abs(y[i]-y_pred[i])
mae=mae/n

# محاسبه R2
ss_total=0
for i in range(n):
    ss_total+=(y[i]-y_mean)**2

r2=1-(sse/ss_total)

print(f"SSE = {sse:.2f}")
print(f"MSE = {mse:.2f}")
print(f"RMSE = {rmse:.2f}")
print(f"MAE = {mae:.2f}")
print(f"R2 = {r2:.2f}")