import random

LEFT, RIGHT = 0, 1                      # کدهای کنش

def step(s, a):                         # مدل محیط
    if s == 2: return 2, 0.0, True      # حالت پایانی
    if s == 0: return (1, -1.0, False) if a == RIGHT else (0, -1.0, False)
    if s == 1: return (2, 10.0, True) if a == RIGHT else (0, -1.0, False)

def choose_action(s, Q, eps):           # سیاست epsilon-greedy
    if random.random() < eps: return random.choice([LEFT, RIGHT])  # اکتشاف
    return RIGHT if Q[s][RIGHT] >= Q[s][LEFT] else LEFT            # بهره‌برداری

random.seed(7)                          # تکرارپذیری نتیجه
alpha, gamma, eps, episodes = 0.5, 0.9, 0.2, 1000   # پارامترهای یادگیری
Q = [[0.0, 0.0], [0.0, 0.0]]            # جدول Q برای حالت 0 و 1

for ep in range(episodes):              # حلقه اپیزودها
    s = 0                               # شروع از حالت اول
    a = choose_action(s, Q, eps)        # انتخاب کنش اول قبل از حلقه

    while s != 2:                       # تا رسیدن به هدف
        s2, r, done = step(s, a)        # اجرای کنش در محیط
        a2 = 0 if done else choose_action(s2, Q, eps)  # کنش بعدی واقعی (امضای SARSA)
        target = r if done else r + gamma * Q[s2][a2]  # هدف با Q(s',a') نه max
        Q[s][a] += alpha * (target - Q[s][a])          # به‌روزرسانی SARSA
        s, a = s2, a2                   # رفتن به جفت (حالت، کنش) بعدی

print("Final Q table:")
print("State 0: LEFT =", round(Q[0][LEFT], 2), ", RIGHT =", round(Q[0][RIGHT], 2))
print("State 1: LEFT =", round(Q[1][LEFT], 2), ", RIGHT =", round(Q[1][RIGHT], 2))
print("Learned policy:")
print("State 0 ->", "RIGHT" if Q[0][RIGHT] >= Q[0][LEFT] else "LEFT")
print("State 1 ->", "RIGHT" if Q[1][RIGHT] >= Q[1][LEFT] else "LEFT")