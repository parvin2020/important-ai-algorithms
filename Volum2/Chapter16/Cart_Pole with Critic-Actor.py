
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt

# --- 1. تنظیمات محیط ---
env = gym.make("CartPole-v1")
state_size = env.observation_space.shape[0]  # تعداد ویژگی‌های حالت (4 تا)
'''
1.	موقعیت افقی ارابه (Cart Position): ارابه در کجای ریل قرار دارد؟ (مثلاً بین ۲.۴- تا ۲.۴+)
2.	سرعت افقی ارابه (Cart Velocity): ارابه با چه سرعتی به چپ یا راست حرکت می‌کند؟
3.	زاویه میله (Pole Angle): میله چقدر از حالت عمود کج شده است؟ (مثلاً بین ۰.۴۱- تا ۰.۴۱+ رادیان)
4.	سرعت زاویه‌ای میله (Pole Angular Velocity): میله با چه سرعتی در حال چرخش و سقوط است؟
ارتباط با شبکه عصبی: چون محیط ۴ عدد به ما می‌دهد، لایه ورودی (Input Layer) در شبکه‌های عصبی Actor و Critic باید دقیقاً ۴ نورون داشته باشد تا این ۴ ویژگی را دریافت کند.

'''
action_size = env.action_space.n             # تعداد کنش‌های ممکن (2 تا)
'''
عامل فقط ۲ انتخاب برای اعمال نیرو به ارابه دارد:
•	کنش ۰ (Action 0): اعمال نیرو به سمت چپ (Push left)
•	کنش ۱ (Action 1): اعمال نیرو به سمت راست (Push right)
(نکته: در این محیط نمی‌توانید نیرو وارد نکنید یا نیروی متغیر وارد کنید؛ فقط می‌توانید یک نیروی ثابت را به چپ یا راست اعمال کنید).
ارتباط با شبکه عصبی: چون محیط فقط ۲ کنش را می‌پذیرد، لایه خروجی شبکه Actor باید دقیقاً ۲ نورون داشته باشد. این دو نورون، احتمالِ انتخابِ کنشِ چپ و احتمالِ انتخابِ کنشِ راست را (با استفاده از Softmax) محاسبه و به محیط برمی‌گردانند.

'''
# --- 2. تعریف شبکه‌های عصبی ---

# نام کلاس‌ها با حروف بزرگ شروع شد تا با متغیرها تداخل پیدا نکنند
class Actor(nn.Module): 
    def __init__(self, state_size, action_size): 
        super().__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, action_size)
        
    def forward(self, state): 
        x = F.relu(self.fc1(state))
        # خروجی لایه آخر را به صورت Logits برمی‌گردانیم (بدون Softmax)
        # این کار پایداری عددی را در محاسبات احتمالات افزایش می‌دهد
        logits = self.fc2(x) 
        return logits

class Critic(nn.Module): 
    def __init__(self, state_size): 
        super().__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 1) # خروجی یک عددscalar برای ارزش حالت
        
    def forward(self, state): 
        x = F.relu(self.fc1(state))
        value = self.fc2(x)
        return value

# ساخت نمونه‌هایی از شبکه‌ها (با نام‌های متفاوت از نام کلاس)
actor_net = Actor(state_size, action_size)
critic_net = Critic(state_size)

# تعریف بهینه‌سازها (Optimizer)
actor_optimizer = optim.Adam(actor_net.parameters(), lr=0.001)
critic_optimizer = optim.Adam(critic_net.parameters(), lr=0.005)

# --- 3. تابع انتخاب کنش ---
def select_action(state): 
    state_tensor = torch.tensor(state, dtype=torch.float32)
    logits = actor_net(state_tensor)
    # استفاده از logits برای ساخت توزیع احتمالات (ایمن‌تر از probs)
    distribution = torch.distributions.Categorical(logits=logits)
    action = distribution.sample()
    # بازگشت خودِ کنش و لگاریتم احتمال آن (برای محاسبه گرادیان Actor)
    return action.item(), distribution.log_prob(action)

# --- 4. حلقه اصلی آموزش ---
episodes = 500
scores = []      # برای ذخیره امتیاز هر اپیزود جهت رسم نمودار
gamma = 0.99     # ضریب تخفیف

for episode in range(episodes): 
    state, _ = env.reset()
    total_reward = 0
    done = False
    
    while not done: 
        state_tensor = torch.tensor(state, dtype=torch.float32)
        
        # دریافت ارزش حالت فعلی از Critic
        value = critic_net(state_tensor)
        
        # انتخاب کنش و دریافت لگاریتم احتمال آن از Actor
        action, log_prob = select_action(state)
        
        # اجرای کنش در محیط
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_reward += reward
        
        next_state_tensor = torch.tensor(next_state, dtype=torch.float32)
        
        # --- محاسبه هدف TD و مزیت (Advantage) ---
        if done:
            # اگر اپیزود تمام شده، ارزش حالت بعدی صفر است
            td_target = reward 
        else:
            # در غیر این صورت، از تخمین Critic برای حالت بعدی استفاده می‌کنیم (Bootstrapping)
            next_value = critic_net(next_state_tensor)
            td_target = reward + gamma * next_value.item()
            
        # محاسبه خطای TD (Advantage): چقدر تخمین ما با واقعیت جدید فاصله دارد؟
        advantage = td_target - value.item()
        
        # --- آپدیت شبکه Critic (کاهش خطای تخمین ارزش) ---
        # تابع هزینه: میانگین مربعات خطا (MSE) بین تخمین فعلی و هدف TD
        critic_loss = F.mse_loss(value, torch.tensor([td_target], dtype=torch.float32))
        critic_optimizer.zero_grad()
        critic_loss.backward()
        critic_optimizer.step()
        
        # --- آپدیت شبکه Actor (تقویت کنش‌های خوب، تضعیف کنش‌های بد) ---
        # تابع هزینه: منفیِ لگاریتم احتمال ضربدر مزیت (Advantage)
        # اگر advantage مثبت باشد، گرادیان باعث افزایش احتمال این کنش می‌شود
        actor_loss = -log_prob * advantage 
        actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_optimizer.step()
        
        # حرکت به حالت بعدی
        state = next_state

    # ذخیره امتیاز کل این اپیزود برای رسم نمودار
    scores.append(total_reward)
    
    # چاپ پیشرفت هر 50 اپیزود
    if (episode + 1) % 50 == 0:
        print(f"Episode {episode + 1} | Total Reward: {total_reward}")

# --- 5. رسم نمودار نتایج ---
plt.figure(figsize=(10, 5))
plt.plot(scores, color='blue', alpha=0.7)
# رسم میانگین متحرک (Moving Average) برای هموار کردن نمودار
moving_avg = [sum(scores[i-10:i])/10 for i in range(10, len(scores))]
plt.plot(range(10, len(scores)), moving_avg, color='red', linewidth=2, label='Moving Avg (10 ep)')

plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Actor-Critic (TD) Training on CartPole-v1")
plt.legend()
plt.grid(True)
plt.show()