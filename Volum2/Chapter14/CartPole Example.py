import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import random
import collections
import numpy as np  # ✅ اضافه شده برای رفع هشدار کندی

# --- ۱. تعریف شبکه عصبی (مغز ربات) ---
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        # یک شبکه ساده با یک لایه مخفی
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.out = nn.Linear(64, action_size) # خروجی: امتیاز حرکت چپ و راست

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.out(x)  # ✅ اصلاح شده: قبلاً self.out بود (بدون x)

# --- ۲. تعریف عامل هوشمند (Agent) ---
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        
        # تنظیمات مهم (Hyperparameters)
        self.memory = collections.deque(maxlen=10000) # حافظه تجربه
        self.gamma = 0.99    # ضریب اهمیت پاداش‌های آینده
        self.epsilon = 1.0   # نرخ اکتشاف (اول کاملاً تصادفی)
        self.epsilon_min = 0.01 # حداقل نرخ اکتشاف
        self.epsilon_decay = 0.995 # میزان کاهش تدریجی اکتشاف
        self.learning_rate = 0.001
        
        # ساخت شبکه اصلی و شبکه هدف
        self.q_network = QNetwork(state_size, action_size)
        self.target_network = QNetwork(state_size, action_size)
        self.target_network.load_state_dict(self.q_network.state_dict()) # کپی اولیه وزن‌ها
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
        self.loss_fn = nn.MSELoss() # تابع خطا (تفاوت بین پیش‌بینی و واقعیت)

    # انتخاب حرکت (اکتشاف یا بهره‌برداری)
    def select_action(self, state):
        # اگر عدد تصادفی کمتر از اپسیلون بود، حرکت تصادفی (اکتشاف)
        if random.random() < self.epsilon:
            return random.randrange(self.action_size)
        
        # در غیر این صورت، از مغز بپرس بهترین حرکت چیست (بهره‌برداری)
        with torch.no_grad(): # در زمان پیش‌بینی نیازی به محاسبه گرادیان نیست
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.q_network(state_tensor)
            return torch.argmax(q_values).item()

    # ذخیره تجربه در حافظه
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    # فرآیند یادگیری (مهم‌ترین بخش)
    def learn(self, batch_size=64):
        if len(self.memory) < batch_size:
            return # اگر حافظه خالی است، چیزی برای یادگیری نیست

        # انتخاب تصادفی یک دسته از حافظه
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # ✅ اصلاح شده: تبدیل به numpy array قبل از tensor (رفع هشدار کندی)
        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(np.array(actions)).unsqueeze(1)
        rewards = torch.FloatTensor(np.array(rewards))
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.FloatTensor(np.array(dones))

        # ۱. محاسبه پیش‌بینی شبکه اصلی (Q-value برای حرکات انجام شده)
        current_q_values = self.q_network(states).gather(1, actions).squeeze()

        # ۲. محاسبه هدف (Target) با استفاده از شبکه هدف
        with torch.no_grad():
            max_next_q_values = self.target_network(next_states).max(1)[0]
            # فرمول بلومن-بکمن: اگر بازی تمام شده پاداش، وگرنه پاداش + گاما * بهترین امتیاز بعدی
            target_q_values = rewards + (self.gamma * max_next_q_values * (1 - dones))

        # ۳. محاسبه خطا و به‌روزرسانی وزن‌ها (Backpropagation)
        loss = self.loss_fn(current_q_values, target_q_values)
        self.optimizer.zero_grad() # پاک کردن گرادیان‌های قبلی
        loss.backward() # محاسبه گرادیان‌ها
        self.optimizer.step() # به‌روزرسانی وزن‌های شبکه اصلی

        # کاهش نرخ اکتشاف (هرچه بیشتر یاد گرفت، کمتر تصادفی عمل کند)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    # کپی کردن وزن‌های شبکه اصلی به شبکه هدف (برای ثبات آموزش)
    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

# --- ۳. حلقه اصلی آموزش ---
def train():
    env = gym.make('CartPole-v1') # ساخت محیط بازی
    state_size = env.observation_space.shape[0] # ۴ ویژگی (موقعیت، سرعت، زاویه، سرعت زاویه‌ای)
    action_size = env.action_space.n # ۲ حرکت (چپ، راست)
    
    agent = DQNAgent(state_size, action_size)
    
    episodes = 50 # تعداد دفعات بازی
    batch_size = 64
    target_update_freq = 10 # هر ۱۰ مرحله شبکه هدف را آپدیت کن

    for episode in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            # env.render() # اگر می‌خواهید بازی را ببینید این خط را از کامنت خارج کنید
            
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated # اگر میله افتاد یا زمان تمام شد، بازی تمام است
            
            # ذخیره تجربه
            agent.remember(state, action, reward, next_state, float(done))
            
            # یادگیری از روی حافظه
            agent.learn(batch_size)
            
            state = next_state
            total_reward += reward
            
        # آپدیت شبکه هدف در فواصل مشخص
        if episode % target_update_freq == 0:
            agent.update_target_network()
            
        print(f"Episode: {episode+1}, Score: {total_reward}, Epsilon: {agent.epsilon:.2f}")
        
        # اگر امتیاز به حداکثر رسید (۵۰۰)، آموزش را متوقف کن
        if total_reward == 500:
            print("محیط با موفقیت حل شد!")
            break

if __name__ == "__main__":
    train()