import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import random
import collections
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# --- ۱. تعریف شبکه عصبی ---
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.out = nn.Linear(64, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.out(x)

# --- ۲. تعریف عامل هوشمند ---
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = collections.deque(maxlen=10000)
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        
        self.q_network = QNetwork(state_size, action_size)
        self.target_network = QNetwork(state_size, action_size)
        self.target_network.load_state_dict(self.q_network.state_dict())
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.learning_rate)
        self.loss_fn = nn.MSELoss()

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.action_size)
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = self.q_network(state_tensor)
            return torch.argmax(q_values).item()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def learn(self, batch_size=64):
        if len(self.memory) < batch_size:
            return
        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(np.array(states))
        actions = torch.LongTensor(np.array(actions)).unsqueeze(1)
        rewards = torch.FloatTensor(np.array(rewards))
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.FloatTensor(np.array(dones))

        current_q_values = self.q_network(states).gather(1, actions).squeeze()

        with torch.no_grad():
            max_next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (self.gamma * max_next_q_values * (1 - dones))

        loss = self.loss_fn(current_q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

# --- ۳. کلاس نمایش گرافیکی ---
class CartPoleVisualizer:
    def __init__(self):
        # ایجاد پنجره با دو زیرپنجره
        self.fig = plt.figure(figsize=(10, 8))
        
        # زیرپنجره بالا: نمایش CartPole
        self.ax1 = self.fig.add_subplot(211)
        self.ax1.set_xlim(-3, 3)
        self.ax1.set_ylim(-0.5, 2)
        self.ax1.set_title(' CartPole Environment', fontsize=14, fontweight='bold')
        self.ax1.grid(True, alpha=0.3)
        
        # رسم زمین (خط پایه)
        self.ax1.axhline(y=0, color='black', linewidth=2)
        
        # ایجاد اشکای گاری و میله
        self.cart = patches.Rectangle((-0.3, -0.15), 0.6, 0.3, 
                                      linewidth=2, edgecolor='blue', facecolor='lightblue')
        self.pole = patches.FancyArrowPatch((0, 0), (0, 1),
                                            arrowstyle='-', mutation_scale=20,
                                            linewidth=4, color='red')
        self.ax1.add_patch(self.cart)
        self.ax1.add_patch(self.pole)
        
        # متن اطلاعات
        self.info_text = self.ax1.text(0.02, 0.95, '', transform=self.ax1.transAxes,
                                       fontsize=11, verticalalignment='top',
                                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # زیرپنجره پایین: نمودار امتیاز
        self.ax2 = self.fig.add_subplot(212)
        self.ax2.set_xlabel('Episode', fontsize=12)
        self.ax2.set_ylabel('Grade', fontsize=12)
        self.ax2.set_title(' Learning Grapgh', fontsize=14, fontweight='bold')
        self.ax2.grid(True, alpha=0.3)
        
        self.scores = []
        self.episodes = []
        self.line, = self.ax2.plot([], [], 'b-', linewidth=2, label='امتیاز')
        self.ax2.legend(loc='upper left')
        
        plt.tight_layout()
        plt.ion()  # فعال کردن حالت تعاملی
        
    def update_cartpole(self, state):
        """به‌روزرسانی موقعیت گاری و میله"""
        cart_pos, cart_vel, pole_angle, pole_vel = state
        
        # به‌روزرسانی موقعیت گاری
        self.cart.set_xy((cart_pos - 0.3, -0.15))
        
        # به‌روزرسانی موقعیت میله
        pole_length = 1.0
        pole_end_x = cart_pos + pole_length * np.sin(pole_angle)
        pole_end_y = pole_length * np.cos(pole_angle)
        self.pole.set_positions((cart_pos, 0), (pole_end_x, pole_end_y))
        
        # به‌روزرسانی متن اطلاعات
        self.info_text.set_text(
            f'موقعیت گاری: {cart_pos:.2f}\n'
            f'زاویه میله: {np.degrees(pole_angle):.1f}°'
        )
        
    def update_chart(self, episode, score):
        """به‌روزرسانی نمودار امتیاز"""
        self.episodes.append(episode)
        self.scores.append(score)
        
        self.line.set_data(self.episodes, self.scores)
        self.ax2.set_xlim(0, max(1, len(self.episodes)))
        self.ax2.set_ylim(0, max(1, max(self.scores) * 1.1))
        
    def draw(self):
        """رسم نهایی"""
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.001)
    
    def close(self):
        plt.close()

# --- ۴. حلقه اصلی آموزش با نمایش گرافیکی ---
def train():
    env = gym.make('CartPole-v1')
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    agent = DQNAgent(state_size, action_size)
    visualizer = CartPoleVisualizer()
    
    episodes = 300
    batch_size = 64
    target_update_freq = 10

    try:
        for episode in range(episodes):
            state, _ = env.reset()
            total_reward = 0
            done = False
            
            while not done:
                action = agent.select_action(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                
                agent.remember(state, action, reward, next_state, float(done))
                agent.learn(batch_size)
                
                # به‌روزرسانی نمایش گرافیکی
                visualizer.update_cartpole(next_state)
                visualizer.draw()
                
                state = next_state
                total_reward += reward
                
            if episode % target_update_freq == 0:
                agent.update_target_network()
                
            # به‌روزرسانی نمودار
            visualizer.update_chart(episode + 1, total_reward)
            visualizer.draw()
            
            print(f"Episode: {episode+1}, Score: {total_reward}, Epsilon: {agent.epsilon:.2f}")
            
            if total_reward >= 495:
                print("🎉 محیط با موفقیت حل شد!")
                break
                
    except KeyboardInterrupt:
        print("\n⏹️ آموزش متوقف شد.")
    finally:
        visualizer.close()
        env.close()

if __name__ == "__main__":
    train()