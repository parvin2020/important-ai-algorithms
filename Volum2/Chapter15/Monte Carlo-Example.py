import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
# -----------------------------
# ایجاد محیط
# -----------------------------

env = gym.make(
    "FrozenLake-v1",
    is_slippery=False
)
# -----------------------------
# پارامترها
# -----------------------------
gamma = 0.95
epsilon = 0.1
episodes = 5000

# -----------------------------
# جدول Q
# -----------------------------
Q = defaultdict(
    lambda: np.zeros(env.action_space.n)
)

# ذخیره Returnها
Returns = defaultdict(list)

# ذخیره موفقیت‌ها
success_history = []

# -----------------------------
# انتخاب عمل
# -----------------------------
def choose_action(state):

    # اگر هنوز چیزی درباره این حالت نمی‌دانیم
    if np.all(Q[state] == 0):

        return env.action_space.sample()

    # Exploration
    if np.random.rand() < epsilon:

        return env.action_space.sample()

    # Exploitation
    return np.argmax(Q[state])

# -----------------------------
# آموزش Monte Carlo
# -----------------------------
for episode_number in range(episodes):
    state, _ = env.reset()
    episode = []
    done = False
    # اجرای Episode
    while not done:
        action = choose_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        episode.append(
            (state, action, reward)
        )
        state = next_state

    # ذخیره موفقیت
    if reward == 1:
        success_history.append(1)
    else:
        success_history.append(0)

    # -------------------------
    # محاسبه Return
    # -------------------------
    G = 0
    visited = set()
    for state, action, reward in reversed(episode):
        G = reward + gamma * G
        # First-Visit Monte Carlo
        if (state, action) not in visited:
            visited.add((state, action))
            Returns[(state, action)].append(G)
            Q[state][action] = np.mean(
                Returns[(state, action)]
            )

# -----------------------------
# نمایش Q-Table
# -----------------------------
print("\nQ-Table:\n")
for state in range(env.observation_space.n):
    print(
        "State",
        state,
        ":",
        np.round(Q[state], 3)
    )
# -----------------------------
# محاسبه نرخ موفقیت
# -----------------------------
window = 100
success_rate = []
for i in range(0, episodes, window):
    rate = np.mean(
        success_history[i:i + window]
    )
    success_rate.append(rate)

# -----------------------------
# رسم نمودار
# -----------------------------
plt.figure(figsize=(10, 5))
plt.plot(
    range(
        window,
        episodes + 1,
        window
    ),
    success_rate)
plt.xlabel("Episode")
plt.ylabel("Success Rate")
plt.title(
    "Monte Carlo Learning on FrozenLake"
)
plt.grid()
plt.show()

# -----------------------------
# اجرای سیاست نهایی
# -----------------------------
print("\nAgent is running...\n")
state, _ = env.reset()
done = False
total_reward = 0
while not done:
    action = np.argmax(Q[state])
    next_state, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated
    print(
        "State:",
        state,
        "Action:",
        action,
        "Reward:",
        reward
    )
    total_reward += reward
    state = next_state

print("\nTotal Reward:", total_reward)
env.close()