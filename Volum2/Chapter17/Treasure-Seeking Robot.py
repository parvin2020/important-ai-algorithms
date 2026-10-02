import gymnasium as gym
import numpy as np

# -------------------------------------------------
# 1. Environment
# -------------------------------------------------
env = gym.make("FrozenLake-v1", is_slippery=False)

# 16 states × 4 actions
theta = np.zeros((env.observation_space.n,
                  env.action_space.n))

lr = 0.1
gamma = 0.9

# -------------------------------------------------
# 2. Softmax
# -------------------------------------------------
def softmax(x):
    x = x - np.max(x)
    p = np.exp(x)
    return p / np.sum(p)

# -------------------------------------------------
# 3. Training - REINFORCE
# -------------------------------------------------
for episode in range(1000):

    state, info = env.reset()

    states = []
    actions = []
    rewards = []

    done = False

    while not done:

        # Policy → probabilities
        p = softmax(theta[state])

        # Select action
        action = np.random.choice(4, p=p)

        # Environment
        next_state, reward, terminated, truncated, info = env.step(action)

        states.append(state)
        actions.append(action)
        rewards.append(reward)
        state = next_state
        done = terminated or truncated

    # -------------------------------------------------
    # 4. Calculate Returns
    # -------------------------------------------------
    G = 0
    returns = []

    for reward in rewards[::-1]:
        G = reward + gamma * G
        returns.insert(0, G)

    # -------------------------------------------------
    # 5. Policy Gradient
    # -------------------------------------------------
    for state, action, G in zip(states, actions, returns):

        p = softmax(theta[state])

        gradient = -p
        gradient[action] += 1

        theta[state] += lr * G * gradient
env.close()

print("Training finished.")