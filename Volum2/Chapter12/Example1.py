import random

# Actions
LEFT = 0
RIGHT = 1

# Environment:
# State 0 = start
# State 1 = middle
# State 2 = goal / terminal
def step(s, a):
    # Terminal state
    if s == 2:
        return 2, 0.0, True

    # Start state
    if s == 0:
        if a == RIGHT:
            return 1, -1.0, False
        else:
            return 0, -1.0, False

    # Middle state
    if s == 1:
        if a == RIGHT:
            return 2, 10.0, True
        else:
            return 0, -1.0, False


# Epsilon-greedy action selection
def choose_action(s, Q, epsilon):
    # Exploration: choose random action
    if random.random() < epsilon:
        return random.choice([LEFT, RIGHT])

    # Exploitation: choose best action
    # If tie, choose RIGHT for faster learning
    if Q[s][RIGHT] >= Q[s][LEFT]:
        return RIGHT
    else:
        return LEFT


# For reproducible results
random.seed(42)

# Q-learning parameters
alpha = 0.5      # learning rate
gamma = 0.9      # discount factor
epsilon = 0.2    # exploration rate
episodes = 1000  # number of episodes

# Q table for states 0 and 1
# State 2 is terminal, so it has no Q row
Q = [
    [0.0, 0.0],  # State 0: LEFT, RIGHT
    [0.0, 0.0]   # State 1: LEFT, RIGHT
]

# Training loop
for episode in range(1, episodes + 1):

    # Each episode starts at state 0
    s = 0

    # Continue until reaching goal
    while s != 2:

        # Choose action based on epsilon-greedy policy
        a = choose_action(s, Q, epsilon)

        # Take action in environment
        next_s, r, done = step(s, a)

        # Compute Q-learning target
        if done:
            target = r
        else:
            target = r + gamma * max(Q[next_s])

        # Q update formula
        Q[s][a] = Q[s][a] + alpha * (target - Q[s][a])

        # Move to next state
        s = next_s


# Print final Q table
print("Final Q table:")
print("State 0: LEFT =", round(Q[0][LEFT], 3), ", RIGHT =", round(Q[0][RIGHT], 3))
print("State 1: LEFT =", round(Q[1][LEFT], 3), ", RIGHT =", round(Q[1][RIGHT], 3))

# Extract learned policy
policy0 = "RIGHT" if Q[0][RIGHT] >= Q[0][LEFT] else "LEFT"
policy1 = "RIGHT" if Q[1][RIGHT] >= Q[1][LEFT] else "LEFT"

# Print learned policy
print("Learned policy:")
print("State 0 ->", policy0)
print("State 1 ->", policy1)