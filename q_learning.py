
import gymnasium as gym
import numpy as np
import random
import matplotlib.pyplot as plt
import time

# --------------------------------------------------
# 1. CREATE THE CUSTOM 5x5 MAZE
# --------------------------------------------------

custom_maze = [
    "SFFFF",
    "FHHHF",
    "FFFFF",
    "FHHHF",
    "FFFFG"
]

env = gym.make(
    "FrozenLake-v1",
    desc=custom_maze,
    is_slippery=False
)

# --------------------------------------------------
# 2. Q-LEARNING HYPERPARAMETERS
# --------------------------------------------------

learning_rate = 0.8
discount_factor = 0.95
episodes = 1000

exploration_rate = 1.0
min_exploration = 0.01
exploration_decay = 0.001

# For reproducible results
random.seed(42)
np.random.seed(42)

# --------------------------------------------------
# 3. INITIALIZE THE Q-TABLE
# --------------------------------------------------

state_space = env.observation_space.n
action_space = env.action_space.n

q_table = np.zeros((state_space, action_space))

rewards_all_episodes = []

# --------------------------------------------------
# 4. TRAIN THE AGENT
# --------------------------------------------------

for episode in range(episodes):

    state, info = env.reset()
    done = False
    rewards_current_episode = 0

    while not done:

        # Epsilon-greedy action selection
        if random.random() > exploration_rate:
            action = int(np.argmax(q_table[state, :]))
        else:
            action = env.action_space.sample()

        # Perform the action
        new_state, reward, terminated, truncated, info = (
            env.step(action)
        )

        done = terminated or truncated

        # Custom reward system
        if reward == 1.0:
            reward = 100.0
        elif terminated:
            reward = -1.0
        elif truncated:
            reward = -1.0
        else:
            reward = -0.01

        # Calculate the maximum future Q-value
        if done:
            next_max = 0.0
        else:
            next_max = np.max(q_table[new_state, :])

        # Q-learning update
        old_value = q_table[state, action]

        new_value = (
            (1 - learning_rate) * old_value
            + learning_rate
            * (reward + discount_factor * next_max)
        )

        q_table[state, action] = new_value

        # Move to the next state
        state = new_state
        rewards_current_episode += reward

    # Store exactly one reward per episode
    rewards_all_episodes.append(rewards_current_episode)

    # Linear exploration decay
    exploration_rate = max(
        min_exploration,
        exploration_rate - exploration_decay
    )

env.close()

print("Training complete!")
print("\nFinal Q-Table:")
print(np.round(q_table, 3))

# --------------------------------------------------
# 5. PLOT AVERAGE REWARD EVERY 100 EPISODES
# --------------------------------------------------

episodes_per_block = 100

average_rewards = [
    np.mean(rewards_all_episodes[i:i + episodes_per_block])
    for i in range(0, episodes, episodes_per_block)
]

plt.figure(figsize=(10, 5))

plt.plot(
    range(episodes_per_block, episodes + 1, episodes_per_block),
    average_rewards
)

plt.title("Q-Learning: Agent Improvement Over Time")
plt.xlabel("Episodes")
plt.ylabel("Average Reward (Higher Is Better)")
plt.grid(True)
plt.tight_layout()
plt.show()

# --------------------------------------------------
# 6. EVALUATE THE TRAINED AGENT
# --------------------------------------------------

evaluation_episodes = 100
successful_episodes = 0

# Goal state is the bottom-right cell: state 24
goal_state = state_space - 1

for episode in range(evaluation_episodes):

    state, info = env.reset()
    done = False

    while not done:

        # Greedy action: use the learned Q-table
        action = int(np.argmax(q_table[state, :]))

        state, reward, terminated, truncated, info = (
            env.step(action)
        )

        done = terminated or truncated

    if state == goal_state:
        successful_episodes += 1

success_rate = (
    successful_episodes / evaluation_episodes
) * 100

print("\nEvaluation Results")
print("------------------")
print("Evaluation episodes:", evaluation_episodes)
print("Successful episodes:", successful_episodes)
print(f"Success rate: {success_rate:.2f}%")

# --------------------------------------------------
# 7. VISUALIZE THE TRAINED AGENT
# --------------------------------------------------

env.close()

visual_env = gym.make(
    "FrozenLake-v1",
    desc=custom_maze,
    is_slippery=False,
    render_mode="human"
)

state, info = visual_env.reset()
done = False

print("\nLaunching the trained agent...")
time.sleep(1)

while not done:

    # Choose the best learned action
    action = int(np.argmax(q_table[state, :]))

    new_state, reward, terminated, truncated, info = (
        visual_env.step(action)
    )

    state = new_state
    done = terminated or truncated

    time.sleep(0.5)

if state == goal_state:
    print("Agent successfully reached the goal!")
else:
    print("Agent did not reach the goal.")

visual_env.close()
