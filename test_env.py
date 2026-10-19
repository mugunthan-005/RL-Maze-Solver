import gymnasium as gym

# Load the built-in maze environment
env = gym.make("FrozenLake-v1", is_slippery=False, render_mode="human")
env.reset()

print("Environment loaded successfully! Taking random steps...")

# Take 5 random steps
for step in range(5):
    random_action = env.action_space.sample()
    env.step(random_action)
    env.render()

env.close()
print("Test complete.")