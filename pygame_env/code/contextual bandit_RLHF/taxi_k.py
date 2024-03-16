import gymnasium as gym

env = gym.make('Taxi-v3', render_mode="human")
env.reset()

while True:
    action = env.action_space.sample()  # agent policy that uses the observation and info
    observation, reward, terminated, truncated, info = env.step(action)

    print(observation)
    print(reward)

    if terminated or truncated:
        observation, info = env.reset()

env.close()