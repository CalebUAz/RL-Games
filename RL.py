from stable_baselines3 import PPO
from game import BowlingEnv

# Create the environment
env = BowlingEnv()

# Train the model
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=1000000)

# Test the trained model
obs, _ = env.reset()  # Unpack the tuple returned by reset()
while True:
    action, _states = model.predict(obs)  # Pass only obs to predict()
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()
    if terminated or truncated:
        break

env.close()

