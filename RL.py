import mlflow
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from game import BowlingEnv

# Initialize MLflow
mlflow.set_experiment("Reinforcement Learning with PPO")

# Create the environment
env = BowlingEnv()

# Start an MLflow run
with mlflow.start_run():
    # Log parameters
    mlflow.log_param("algorithm", "PPO")
    mlflow.log_param("policy", "MlpPolicy")
    mlflow.log_param("total_timesteps", 1000000)

    # Train the model and log loss (approximated by reward or custom metric)
    model = PPO("MlpPolicy", env, verbose=1)
    
    rewards = []
    for timestep in range(1, 1000001):
        # Simulate training loop and log a pseudo-loss (negative reward)
        model.learn(total_timesteps=1, reset_num_timesteps=False)
        obs, _ = env.reset()
        action, _states = model.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Log negative reward as loss for visualization purposes
        rewards.append(reward)
        mlflow.log_metric("loss", -reward, step=timestep)

    # Save the trained model and log it with MLflow
    model_path = "ppo_bowling_model.zip"
    model.save(model_path)
    mlflow.log_artifact(model_path, artifact_path="models")

    # Plot and log the loss curve
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(rewards) + 1), [-r for r in rewards], label="Loss (Negative Reward)")
    plt.xlabel("Timestep")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.legend()
    
    # Save and log the plot
    plot_path = "loss_curve.png"
    plt.savefig(plot_path)
    mlflow.log_artifact(plot_path, artifact_path="plots")

env.close()
