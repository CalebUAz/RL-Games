import gymnasium as gym
from gymnasium import spaces
import numpy as np
from game import BowlingGame
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import BaseCallback
import mlflow
import pygame


class BowlingEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}
    def __init__(self, render_mode=None):
        self.game = BowlingGame()
        
        # Define action space (must be a gym.spaces object)
        self.action_space = spaces.Discrete(3)  # 0=left, 1=right, 2=roll
        
        # Define observation space (must be a gym.spaces object)
        self.observation_space = spaces.Dict({
            "ball": spaces.Box(low=0, high=800, shape=(2,), dtype=np.float32),
            "pins": spaces.Box(low=-1, high=800, shape=(10, 2), dtype=np.float32)
        })
        # Add to existing __init__
        self.render_mode = render_mode
        self.screen = None  # PyGame surface
        self.clock = pygame.time.Clock()

    def reset(self, seed=None, options=None):
        obs = self.game.reset()
        # Return fixed observation structure
        return {
            "ball": np.array(self.game.ball.center, dtype=np.float32),
            "pins": np.array(
                [pin.center if pin in self.game.pins else (-1, -1) 
                 for pin in self.game.original_pins],  # Use original reference
                dtype=np.float32
            )
        }, {}

    def step(self, action):
        obs, reward, done, _ = self.game.step(action)
        # Maintain fixed 10-pin structure with (-1,-1) placeholder
        return {
            "ball": np.array(obs["ball"], dtype=np.float32),
            "pins": np.array([p if p in obs["pins"] else (-1,-1) 
                            for p in self.game.original_pins], dtype=np.float32)
        }, reward, done, False, {}
    
    def render(self):
        if self.render_mode not in ["human", "rgb_array"]:
            return
        
        # Initialize PyGame once
        if self.screen is None and self.render_mode == "human":
            pygame.init()
            self.screen = pygame.display.set_mode((800, 600))
        
        # Draw game elements
        self.screen.fill((0, 0, 0))
        self.game.draw(self.screen)  # Ensure BowlingGame has draw() method
        
        if self.render_mode == "human":
            pygame.event.pump()
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])
        else:  # rgb_array
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), 
                axes=(1, 0, 2)
            )

def train_rl():
    # Add render_mode parameter
    env = make_vec_env(
        lambda: BowlingEnv(render_mode="human"), 
        n_envs=1  # Start with 1 env for rendering
    )
    
    with mlflow.start_run():
        model = PPO("MultiInputPolicy", env, verbose=1,
                   learning_rate=0.0003,
                   n_steps=2048,
                   batch_size=64)
        
        model.learn(
            total_timesteps=100000,
            callback=MLflowCallback(),  # Instantiate callback properly
            progress_bar=True
        )
        
        model.save("bowling_ppo")
        # mlflow.log_artifact("bowling_ppo")

class MLflowCallback(BaseCallback):
    """Custom callback for logging metrics to MLflow"""
    
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
   
    def _on_step(self) -> bool:
        # Add rendering every 1000 steps
        if self.num_timesteps % 1000 == 0:
            self.training_env.render()
        reward = self.locals.get("rewards", [0])[-1]  # Example for last reward
        mlflow.log_metric("episode_reward", reward, step=self.num_timesteps)
        return True
    
def analyze_results(run_id):
    client = mlflow.tracking.MlflowClient()
    metrics = client.get_metric_history(run_id, "episode_reward")
    
    rewards = [m.value for m in metrics]
    steps = [m.step for m in metrics]
    
    plt.figure(figsize=(10, 5))
    plt.plot(steps, rewards)
    plt.title("Training Progress")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.savefig("training_progress.png")
    
    mlflow.log_artifact("training_progress.png")

if __name__ == "__main__":
    # Create and check your environment
    # env = BowlingEnv()
    # check_env(env)
    train_rl()