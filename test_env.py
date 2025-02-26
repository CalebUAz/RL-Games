from stable_baselines3.common.env_checker import check_env
from game import BowlingEnv

env = BowlingEnv()
check_env(env)  # This should pass without errors.
