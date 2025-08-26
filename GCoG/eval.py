import os
os.environ["METAMON_CACHE_DIR"] = "data/"

import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize

# Import to ensure the custom extractor is registered on load:
from feature_extractor import TextNumericExtractor  # noqa: F401

from utils import make_ladder_env, latest_checkpoint
from train import teams


best_team = teams[0]
CHECKPOINT_DIR = "./checkpoints/run_76"
model_path, vecnorm_path = latest_checkpoint(CHECKPOINT_DIR)
print(f"Loading checkpoint from {model_path}")




N = 100
wins = 0
# 1) Build eval env exposing Dict obs (numbers + text_tokens)
eval_base = make_vec_env(lambda: make_ladder_env(numbers_only=False, my_team = best_team, num_games = N), n_envs=1)

# 2) Load VecNormalize in the same position/order as training
vecnorm = VecNormalize.load(vecnorm_path, eval_base)
vecnorm.training = False
vecnorm.norm_reward = False
vecnorm.norm_obs = False  # Dict obs; keep False

eval_env = vecnorm  # no RND when using text

# 3) Load model with the SAME env it will be stepped on
device = "auto" if torch.cuda.is_available() else "cpu"
model = PPO.load(model_path, env=eval_env, device=device)
obs = eval_env.reset()
for _ in range(N):
    # 5) VecEnv-style evaluation loop
    done = np.array([False])
    while not done[0]:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = eval_env.step(action)
        if done[0] and info[0].get("won", False):
            wins += 1
print(f"Eval WR over {N} ladder games: {wins/N:.3f}")