# eval_all_baselines.py
import os, glob
import numpy as np
os.environ["METAMON_CACHE_DIR"] = "data/"

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize

from metamon.baselines import get_baseline, get_all_baseline_names
from metamon.env import BattleAgainstBaseline
from utils import PokemonWrapper, obs_space, action_space, reward_fn, team_set, latest_checkpoint
from rnd import RNDVecWrapper


CHECKPOINT_DIR = "./checkpoints/run_11/"
EPISODES_PER_BASELINE = 5  # keep it small & simple



def make_env_factory(opponent_name: str):
    """Build the exact same stack as training, but with a fixed opponent."""
    def _thunk():
        base = BattleAgainstBaseline(
            battle_format="gen1ou",
            observation_space=obs_space,
            action_space=action_space,
            reward_function=reward_fn,
            team_set=team_set,
            opponent_type=get_baseline(opponent_name),
        )
        env = PokemonWrapper(base)
        return env
    return _thunk

def build_eval_vecenv(opponent_name: str, vecnorm_path: str = None):
    venv = make_vec_env(make_env_factory(opponent_name), n_envs=1)
    venv = RNDVecWrapper(
        venv,
        embedding_dim=128,
        learning_rate=1e-4,
        intrinsic_reward_scale=0.0,  # no intrinsic reward during eval
        rew_norm=True,
    )
    venv = VecNormalize.load(vecnorm_path, venv)
    venv.training = False
    venv.norm_reward = False
    return venv

def run_episodes(env, model: PPO, n_episodes: int):
    wins, rets = [], []
    for _ in range(n_episodes):
        obs = env.reset()
        done = np.array([False])
        ret = 0.0
        while not done[0]:
            action, _ = model.predict(obs, deterministic=True)
            obs, rewards, dones, infos = env.step(action)
            ret += float(rewards[0])
            done = dones
            if done[0]:
                info = infos[0]
                wins.append(info["won"])
        rets.append(ret)
    return float(np.mean(wins)), float(np.mean(rets))

if __name__ == "__main__":
    names = get_all_baseline_names()
    first = names[0]

    model_path, vec_norm = latest_checkpoint(CHECKPOINT_DIR)
    tmp_env = build_eval_vecenv(first, vec_norm)

    print(f"Using checkpoint: {model_path}")
    model = PPO.load(model_path, env=tmp_env, device="cpu")

    print("\n=== Eval vs all baselines ===")
    return_list = []
    for opp in names:
        env = build_eval_vecenv(opp, vec_norm)

        model.set_env(env)
        win_rate, avg_return = run_episodes(env, model, EPISODES_PER_BASELINE)
        print(f"{opp:>20s} | win_rate={win_rate:.3f} | avg_return={avg_return:.3f}")