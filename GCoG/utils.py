import gymnasium as gym
from gymnasium import ObservationWrapper, spaces
import numpy as np
from itertools import cycle
import glob

import os

os.environ["METAMON_CACHE_DIR"] = "data/"  # directory for logs
from metamon.baselines import get_baseline, get_all_baseline_names
from metamon.env import BattleAgainstBaseline, get_metamon_teams, PokeAgentLadder
from metamon.interface import DefaultObservationSpace, ExpandedObservationSpace ,TokenizedObservationSpace, DefaultShapedReward, DefaultActionSpace
from metamon.tokenizer.tokenizer import get_tokenizer

from sb3_contrib.common.wrappers import ActionMasker
from feature_extractor import TextNumericExtractor, _infer_vocab_and_pad

tokenizer = get_tokenizer("DefaultObservationSpace-v0")
team_set = get_metamon_teams("gen1ou", "competitive")
obs_space = TokenizedObservationSpace(ExpandedObservationSpace(), tokenizer)
reward_fn = DefaultShapedReward()
action_space = DefaultActionSpace()

opponent_cycle = cycle(get_all_baseline_names())  # Cycle through all available opponents


CHECKPOINT_DIR = "./checkpoints/"


def latest_checkpoint(path=CHECKPOINT_DIR):
    files = glob.glob(os.path.join(path, "*.zip"))
    if not files:
        raise FileNotFoundError(f"No .zip checkpoints in {path}")
    latest_file = max(files, key=os.path.getmtime)

    vec_files = glob.glob(os.path.join(path, "ppo_showdown_agent_vecnormalize_*_steps.pkl"))
    latest_vec_file = max(vec_files, key=os.path.getmtime) if vec_files else None

    return latest_file, latest_vec_file


class PokemonWrapper(ObservationWrapper):
    def __init__(self, env, numbers_only = False):
        super().__init__(env)
        self.numbers_only = numbers_only
        if numbers_only:
            num_space = env.observation_space["numbers"]
        
            self.observation_space = spaces.Box(
                low=num_space.low, high=num_space.high,
                shape=num_space.shape, dtype=num_space.dtype
            )
        else:
            self.observation_space = env.observation_space

    def observation(self, obs):
        if self.numbers_only:
            return obs["numbers"]
        return obs

def make_train_env(opp, format = "gen1ou", numbers_only = False):
    print(f"Using opponent: {opp}")
    base = BattleAgainstBaseline(
        battle_format=format,
        observation_space=obs_space,
        action_space=action_space,
        reward_function=reward_fn,
        team_set=team_set,
        opponent_type=get_baseline(opp),
        )
    return PokemonWrapper(base, numbers_only)

def make_ladder_env(battle_format = "gen1ou", numbers_only = False, my_team = team_set, num_games = 100):
    base = PokeAgentLadder(
        battle_format=battle_format,
        player_username ="PAC-GCOGS",
        player_password ="?gcog-2025?",
        num_battles=num_games,
        observation_space=obs_space,
        action_space=action_space,
        reward_function=reward_fn,
        player_team_set=my_team
    )
    return PokemonWrapper(base, numbers_only)

if __name__ == "__main__":
    env = make_train_env(opp = "Grunt")
    vocab_size, pad_id = _infer_vocab_and_pad(tokenizer)
    feature_extractor = TextNumericExtractor(env.observation_space, vocab_size)
    for i in range(100):
        done = False
        obs = env.reset()
        while not done:
            obs = env.step(env.action_space.sample())