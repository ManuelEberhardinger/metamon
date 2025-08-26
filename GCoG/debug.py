import os
from itertools import cycle
os.environ["METAMON_CACHE_DIR"] = "data/"  # directory for logs

from metamon.baselines import get_baseline, get_all_baseline_names
from metamon.env import BattleAgainstBaseline, get_metamon_teams, PokeAgentLadder
from metamon.interface import DefaultObservationSpace, DefaultShapedReward, DefaultActionSpace

from utils import PokemonWrapper

team_set = get_metamon_teams("gen1ou", "competitive")
obs_space = DefaultObservationSpace()
reward_fn = DefaultShapedReward()
action_space = DefaultActionSpace()

opponent_cycle = cycle(get_all_baseline_names())


base = BattleAgainstBaseline(
        battle_format="gen1ou",
        observation_space=obs_space,
        action_space=action_space,
        reward_function=reward_fn,
        team_set=team_set,
        opponent_type=get_baseline(next(opponent_cycle)),
) 
env = PokemonWrapper(base)

while True:
    terminated = False
    obs = env.reset()
    while not terminated:
        action = env.action_space.sample()  # Random action for demonstration
        next_obs, reward, terminated, truncated = env.step(action)
        obs = next_obs