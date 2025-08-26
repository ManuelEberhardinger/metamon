import os
os.environ["METAMON_CACHE_DIR"] = "data/"  # directory for logs

from stable_baselines3 import PPO
from sb3_contrib import MaskablePPO, RecurrentPPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback,  BaseCallback, CallbackList
from rnd import RNDVecWrapper

import gymnasium as gym
import numpy as np
from collections import defaultdict
from poke_env.teambuilder import Teambuilder
from tqdm import tqdm
import torch

import time, secrets, random


from metamon.baselines import get_baseline, get_all_baseline_names
from metamon.env import BattleAgainstBaseline, get_metamon_teams, PokeAgentLadder, PokeEnvWrapper
from metamon.interface import DefaultObservationSpace, DefaultShapedReward, DefaultActionSpace

from utils import PokemonWrapper, obs_space, action_space, reward_fn, team_set, CHECKPOINT_DIR, latest_checkpoint, tokenizer
from feature_extractor import TextNumericExtractor, _infer_vocab_and_pad
from bandit import BetaBernoulliBandit


class ResampleTeamOnReset(gym.Wrapper):
    """Resample self/team each episode and expose team_idx_used for SB3 get_attr()."""
    def __init__(self, env, teams, sample_idx_fn):
        super().__init__(env)
        self._teams = teams
        self._sample_idx = sample_idx_fn
        self.team_idx_used = None  # read by your callback via get_attr

    def _apply_team(self, idx: int):
        team = self._teams[idx]
        base = self.unwrapped
        # keep wrapper bookkeeping
        if hasattr(base, "player_team_set"):
            base.player_team_set = team
        if hasattr(base, "my_team"):
            base.my_team = team
        # ✅ make the poke-env player use the new Teambuilder
        if hasattr(base, "agent") and hasattr(base.agent, "_team"):
            base.agent._team = team
        # (optional) keep this in sync for logs
        if hasattr(base, "metamon_team_set"):
            base.metamon_team_set = team
        self.team_idx_used = idx


    def reset(self, **kwargs):
        idx = self._sample_idx()
        self._apply_team(idx)
        obs, info = self.env.reset(**kwargs)  # Gymnasium API
        return obs, info

class TeamBanditCallback(BaseCallback):
    def __init__(self, bandit, verbose=0):
        super().__init__(verbose)
        self.bandit = bandit

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])
        for i, done in enumerate(dones):
            if not done:
                continue
            win = int(infos[i].get("won", 0)) if isinstance(infos[i], dict) else 0

            # team -> bandit
            try:
                team_idx = self.training_env.get_attr("team_idx_used", indices=[i])[0]
            except Exception:
                team_idx = None
            if team_idx is not None:
                self.bandit.update(team_idx, win)

            # opponent -> online weights
            try:
                opp_name = self.training_env.get_attr("opp_name_used", indices=[i])[0]
            except Exception:
                opp_name = None
            if opp_name is not None:
                online_update_opponent_weights(opp_name, win)

        return True

class BaselineSetOpponent(PokeEnvWrapper):
    def __init__(
        self,
        battle_format: str,
        observation_space,
        action_space,
        reward_function,
        my_team,
        opponent_type,
        turn_limit: int = 200,
        save_trajectories_to=None,
        save_team_results_to=None,
        battle_backend: str = "poke-env",
        player_username: str | None = None,
        opponent_username: str | None = None,
    ):
        super().__init__(
            battle_format=battle_format,
            observation_space=observation_space,
            action_space=action_space,
            reward_function=reward_function,
            player_team_set=my_team,
            opponent_team_set=team_set,
            opponent_type=opponent_type,
            turn_limit=turn_limit,
            save_trajectories_to=save_trajectories_to,
            save_team_results_to=save_team_results_to,
            battle_backend=battle_backend,
            player_username=player_username,
            opponent_username=opponent_username,
        )
        self._opponent_team_set = team_set

# ----------------- CONFIG -----------------
N_ENVS = 32
CHUNK_STEPS = 200_000
TOTAL_STEPS = 100_000_000

teams = []
for i in range(len(team_set.team_files)):
    new_team = get_metamon_teams("gen1ou", "competitive")
    new_team.team_files = [new_team.team_files[i]]
    teams.append(new_team)


TEAM_BANDIT = BetaBernoulliBandit(n_arms=len(teams))

USE_RNN = False
USE_TEXT = True

MODEL_TYPE  = RecurrentPPO if USE_RNN else PPO
POLICY_TYPE = ("MultiInputLstmPolicy" if USE_RNN else "MultiInputPolicy") if USE_TEXT else ("MlpLstmPolicy" if USE_RNN else "MlpPolicy")

RESUME_TRAINING = "checkpoints/run_74/"

all_names = list(get_all_baseline_names())
_wr_ema = {name: 0.5 for name in all_names}
weights = {name: 1.0 for name in all_names}
EMA_TOP_K = 10  # set None to show all



HARDNESS_POW = 2.5          # ↑ more > 1.0 = emphasize hard opponents more
MIN_PROB_PER_OPP = 0.002    # floor per opponent to avoid starvation (0.2% each)
OPP_EMA_DECAY = 0.9         # keep if you like it smooth; smaller = faster adaptation

TENSORBOARD_DIR = "./ppo_gen1ou_tensorboard/"


def print_opp_wr_ema(top_k=EMA_TOP_K):
    # lower EMA == harder opponent (since it's your win-rate)
    items = sorted(_wr_ema.items(), key=lambda kv: kv[1])  # hardest → easiest
    if top_k is not None:
        items = items[:top_k]
    line = " | ".join([f"{name}:{ema:.3f}" for name, ema in items])
    print(f"[Opp EMA] hardest→easiest: {line}")


def online_update_opponent_weights(opp_name: str, win: int):
    global _wr_ema, weights, all_names

    # EMA of win-rate for this opponent
    prev = _wr_ema.get(opp_name, 0.5)
    _wr_ema[opp_name] = OPP_EMA_DECAY * prev + (1.0 - OPP_EMA_DECAY) * float(win)

    # Hardness = 1 - EMA; raise to power to emphasize hard opps
    hardness = np.array([max(1e-8, 1.0 - _wr_ema[n]) for n in all_names], dtype=float)
    hardness **= HARDNESS_POW

    probs = hardness / hardness.sum()

    # Apply a per-opp probability floor so no one gets starved
    if MIN_PROB_PER_OPP > 0:
        probs = np.maximum(probs, MIN_PROB_PER_OPP)
        probs /= probs.sum()

    weights = {n: float(p) for n, p in zip(all_names, probs)}

def sample_my_team_idx() -> int:
    return TEAM_BANDIT.select()

def sample_opponent() -> str:
    names = list(weights.keys())
    w = np.array([weights[n] for n in names], dtype=float)
    return random.choices(names, weights=w, k=1)[0]

def make_weighted_env():
    opp = sample_opponent()
    base = BaselineSetOpponent(
        battle_format="gen1ou",
        observation_space=obs_space,     # <- should be TokenizedObservationSpace when USE_TEXT=True
        action_space=action_space,
        reward_function=reward_fn,
        my_team=teams[0],
        opponent_type=get_baseline(opp),
    )
    # keep Dict obs when using text
    env = PokemonWrapper(base, numbers_only=not USE_TEXT)
    env = ResampleTeamOnReset(env, teams, sample_my_team_idx)
    env.opp_name_used = opp
    return env

def build_train_vecenv(run_dir):
    venv = make_vec_env(make_weighted_env, n_envs=N_ENVS)
    try:
        _, vecnorm_path = latest_checkpoint(run_dir)
        venv = VecNormalize.load(vecnorm_path, venv)
    except:
        print(f'[Train env] did not find vecnorm, rebuilding')
        venv = VecNormalize(venv, norm_obs=(not USE_TEXT), norm_reward=True, clip_reward=10.0)

    # ensure training mode and flags are set either way
    venv.training = True
    venv.norm_reward = True
    venv.norm_obs = (not USE_TEXT)


    # RND expects flat Box obs; skip when using text
    if not USE_TEXT:
        venv = RNDVecWrapper(
            venv,
            embedding_dim=128,
            learning_rate=1e-4,
            intrinsic_reward_scale=0.1,
            rew_norm=False,
        )
    return venv

def update_weights_from_wr(wr: dict[str, float]):
    global weights, _wr_ema
    for n in all_names:
        _wr_ema[n] = 0.7 * _wr_ema.get(n, 0.5) + 0.3 * wr.get(n, 0.5)
    diffs = np.array([1.0 - _wr_ema[n] for n in all_names], dtype=float)
    diffs = np.clip(diffs, 1e-8, None)
    probs = diffs / diffs.sum()
    weights = {n: float(p) for n, p in zip(all_names, probs)}

    print(f'[OPS] New weights: {weights}')

def update_teams_from_wr(wr: dict[str, float]):
    global TEAM_WEIGHTS, TEAM_WR_EMA
    for i in TEAM_WR_EMA.keys():
        TEAM_WR_EMA[i] = 0.8 * TEAM_WR_EMA[i] + 0.2 * wr.get(i, 0.5)
    
    
    diffs = np.array([TEAM_WR_EMA[i] for i in TEAM_WR_EMA.keys()], dtype=float)
    diffs = np.clip(diffs, 1e-8, None)
    probs = diffs / diffs.sum()
    TEAM_WEIGHTS = {i: float(probs[i]) for i in range(len(teams))}

    print('[Teams] EMA:', {i: round(TEAM_WR_EMA[i], 3) for i in range(len(teams))})
    print('[Teams] New weights:', TEAM_WEIGHTS)

if __name__ == "__main__":
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    run_num = len(os.listdir('./checkpoints/'))
    save_path = f"./checkpoints/run_{run_num}/"
    bandit_cb = TeamBanditCallback(TEAM_BANDIT)
    checkpoint_callback = CheckpointCallback(
        save_freq=int(CHUNK_STEPS/N_ENVS),
        save_path=save_path,
        name_prefix="ppo_showdown_agent",
        save_vecnormalize=True,
        save_replay_buffer=True
    )
    
    callback = CallbackList([checkpoint_callback, bandit_cb])
    steps_done = 0
    first = True

    if RESUME_TRAINING:
        model_path, vecnorm_path = latest_checkpoint(RESUME_TRAINING)
        print(f"Resuming from {model_path}")

        vec_env = make_vec_env(make_weighted_env, n_envs=N_ENVS)
        vec_env = VecNormalize.load(vecnorm_path, vec_env)
        vec_env.training = True
        vec_env.norm_reward = True
        vec_env.norm_obs = (not USE_TEXT)

        if not USE_TEXT:
            vec_env = RNDVecWrapper(
                vec_env, embedding_dim=128, learning_rate=1e-4,
                intrinsic_reward_scale=0.1, rew_norm=False
            )

        model = MODEL_TYPE.load(model_path, env=vec_env, device="cpu")
        steps_done = model.num_timesteps
        first = False

    else:
        vec_env = build_train_vecenv(save_path)

         # 🔍 Sanity check (run only once on a single env, not the vectorized one)
        test_env = make_weighted_env()
        obs, _ = test_env.reset()
        if USE_TEXT:
            assert "numbers" in obs and "text_tokens" in obs, \
                f"Unexpected obs keys: {obs.keys()}"
            assert obs["text_tokens"].dtype.kind in "iu", \
                f"tokens must be int, got {obs['text_tokens'].dtype}"
            print("numbers:", obs["numbers"].shape, obs["numbers"].dtype)
            print("text_tokens:", obs["text_tokens"].shape, obs["text_tokens"].dtype,
                "min", obs["text_tokens"].min(), "max", obs["text_tokens"].max())
        else:
            assert isinstance(obs, np.ndarray), f"Expected np.ndarray, got {type(obs)}"
            print("numbers only:", obs.shape, obs.dtype)

        if USE_RNN:
            policy_kwargs = dict(
                net_arch=dict(pi=[256, 256, 256], vf=[256, 256, 256]),
                lstm_hidden_size=256,
                n_lstm_layers=1,
                activation_fn=torch.nn.SiLU,
                ortho_init=True,
            )
        else:
            policy_kwargs = dict(
                net_arch=dict(pi=[512, 512, 512, 512], vf=[512, 512, 512, 512]),
                activation_fn=torch.nn.SiLU,
                ortho_init=True,
            )

        if USE_TEXT:
            vocab_size, pad_id = _infer_vocab_and_pad(tokenizer)
            policy_kwargs.update(
                features_extractor_class=TextNumericExtractor,
                features_extractor_kwargs=dict(
                    vocab_size=vocab_size,
                    embed_dim=128,
                    num_hidden=128,
                    padding_idx=pad_id,
                ),
            )

        model = MODEL_TYPE(
            POLICY_TYPE,
            vec_env,
            device="auto",
            verbose=1,
            tensorboard_log=TENSORBOARD_DIR,
            policy_kwargs=policy_kwargs,
            learning_rate=3e-4,
            n_steps=512,
            batch_size=4096,
            n_epochs=10,
            ent_coef=0.005,
            gamma=0.99,
            gae_lambda=0.95,
            target_kl=0.05,
            clip_range=0.2,
            max_grad_norm=0.5,
        )

    
    while steps_done < TOTAL_STEPS:
        chunk = min(CHUNK_STEPS, TOTAL_STEPS - steps_done)
        model.learn(total_timesteps=chunk, reset_num_timesteps=first,
                    progress_bar=True, callback=callback)
        first = False
        steps_done += chunk
        vec_env.close()
        vec_env = build_train_vecenv(save_path)
        model.set_env(vec_env)
        TEAM_BANDIT.decay(rate = 0.99)

        tm = TEAM_BANDIT.alpha / (TEAM_BANDIT.alpha + TEAM_BANDIT.beta)
        top = sorted(weights.items(), key=lambda kv: kv[1], reverse=True)[:5]
        print(f"[Chunk] steps_done={steps_done} | team_means={np.round(tm,3).tolist()} "
            f"| top-5 opp weights={[(k, round(v,3)) for k,v in top]}"
            )
        print_opp_wr_ema()