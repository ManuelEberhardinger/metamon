# selfplay_train.py
import os
os.environ["METAMON_CACHE_DIR"] = "data/"  # directory for logs

import glob
import random
import time
from collections import defaultdict

import gymnasium as gym
import numpy as np
import torch

from stable_baselines3 import PPO
from sb3_contrib import MaskablePPO, RecurrentPPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback, CallbackList

from rnd import RNDVecWrapper

import gymnasium as gym

from metamon.baselines import get_all_baseline_names
from metamon.env import get_metamon_teams, PokeEnvWrapper
from metamon.interface import DefaultObservationSpace, DefaultShapedReward, DefaultActionSpace

from utils import PokemonWrapper, obs_space, action_space, reward_fn, team_set, CHECKPOINT_DIR, latest_checkpoint, tokenizer
from feature_extractor import TextNumericExtractor, _infer_vocab_and_pad
from bandit import BetaBernoulliBandit


# =========================
# Self-Play Opponent Adapter
# =========================

class SB3SnapshotOpponent:
    """
    A 'baseline-compatible-like' opponent that uses a frozen SB3 model.
    Your PokemonWrapper should, at step-time, detect `env.opponent_agent` and call:
        opp_action = env.opponent_agent.act_from_processed(opp_obs, opp_mask)
    where opp_obs/mask are from the opponent's POV (the same Dict observation structure).
    """
    def __init__(self, model, use_text=True):
        self.model = model
        self.use_text = use_text
        self.model.policy.eval()

    def act_from_processed(self, obs_dict, action_mask=None):
        """
        obs_dict: Dict observation as seen by the learner (numbers/text_tokens)
        action_mask: Optional mask for MaskablePPO
        returns: int action id
        """
        if action_mask is not None and hasattr(self.model, "predict"):
            # sb3-contrib MaskablePPO will accept action_masks via keyword on predict
            try:
                action, _ = self.model.predict(obs_dict, deterministic=True, action_masks=action_mask)
            except TypeError:
                # Non-maskable algo or older interface
                action, _ = self.model.predict(obs_dict, deterministic=True)
        else:
            action, _ = self.model.predict(obs_dict, deterministic=True)

        if isinstance(action, (list, tuple, np.ndarray)):
            return int(action[0])
        return int(action)


class OpponentPool:
    """
    Holds recent self-play snapshots produced by CheckpointCallback.
    """
    def __init__(self, run_dir, algo_class, max_pool=10):
        self.run_dir = run_dir
        self.algo_class = algo_class
        self.max_pool = max_pool
        self.paths = []

    def _list_snapshots(self):
        # default SB3 naming from CheckpointCallback
        # NOTE: adjust the glob if you change the prefix
        pattern = os.path.join(self.run_dir, "ppo_showdown_agent_*_steps.zip")
        return sorted(glob.glob(pattern))

    def refresh(self):
        found = self._list_snapshots()
        if not found:
            return
        self.paths = found[-self.max_pool:]

    def sample_model(self):
        if not self.paths:
            self.refresh()
        if not self.paths:
            return None
        path = random.choice(self.paths)
        return self.algo_class.load(path, env=None, device="cpu")


# =========================
# Team handling & callbacks
# =========================

class ResampleTeamOnReset(gym.Wrapper):
    """Resample self/team each episode and expose team_idx_used for SB3 get_attr()."""
    def __init__(self, env, teams, sample_idx_fn):
        super().__init__(env)
        self._teams = teams
        self._sample_idx = sample_idx_fn
        self.team_idx_used = None  # read by callback via get_attr

    def _apply_team(self, idx: int):
        team = self._teams[idx]
        base = self.unwrapped
        # keep wrapper bookkeeping
        if hasattr(base, "player_team_set"):
            base.player_team_set = team
        if hasattr(base, "my_team"):
            base.my_team = team
        # make the poke-env player use the new Teambuilder
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
    """
    Keep your team bandit (only updates team success). Opponent sampling is handled by the self-play pool.
    """
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
            try:
                team_idx = self.training_env.get_attr("team_idx_used", indices=[i])[0]
            except Exception:
                team_idx = None
            if team_idx is not None:
                self.bandit.update(team_idx, win)
        return True


# =========================
# Self-Play Env
# =========================

class BaselineSetOpponent(PokeEnvWrapper):
    """
    Same as your earlier helper, but we won’t use a 'baseline name'. We plug in a policy opponent.
    """
    def __init__(
        self,
        battle_format: str,
        observation_space,
        action_space,
        reward_function,
        my_team,
        opponent_agent,  # <- SB3SnapshotOpponent
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
            opponent_type=None,  # not used; we inject the agent directly below
            turn_limit=turn_limit,
            save_trajectories_to=save_trajectories_to,
            save_team_results_to=save_team_results_to,
            battle_backend=battle_backend,
            player_username=player_username,
            opponent_username=opponent_username,
        )
        # Make the policy-based opponent visible to the wrapper.
        # Your PokemonWrapper should check for this and call .act_from_processed(...)
        self.opponent_agent = opponent_agent


# =========================
# Config
# =========================

N_ENVS = 32
CHUNK_STEPS = 200_000
TOTAL_STEPS = 100_000_000

# Build individual single-team containers from your team_set
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

# If resuming, the pool will read snapshots from there
RESUME_TRAINING = ""  # e.g., "checkpoints/run_74/"

TENSORBOARD_DIR = "./ppo_gen1ou_tensorboard/"


def sample_my_team_idx() -> int:
    return TEAM_BANDIT.select()


# =========================
# Factories
# =========================

def make_selfplay_env(opponent_pool, frozen_bootstrap_fn):
    """
    Create one env that pits the learner vs a frozen policy snapshot.
    If pool is empty on the first call, we bootstrap from the current model via frozen_bootstrap_fn().
    """
    frozen_model = opponent_pool.sample_model()
    if frozen_model is None and frozen_bootstrap_fn is not None:
        # First rollout or no snapshot yet: use a lagged copy of the current model
        frozen_model = frozen_bootstrap_fn()

    opponent_agent = SB3SnapshotOpponent(model=frozen_model, use_text=USE_TEXT)

    base = BaselineSetOpponent(
        battle_format="gen1ou",
        observation_space=obs_space,
        action_space=action_space,
        reward_function=reward_fn,
        my_team=teams[0],  # will be resampled on reset
        opponent_agent=opponent_agent,
    )

    env = PokemonWrapper(base, numbers_only=not USE_TEXT)
    env = ResampleTeamOnReset(env, teams, sample_my_team_idx)
    # Important: Your PokemonWrapper must, during step(), if hasattr(env, "opponent_agent"),
    # call env.opponent_agent.act_from_processed(opp_obs_dict, opp_mask)
    return env


def build_train_vecenv_selfplay(run_dir, opponent_pool, frozen_bootstrap_fn):
    venv = make_vec_env(lambda: make_selfplay_env(opponent_pool, frozen_bootstrap_fn), n_envs=N_ENVS)
    try:
        _, vecnorm_path = latest_checkpoint(run_dir)
        venv = VecNormalize.load(vecnorm_path, venv)
    except Exception:
        print(f'[Train env] did not find vecnorm, rebuilding')
        venv = VecNormalize(venv, norm_obs=(not USE_TEXT), norm_reward=True, clip_reward=10.0)

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


# =========================
# Main
# =========================

if __name__ == "__main__":
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    run_num = len(os.listdir('./checkpoints/'))
    save_path = f"./checkpoints/run_{run_num}/"

    # Opponent pool watches the current run's checkpoint dir (or RESUME_TRAINING if provided)
    pool_root = RESUME_TRAINING if RESUME_TRAINING else save_path
    OPP_POOL = OpponentPool(run_dir=pool_root, algo_class=MODEL_TYPE, max_pool=10)

    bandit_cb = TeamBanditCallback(TEAM_BANDIT)
    checkpoint_callback = CheckpointCallback(
        save_freq=int(CHUNK_STEPS // N_ENVS),
        save_path=save_path,
        name_prefix="ppo_showdown_agent",
        save_vecnormalize=True,
        save_replay_buffer=True
    )
    callback = CallbackList([checkpoint_callback, bandit_cb])

    steps_done = 0
    first = True

    # Build model & env (resume or fresh)
    if RESUME_TRAINING:
        model_path, vecnorm_path = latest_checkpoint(RESUME_TRAINING)
        print(f"Resuming from {model_path}")

        # During resume, we still build a self-play env (pool points to RESUME_TRAINING)
        def _bootstrap_copy():
            tmp = "/tmp/_sp_bootstrap.zip"
            # load then save to guarantee same algo class serialization
            mdl = MODEL_TYPE.load(model_path, env=None, device="cpu")
            mdl.save(tmp)
            return MODEL_TYPE.load(tmp, env=None, device="cpu")

        vec_env = build_train_vecenv_selfplay(save_path, OPP_POOL, _bootstrap_copy)

        # load vector normalizer state into the freshly constructed env
        vec_env = VecNormalize.load(vecnorm_path, vec_env)
        vec_env.training = True
        vec_env.norm_reward = True
        vec_env.norm_obs = (not USE_TEXT)

        if not USE_TEXT:
            vec_env = RNDVecWrapper(
                vec_env, embedding_dim=128, learning_rate=1e-4,
                intrinsic_reward_scale=0.1, rew_norm=False
            )

        model = MODEL_TYPE.load(model_path, env=vec_env, device="auto")
        steps_done = model.num_timesteps
        first = False

    else:
        # Fresh training: construct env first so we can infer vocab/check shapes and build policy kwargs
        # We'll temporarily create a single self-play env with a bootstrap that will be overwritten after model exists.
        _tmp_bootstrap = lambda: None  # no model to copy yet
        test_env = make_selfplay_env(OPP_POOL, _tmp_bootstrap)
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

        # Build policy kwargs
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

        # Now create the real vectorized env with a live bootstrap copy function
        def _current_model_copy():
            tmp = "/tmp/_sp_bootstrap.zip"
            model.save(tmp)
            return MODEL_TYPE.load(tmp, env=None, device="cpu")

        vec_env = build_train_vecenv_selfplay(save_path, OPP_POOL, _current_model_copy)

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

    # After model exists, redefine bootstrap function to copy *this* model
    def _current_model_copy():
        tmp = "/tmp/_sp_bootstrap.zip"
        model.save(tmp)
        return MODEL_TYPE.load(tmp, env=None, device="cpu")

    # Build env (again) to ensure all workers use the up-to-date bootstrap method
    vec_env = build_train_vecenv_selfplay(save_path, OPP_POOL, _current_model_copy)
    model.set_env(vec_env)

    # =========================
    # Train Loop
    # =========================
    while steps_done < TOTAL_STEPS:
        chunk = min(CHUNK_STEPS, TOTAL_STEPS - steps_done)
        model.learn(total_timesteps=chunk, reset_num_timesteps=first,
                    progress_bar=True, callback=callback)
        first = False
        steps_done += chunk

        # Make newly saved checkpoints available to the opponent pool
        OPP_POOL.refresh()

        # Rebuild env to reshuffle workers and pick new opponents
        vec_env.close()
        vec_env = build_train_vecenv_selfplay(save_path, OPP_POOL, _current_model_copy)
        model.set_env(vec_env)

        # Decay team bandit (same as your original)
        TEAM_BANDIT.decay(rate=0.99)

        # Logging
        tm = TEAM_BANDIT.alpha / (TEAM_BANDIT.alpha + TEAM_BANDIT.beta)
        print(f"[Chunk] steps_done={steps_done} | team_means={np.round(tm,3).tolist()}")