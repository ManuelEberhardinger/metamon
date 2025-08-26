import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from stable_baselines3.common.vec_env import VecEnvWrapper
from gymnasium import spaces

class RunningMeanStd:
    def __init__(self, shape, eps=1e-4):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = eps

    def update(self, x: np.ndarray):
        # x: [batch, ...]
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta**2 * self.count * batch_count / tot_count
        new_var = M2 / tot_count
        self.mean, self.var, self.count = new_mean, new_var, tot_count

    def normalize(self, x, clip=5.0):
        x_norm = (x - self.mean) / (np.sqrt(self.var) + 1e-8)
        return np.clip(x_norm, -clip, clip)

class RNDNet(nn.Module):
    def __init__(self, obs_dim, embed_dim=128):
        super().__init__()
        def mlp():
            return nn.Sequential(
                nn.Linear(obs_dim, 256), nn.ReLU(),
                nn.Linear(256, embed_dim)
            )
        self.target = mlp()
        for p in self.target.parameters():
            p.requires_grad = False
        self.predictor = mlp()

    def forward(self, x):  # x: [B, obs_dim]
        with torch.no_grad():
            t = self.target(x)
        p = self.predictor(x)
        return p, t

class RNDVecWrapper(VecEnvWrapper):
    """
    Adds intrinsic reward using Random Network Distillation.
    Works with SB3 VecEnvs (returns (obs, rewards, dones, infos)).
    """
    def __init__(
        self,
        venv,
        intrinsic_reward_scale=0.1,
        obs_norm=True,
        rew_norm=True,
        learning_rate=1e-4,
        embedding_dim=128,
        device="cpu",
    ):
        super().__init__(venv)
        assert isinstance(venv.observation_space, spaces.Box), "RND expects Box obs"
        assert len(venv.observation_space.shape) == 1, "Use a flat 1D feature vector"

        self.device = torch.device(device)
        self.intrinsic_scale = intrinsic_reward_scale

        obs_dim = venv.observation_space.shape[0]
        self.net = RNDNet(obs_dim, embed_dim=embedding_dim).to(self.device)
        self.opt = optim.Adam(self.net.predictor.parameters(), lr=learning_rate)

        self.obs_rms = RunningMeanStd((obs_dim,)) if obs_norm else None
        self.int_rew_rms = RunningMeanStd((1,)) if rew_norm else None

        # last observations for each env (needed because VecEnv step gets actions only)
        self._last_obs = None

    def reset(self):
        obs = self.venv.reset()
        # SB3 VecEnv reset returns obs (no info dict)
        self._last_obs = obs.copy()
        return obs

    def step_async(self, actions):
        self.venv.step_async(actions)

    def step_wait(self):
        obs, ext_rewards, dones, infos = self.venv.step_wait()

        # Use *new* observations to compute curiosity (standard choice)
        batch_obs = obs.astype(np.float32)

        # normalize observations for RND
        if self.obs_rms is not None:
            # update running stats with the incoming batch
            self.obs_rms.update(batch_obs)
            batch_obs = self.obs_rms.normalize(batch_obs)

        # compute intrinsic rewards
        with torch.no_grad():
            pass  # just to be explicit

        x = torch.as_tensor(batch_obs, dtype=torch.float32, device=self.device)
        pred, tgt = self.net(x)
        # mse per sample
        errors = torch.mean((pred - tgt) ** 2, dim=1)  # [n_envs]
        int_rewards = errors.detach().cpu().numpy().reshape(-1, 1)  # [n_envs, 1]

        # train predictor on this batch
        loss = torch.mean(errors)
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        self.opt.step()

        # normalize intrinsic rewards (recommended)
        if self.int_rew_rms is not None:
            self.int_rew_rms.update(int_rewards)
            denom = np.sqrt(self.int_rew_rms.var) + 1e-8
            int_rewards = np.clip(int_rewards / denom, -5.0, 5.0)

        # combine rewards (SB3 expects shape [n_envs])
        ext_rewards = np.asarray(ext_rewards, dtype=np.float32).reshape(-1)
        int_rewards = int_rewards.reshape(-1)
        total_rewards = ext_rewards + self.intrinsic_scale * int_rewards

        self._last_obs = obs.copy()
        return obs, total_rewards, dones, infos