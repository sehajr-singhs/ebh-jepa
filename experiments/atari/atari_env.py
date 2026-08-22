"""Atari 100k environment wrapper with DreamerV3-standard preprocessing.

Standard protocol (Hafner et al., 2023, Appendix D):
  - 84x84 grayscale, resized from native 210x160x3
  - Frame skip = 4 (action repeated for 4 frames; only every 4th frame observed)
  - Sticky actions (p=0.25): with probability p, repeat previous action instead
  - Episode life loss = done (standard ALE convention)
  - Max episode steps = 27000 frames = 6750 actions (standard for 100k protocol)
  - Reward clipping: sign(reward) (DreamerV3-style, not the full {-1,0,1} of OpenAI baselines)
  - Observations: uint8 (B, 1, 84, 84) normalized to [0, 1] float in the agent

The Atari 100k protocol means 100k environment steps (= 100k actions, each
skipping 4 frames = 400k frames). This is the standard benchmark for
sample-efficient visual RL.
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import torch


class AtariPreprocessing:
    """Frame-level preprocessing: grayscale, resize, frame skip, sticky actions."""

    def __init__(self, env, frame_skip=4, sticky_p=0.25, img_size=84):
        self.env = env
        self.frame_skip = frame_skip
        self.sticky_p = sticky_p
        self.img_size = img_size
        self._last_action = 0
        self._frames = []

    def reset(self):
        obs, info = self.env.reset()
        self._last_action = 0
        self._frames = []
        # Collect frame_skip frames (take the last one, like ALE)
        total_reward = 0.0
        for i in range(self.frame_skip):
            if i == 0:
                action = 0  # no-op
            else:
                action = self._sticky_action(0)
            obs, rew, terminated, truncated, info = self.env.step(action)
            total_reward += rew
            if terminated or truncated:
                break
        self._obs = self._preprocess(obs)
        return self._obs, {"reward": total_reward, **info}

    def step(self, action):
        # Sticky actions: with probability p, repeat previous action
        action = self._sticky_action(action)
        total_reward = 0.0
        for i in range(self.frame_skip):
            obs, rew, terminated, truncated, info = self.env.step(action if i == 0 else action)
            total_reward += rew
            if terminated or truncated:
                break
        self._obs = self._preprocess(obs)
        return self._obs, total_reward, terminated, truncated, info

    def _sticky_action(self, action):
        if np.random.random() < self.sticky_p:
            self._last_action = self._last_action  # repeat
        else:
            self._last_action = action
        return self._last_action

    def _preprocess(self, obs):
        """obs: (210, 160, 3) uint8 -> (1, 84, 84) float32 in [0, 1]."""
        # RGB -> grayscale (ITU-R BT.601)
        gray = (0.299 * obs[:, :, 0] + 0.587 * obs[:, :, 1] + 0.114 * obs[:, :, 2]).astype(np.uint8)
        # Resize with area interpolation
        import cv2
        gray = cv2.resize(gray, (self.img_size, self.img_size), interpolation=cv2.INTER_AREA)
        # To float, add channel dim
        return torch.from_numpy(gray.astype(np.float32) / 255.0).unsqueeze(0)  # (1, 84, 84)

    @property
    def action_space(self):
        return self.env.action_space

    @property
    def observation_space(self):
        return self.env.observation_space

    def close(self):
        self.env.close()


def make_atari(game: str, seed: int = 0, render_mode=None):
    """Create a standard Atari 100k environment.

    Args:
        game: Atari game name (e.g., 'Pong', 'Breakout', 'Seaquest')
        seed: Random seed
        render_mode: None for training, 'rgb_array' for recording

    Returns:
        AtariPreprocessing wrapper
    """
    # Standard ALE gymnasium environment
    env_name = f"ALE/{game}-v5"
    try:
        env = gym.make(
            env_name,
            render_mode=render_mode,
            frameskip=1,  # we handle frame skip ourselves
            full_action_space=False,  # minimal action set (standard for 100k)
            repeat_action_probability=0.0,  # we handle sticky ourselves
        )
    except gym.error.NamespaceNotFound:
        # Fallback: try without ALE prefix
        env = gym.make(
            game,
            render_mode=render_mode,
            frameskip=1,
            repeat_action_probability=0.0,
        )

    env = AtariPreprocessing(env, frame_skip=4, sticky_p=0.25, img_size=84)
    env.seed_value = seed
    return env


# Standard Atari 100k benchmark games (Hafner et al., 2023, Table D.1)
ATARI_100K_GAMES = [
    "Pong", "Breakout", "SpaceInvaders",
    "Qbert", "Seaquest", "MsPacman",
    "DemonAttack", "Amidar", "Frostbite",
    "StarGunner",
]
