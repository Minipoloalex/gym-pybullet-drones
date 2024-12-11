from __future__ import annotations

import random
import sys
from typing import Any, Dict, Optional

import gymnasium as gym
import numpy as np

from sample_factory.algo.utils.context import global_model_factory
from sample_factory.cfg.arguments import parse_full_cfg, parse_sf_args
from sample_factory.envs.env_utils import RewardShapingInterface, TrainingInfoInterface, register_env
from sample_factory.train import run_rl
from sf_examples.train_custom_env_custom_model import make_custom_encoder, override_default_params

from gym_pybullet_drones.envs.BaseAviary import DroneModel, Physics
from gym_pybullet_drones.envs.multi_agent_rl.FlockAviary import FlockAviary
from gym_pybullet_drones.envs.multi_agent_rl.LeaderFollowerAviary import LeaderFollowerAviary
from gym_pybullet_drones.envs.multi_agent_rl.MeetupAviary import MeetupAviary
from gym_pybullet_drones.envs.single_agent_rl.BaseSingleAgentAviary import ActionType, ObservationType
from gym_pybullet_drones.utils.Logger import Logger

class CustomMultiEnv(LeaderFollowerAviary, TrainingInfoInterface, RewardShapingInterface):
    """
    Custom multi-agent environment integrating gym_pybullet_drones with Sample Factory.
    Implements a leader-follower scenario with two drones.
    """

    def __init__(self, full_env_name, cfg, render_mode: Optional[str] = None):
        TrainingInfoInterface.__init__(self)
        RewardShapingInterface.__init__(self)

        self.name = full_env_name  # optional
        self.cfg = cfg
        self.curr_episode_steps = 0
        self.res = 8  # 8x8 images
        self.channels = 1  # it's easier when the channel dimension is present, even if it's 1

        # Define observation and action spaces
        self.observation_space = gym.spaces.Box(0, 1, (self.channels, self.res, self.res), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(2)

        self.num_agents = 2
        self.is_multiagent = True

        self.inactive_steps = [3] * self.num_agents

        self.episode_rewards = [[] for _ in range(self.num_agents)]

        self.reward_shaping = [dict(rew=-1.0) for _ in range(self.num_agents)]

        self.obs = self._computeObs()

        self.render_mode = render_mode

    def _obs(self):
        if self.obs is None:
            self.obs = [np.float32(np.random.rand(self.channels, self.res, self.res)) for _ in range(self.num_agents)]
        return self.obs

    def reset(self, **kwargs):
        self.curr_episode_steps = 0
        self.episode_rewards = [[] for _ in range(self.num_agents)]
        return self._computeObs(), [dict() for _ in range(self.num_agents)]

    def step(self, action):
        """Steps the environment with the provided action."""
        obs, reward, terminated, truncated, info = super().step(action)
        self.curr_episode_steps += 1

        # Optionally, implement custom reward shaping here
        # Example: Modify rewards based on some criteria
        # for agent_idx in range(self.num_agents):
        #     self.episode_rewards[agent_idx].append(reward[agent_idx])

        return obs, reward, terminated, truncated, info

    def get_default_reward_shaping(self) -> Optional[Dict[str, Any]]:
        return self.reward_shaping[0]

    def set_reward_shaping(self, reward_shaping: Dict[str, Any], agent_idx: int | slice) -> None:
        if isinstance(agent_idx, int):
            agent_idx = slice(agent_idx, agent_idx + 1)
        for idx in range(agent_idx.start, agent_idx.stop):
            self.reward_shaping[idx] = reward_shaping

    def render(self, mode='human'):
        """Renders the environment."""
        return super().render(mode)

def make_custom_multi_env_func(full_env_name, cfg=None, _env_config=None, render_mode: Optional[str] = None):
    return CustomMultiEnv(full_env_name, cfg, render_mode=render_mode)

def add_extra_params_func(parser):
    """
    Specify any additional command line arguments for this family of custom environments.
    """
    p = parser
    p.add_argument("--custom_env_episode_len", default=10, type=int, help="Number of steps in the episode")

def register_custom_components():
    register_env("my_custom_multi_env_v1", make_custom_multi_env_func)
    global_model_factory().register_encoder_factory(make_custom_encoder)

def parse_custom_args(argv=None, evaluation=False):
    parser, cfg = parse_sf_args(argv=argv, evaluation=evaluation)
    add_extra_params_func(parser)
    override_default_params(parser)
    # Second parsing pass yields the final configuration
    cfg = parse_full_cfg(parser, argv)
    return cfg

def main():
    """Script entry point."""
    register_custom_components()
    cfg = parse_custom_args()
    status = run_rl(cfg)
    return status

if __name__ == "__main__":
    sys.exit(main())
