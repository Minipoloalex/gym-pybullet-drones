from __future__ import annotations

import sys
from typing import Any, Dict, Optional

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from sample_factory.algo.utils.context import global_model_factory
from sample_factory.cfg.arguments import parse_full_cfg, parse_sf_args
from sample_factory.envs.env_utils import RewardShapingInterface, TrainingInfoInterface, register_env
from sample_factory.train import run_rl
from sf_examples.train_custom_env_custom_model import override_default_params

from gym_pybullet_drones.envs.BaseAviary import DroneModel, Physics
from gym_pybullet_drones.envs.multi_agent_rl.FlockAviary import FlockAviary
from gym_pybullet_drones.envs.multi_agent_rl.LeaderFollowerAviary import LeaderFollowerAviary
from gym_pybullet_drones.envs.multi_agent_rl.MeetupAviary import MeetupAviary
from gym_pybullet_drones.envs.single_agent_rl.BaseSingleAgentAviary import ActionType, ObservationType
from gym_pybullet_drones.utils.Logger import Logger

class CustomEncoder(nn.Module):
    def __init__(self, cfg, obs_space):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1), 
            nn.ReLU(),
            nn.Flatten(),  
            nn.Linear(8 * 4 * 6, 64),  
            nn.ReLU()
        )
        self.output_dim = 64  

    def forward(self, x):
        """
        Forward pass of the encoder. Extracts the observation tensor from TensorDict.
        
        Parameters:
        x : TensorDict
            Input data dictionary containing observations and other metadata.
        """
        obs_tensor = x["obs"]  
        
        obs_tensor = obs_tensor.view(-1, 1, 4, 6)
        
        return self.conv(obs_tensor)

    
    def get_out_size(self):
        """
        Return the output size (feature dimension) of the encoder.
        This is required by Sample Factory.
        """
        return self.output_dim
    def device_for_input_tensor(self, input_tensor_name):
        """
        Returns the device (CPU/GPU) where the input tensor should be placed.
        This ensures compatibility with the Sample Factory framework.
        """
        return next(self.parameters()).device
    def type_for_input_tensor(self, input_tensor_name):
        """
        Returns the data type (e.g., torch.float32) for the input tensor.
        """
        return torch.float32

class CustomMultiEnv(LeaderFollowerAviary, TrainingInfoInterface, RewardShapingInterface):
    """
    Custom multi-agent environment with a leader-follower structure.
    """

    def __init__(self, full_env_name, cfg, render_mode: Optional[str] = None):
        super().__init__(
            drone_model=None,
            num_drones=2,
            neighbourhood_radius=10,
            initial_xyzs=None,
            initial_rpys=None,
            physics=None,
            freq=240,
            aggregate_phy_steps=1,
            gui=render_mode == "human",
            record=False,
        )

        self.name = full_env_name
        self.cfg = cfg
        self.num_agents = 2
        self.obs_dim = 12  
        self.action_dim = 4

        self.obs_lower_bound = np.array([
            -1, -1,  0,  
            -1, -1, -1, 
            -1, -1, -1,  
            -1, -1, -1   
        ])
        self.obs_upper_bound = np.array([
            1,  1,  1,   
            1,  1,  1,   
            1,  1,  1,   
            1,  1,  1    
        ])

        combined_obs_size = self.obs_dim * self.num_agents
        self.combined_low = np.tile(self.obs_lower_bound, self.num_agents)  
        self.combined_high = np.tile(self.obs_upper_bound, self.num_agents) 
        self.reward_shaping: Dict[str, Any] = dict(action_rew_coeff=0.01)
        self.observation_space = gym.spaces.Box(
            low=self.combined_low,
            high=self.combined_high,
            dtype=np.float32
        )

        self.action_space = gym.spaces.Box(
            low=-1, high=1, shape=(self.action_dim,), dtype=np.float32
        )


    def step(self, action):
        """Step function to execute actions and provide joint observations."""
        action = np.array(action)

        action_dict = {i: action[i].reshape(4,) for i in range(self.num_agents)} 

        obs_dict, reward, terminated, truncated, info = super().step(action_dict)
        obs = [
            np.clip(np.concatenate([obs_dict[0], obs_dict[1]]), self.combined_low, self.combined_high),
            np.clip(np.concatenate([obs_dict[1], obs_dict[0]]), self.combined_low, self.combined_high),
        ]

        reward = self._compute_rewards(obs_dict[0], obs_dict[1])

        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        """Reset environment and return combined initial observations."""
        self.curr_episode_steps = 0

        obs_dict = super().reset()

        obs_0 = obs_dict[0]  
        obs_1 = obs_dict[1]  

        combined_obs = [
            np.clip(np.concatenate([obs_0, obs_1]), self.combined_low, self.combined_high),
            np.clip(np.concatenate([obs_1, obs_0]), self.combined_low, self.combined_high),
        ]

        return combined_obs, {}


    def get_default_reward_shaping(self) -> Optional[Dict[str, Any]]:
        """
        Provide a default reward shaping scheme.

        This specifies the weights for the leader's and followers' rewards based on
        distance to their respective target positions.
        """
        return {
            "leader_target_penalty": -1.0,   
            "follower_target_penalty": -0.5  
        }
    def set_reward_shaping(self, reward_shaping: Dict[str, Any], agent_idx: int | slice) -> None:
        self.reward_shaping = reward_shaping


def make_custom_multi_env_func(full_env_name, cfg=None, _env_config=None, render_mode: Optional[str] = None):
    """Factory function to create the custom multi-agent environment."""
    return CustomMultiEnv(full_env_name, cfg, render_mode=render_mode)

def add_extra_params_func(parser):
    """
    Specify any additional command line arguments for this family of custom environments.
    """
    p = parser
    p.add_argument("--custom_env_episode_len", default=10, type=int, help="Number of steps in the episode")

def make_custom_encoder(cfg, obs_space):
    """Factory function to create the CustomEncoder."""
    return CustomEncoder(cfg, obs_space)

def register_custom_components():
    register_env("my_custom_multi_env_v1", make_custom_multi_env_func)
    global_model_factory().register_encoder_factory(make_custom_encoder)

def parse_custom_args(argv=None, evaluation=False):
    """Parse custom arguments."""
    parser, cfg = parse_sf_args(argv=argv, evaluation=evaluation)
    add_extra_params_func(parser)
    override_default_params(parser)
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
