"""Learning script for multi-agent problems.

Agents are based on `ray[rllib]`'s implementation of PPO and use a custom centralized critic.

Example
-------
To run the script, type in a terminal:

    $ python multiagent.py --num_drones <num_drones> --env <env> --obs <ObservationType> --act <ActionType> --algo <alg> --num_workers <num_workers>

Notes
-----
Check Ray's status at:
    http://127.0.0.1:8265

"""
import os
import time
import argparse
from datetime import datetime
import subprocess
import pdb
import math
import numpy as np
import pybullet as p
import pickle
import matplotlib.pyplot as plt
import gym
from gym import error, spaces, utils
from gym.utils import seeding
from gym.spaces import Box, Dict
import torch
import torch.nn as nn
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork
import ray
from ray import tune
from ray.tune.logger import DEFAULT_LOGGERS
from ray.tune import register_env
from ray.rllib.agents import ppo
from ray.rllib.agents.ppo import PPOTrainer, PPOTFPolicy
from ray.rllib.examples.policy.random_policy import RandomPolicy
from ray.rllib.utils.test_utils import check_learning_achieved
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models import ModelCatalog
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.env.multi_agent_env import ENV_STATE

from gym_pybullet_drones.envs.BaseAviary import DroneModel, Physics

from gym_pybullet_drones.envs.multi_agent_rl.ChaseAviary import ChaseAviary
from gym_pybullet_drones.envs.single_agent_rl.BaseSingleAgentAviary import ActionType, ObservationType
from gym_pybullet_drones.utils.Logger import Logger

import shared_constants


OWN_OBS_VEC_SIZE = None # Modified at runtime
ACTION_VEC_SIZE = None  # Modified at runtime
NUM_DRONES = None       # Modified at runtime
#### Useful links ##########################################
# Workflow: github.com/ray-project/ray/blob/master/doc/source/rllib-training.rst
# ENV_STATE example: github.com/ray-project/ray/blob/master/rllib/examples/env/two_step_game.py
# Competing policies example: github.com/ray-project/ray/blob/master/rllib/examples/rock_paper_scissors_multiagent.py

############################################################
class CustomTorchCentralizedCriticModel(TorchModelV2, nn.Module):
    """Multi-agent model that implements a centralized value function.

    It assumes the observation is a dict with 'own_obs' and 'opponent_obs', the
    former of which can be used for computing actions (i.e., decentralized
    execution), and the latter for optimization (i.e., centralized learning).

    This model has two parts:
    - An action model that looks at just 'own_obs' to compute actions
    - A value model that also looks at the 'opponent_obs' / 'opponent_action'
      to compute the value (it does this by using the 'obs_flat' tensor).
    """

    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)
        self.action_model = FullyConnectedNetwork(
                                                  Box(low=-1, high=1, shape=(OWN_OBS_VEC_SIZE, )), 
                                                  action_space,
                                                  num_outputs,
                                                  model_config,
                                                  name + "_action"
                                                  )
        self.value_model = FullyConnectedNetwork(
                                                 obs_space, 
                                                 action_space,
                                                 1, 
                                                 model_config, 
                                                 name + "_vf"
                                                 )
        self._model_in = None

    def forward(self, input_dict, state, seq_lens):
        self._model_in = [input_dict["obs_flat"], state, seq_lens]
        return self.action_model({"obs": input_dict["obs"]["own_obs"]}, state, seq_lens)

    def value_function(self):
        value_out, _ = self.value_model({"obs": self._model_in[0]}, self._model_in[1], self._model_in[2])
        return torch.reshape(value_out, [-1])

def central_critic_observer(agent_obs, **kw):
    new_obs = {
        x: {"own_obs": agent_obs[x],} for x in range(NUM_DRONES)
    }
    return new_obs

############################################################
if __name__ == "__main__":
    NUM_DRONES = 2

    #### Define and parse (optional) arguments for the script ##
    # Actually, here we should not pass parameters
    # It is kept to make it easier to create the filenames and do everything else
    parser = argparse.ArgumentParser(description='Multi-agent reinforcement learning experiments script')
    parser.add_argument('--num_drones',  default=NUM_DRONES,        type=int,                                                                                        help=f'Number of drones (default: {NUM_DRONES})', metavar='')
    parser.add_argument('--env',         default='chase',  type=str,             choices=['chase'], help='Help (default: ..)', metavar='')
    parser.add_argument('--obs',         default='kin',             type=ObservationType,                                                                            help='Help (default: ..)', metavar='')
    parser.add_argument('--act',         default='one_d_rpm',       type=ActionType,                                                                                 help='Help (default: ..)', metavar='')
    parser.add_argument('--algo',        default='cc',              type=str,             choices=['cc'],                                                            help='Help (default: ..)', metavar='')
    parser.add_argument('--workers',     default=0,                 type=int,                                                                                        help='Help (default: ..)', metavar='')        
    ARGS = parser.parse_args()
    assert(ARGS.num_drones == NUM_DRONES)
    assert(ARGS.obs == ObservationType.KIN)
    assert(ARGS.act == ActionType.ONE_D_RPM)
    assert(ARGS.algo == 'cc')

    #### Save directory ########################################
    filename = os.path.dirname(os.path.abspath(__file__))+'/results/save-'+ARGS.env+'-'+str(ARGS.num_drones)+'-'+ARGS.algo+'-'+ARGS.obs.value+'-'+ARGS.act.value+'-'+datetime.now().strftime("%m.%d.%Y_%H.%M.%S")
    if not os.path.exists(filename):
        os.makedirs(filename+'/')

    #### Print out current git commit hash #####################
    git_commit = subprocess.check_output(["git", "describe", "--tags"]).strip()
    with open(filename+'/git_commit.txt', 'w+') as f:
        f.write(str(git_commit))

    #### Constants, and errors #################################
    TIMESTEPS = 500_000
    OWN_OBS_VEC_SIZE = 3
    ACTION_VEC_SIZE = 1

    #### Uncomment to debug slurm scripts ######################
    # exit()

    #### Initialize Ray Tune ###################################
    ray.shutdown()
    ray.init(ignore_reinit_error=True)

    #### Register the custom centralized critic model ##########
    ModelCatalog.register_custom_model("cc_model", CustomTorchCentralizedCriticModel)

    #### Register the environment ##############################
    temp_env_name = f"{ARGS.env}-aviary-v0"
    register_env(temp_env_name, lambda _: ChaseAviary(num_drones=ARGS.num_drones,
                                                        aggregate_phy_steps=shared_constants.AGGR_PHY_STEPS,
                                                        obs=ARGS.obs,
                                                        act=ARGS.act
                                                        )
                     )

    #### Unused env to extract the act and obs spaces ##########
    temp_env = ChaseAviary(num_drones=ARGS.num_drones,
                                aggregate_phy_steps=shared_constants.AGGR_PHY_STEPS,
                                obs=ARGS.obs,
                                act=ARGS.act
                                )

    observer_space = Dict({
        "own_obs": temp_env.observation_space[0],
    })
    action_space = temp_env.action_space[0]

    #### Note ##################################################
    # RLlib will create ``num_workers + 1`` copies of the
    # environment since one copy is needed for the driver process.
    # To avoid paying the extra overhead of the driver copy,
    # which is needed to access the env's action and observation spaces,
    # you can defer environment initialization until ``reset()`` is called

    #### Set up the trainer's config ###########################
    config = ppo.DEFAULT_CONFIG.copy() # For the default config, see github.com/ray-project/ray/blob/master/rllib/agents/trainer.py
    config = {
        "env": temp_env_name,
        "num_workers": 0 + ARGS.workers,
        "num_gpus": int(os.environ.get("RLLIB_NUM_GPUS", "0")), # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0
        "batch_mode": "complete_episodes",
        "framework": "torch",
    }

    #### Set up the model parameters of the trainer's config ###
    config["model"] = { 
        "custom_model": "cc_model",
    }
    #### Set up the multiagent params of the trainer's config ##

    # Use several policies
    config["multiagent"] = {
        "policies": {
            f"pol{x}": (None, observer_space, action_space, {"agent_id": x,})
            for x in range(NUM_DRONES)
        },
        "policy_mapping_fn": lambda x: f"pol{x}", # Function mapping agent ids to policy ids
        "observation_fn": central_critic_observer, # See rllib/evaluation/observation_function.py for more info
    }

    #### Ray Tune stopping conditions ##########################
    stop = {
        "timesteps_total": TIMESTEPS,
        # "episode_reward_mean": 0,
        # "training_iteration": 0,
    }

    #### Train #################################################
    results = tune.run(
        "PPO",
        stop=stop,
        config=config,
        verbose=True,
        checkpoint_at_end=True,
        local_dir=filename,
    )
    # check_learning_achieved(results, 1.0)

    #### Save agent ############################################
    checkpoints = results.get_trial_checkpoints_paths(trial=results.get_best_trial('episode_reward_mean', mode='max'),
                                                      metric='episode_reward_mean')
    with open(filename+'/checkpoint.txt', 'w+') as f:
        f.write(checkpoints[0][0])

    #### Shut down Ray #########################################
    ray.shutdown()
