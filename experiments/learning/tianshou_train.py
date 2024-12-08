
import gym
import gym.spaces
import numpy as np
import torch
import tianshou

from tianshou.policy import MultiAgentPolicyManager
from tianshou.trainer import OnpolicyTrainer
from tianshou.utils import TensorboardLogger
from tianshou.policy import PPOPolicy
from tianshou.data import Collector, VectorReplayBuffer
from tianshou.env import DummyVectorEnv
from tianshou.utils.net.common import ActorCritic, Net
from tianshou.utils.net.continuous import ActorProb, Critic
from tianshou.utils.net.continuous import Actor, Critic
from tianshou.policy import MultiAgentPolicyManager

from torch.distributions import Distribution, Uniform

from gym_pybullet_drones.envs.BaseAviary import DroneModel
from gym_pybullet_drones.envs.multi_agent_rl.LeaderFollowerAviary import LeaderFollowerAviary
from gym_pybullet_drones.envs.multi_agent_rl.MeetAtHeightAviary import MeetAtHeightAviary
from gym_pybullet_drones.envs.single_agent_rl.BaseSingleAgentAviary import ActionType, ObservationType

class WrapperMA(gym.Env):
    rllib_env: MeetAtHeightAviary
    num_drones: int
    rllib_env_obs_size: int
    rllib_env_act_size: int
    # TODO: seed / np_random is not implemented

    """
    Logic: flatten environment observations and actions
    """

    def __init__(self, num_drones: int) -> None:
        self.rllib_env = MeetAtHeightAviary(num_drones = num_drones, aggregate_phy_steps = 5)
        self.num_drones = num_drones
        
        observation_space = self.rllib_env._observationSpace()
        self.observation_space = gym.spaces.Box(
            low  = np.hstack([observation_space[i].low  for i in range(num_drones)]),
            high = np.hstack([observation_space[i].high for i in range(num_drones)]),
        )
        self.rllib_env_obs_size = observation_space[0].shape[0]
        
        action_space = self.rllib_env._actionSpace()
        self.action_space = gym.spaces.Box(
            low  = np.hstack([action_space[i].low  for i in range(num_drones)]),
            high = np.hstack([action_space[i].high for i in range(num_drones)]),
        )
        self.rllib_env_act_size = action_space[0].shape[0]
        
        return
    
    def dict_to_flatten(self, dictionary: dict, size: int):
        array = np.array(list(dictionary.values()))
        flattened = array.reshape((self.num_drones * size,))
        return flattened
    
    def flatten_to_dict(self, flattened: np.ndarray, size: int):
        array = flattened.reshape((self.num_drones, size))
        dictionary = { i: array[i] for i in range(self.num_drones)}
        return dictionary
    
    def step(self, action: np.ndarray):
        # Transform the flattened action array into an array of actions for each drone
        action_dict = self.flatten_to_dict(action, self.rllib_env_act_size)

        # Use the underlying environment to perform the `step()` function
        observation_dict, reward_dict, terminated, _ = self.rllib_env.step(action_dict)
        
        # Transform the dictionary of drone observation into a flattened observation array
        observation = self.dict_to_flatten(observation_dict, self.rllib_env_obs_size)
        
        # Transform the rewards into something useful?
        reward = 0 # TODO: this value might be very wrong?
        
        # I'm using the `info` return value to return the rewards for each agent
        return observation, reward, terminated["__all__"], reward_dict

    def reset(self):
        observation_dict = self.rllib_env.reset()
        observation = self.dict_to_flatten(observation_dict, self.rllib_env_obs_size)
        return observation

    def render(self):
        return

    def close(self):
        self.rllib_env.close()
        return

# class PolicyMA(MultiAgentPolicyManager):
#     def __init__():
#         return


if __name__ == "__main__":
    import os

    n_drones = 5
    n_train_envs = 4
    n_test_envs = 2
    buffer_size = 10240
    batch_size = 1024
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # TODO: these 2 should come from env
    obs_size = 3
    act_size = 1
    
    train_envs = DummyVectorEnv([lambda : WrapperMA(num_drones = n_drones) for _ in range(n_train_envs)])
    test_envs  = DummyVectorEnv([lambda : WrapperMA(num_drones = n_drones) for _ in range(n_test_envs )])
    
    np.random.seed(42)
    torch.manual_seed(42)
    train_envs.seed(42)
    test_envs.seed(0)
    actor = Net((obs_size,), (act_size,), (64, 64)) # TODO: set device
    critic = Net((obs_size,), (act_size,), (32, 64, 32)) # TODO: set device
    ac = ActorCritic(actor, critic)

    aux = WrapperMA(num_drones = n_drones)
    distribution = Uniform(torch.tensor(aux.action_space.low[0]), torch.tensor(aux.action_space.high))
    policy = PPOPolicy(
        actor = actor,
        critic = critic,
        optim = torch.optim.Adam(ac.parameters()),
        dist_fn = Distribution(), # TODO
    )

    train_collector = Collector(
        policy, train_envs, VectorReplayBuffer(buffer_size, n_train_envs),
        exploration_noise = True # TODO: check if we need this
    )

    test_collector = Collector(policy, test_envs, exploration_noise = True)
    train_collector.collect(n_step=batch_size * n_train_envs)

    # log_path = os.path.join("logs", 'tic_tac_toe', 'dqn')
    # writer = SummaryWriter(log_path)
    # writer.add_text("args", str(""))
    # logger = TensorboardLogger(writer)

    # def save_best_fn(policy):
    #     model_save_path = os.path.join(
    #         "logs", 'tic_tac_toe', 'dqn', 'policy.pth'
    #     )
    #     torch.save(
    #         policy.policies[agents[args.agent_id - 1]].state_dict(), model_save_path
    #     )
    # def test_fn(epoch, env_step):
    #     policy.policies[agents[args.agent_id - 1]].set_eps(args.eps_test)

    # def train_fn(epoch, env_step):
    #     policy.policies[agents[args.agent_id - 1]].set_eps(args.eps_train)

    # def reward_metric(rews):
    #     return rews[:, args.agent_id - 1]

    # tianshou.trainer.OnpolicyTrainer(
    #     policy = policy,
    #     train_collector = train_collector,
    #     test_collector = test_collector,
    #     epoch = epoch,
    #     step_per_epoch = step_per_epoch,
    #     repeat_per_collect = repeat_per_collect,
    #     episode_per_test = episode_per_test,
    #     batch_size = batch_size,
    #     step_per_collect = step_per_collect,
    #     episode_per_collect = episode_per_collect,
    #     train_fn = train_fn,
    #     test_fn = test_fn,
    #     stop_fn = stop_fn,
    #     save_best_fn = save_best_fn,
    #     save_checkpoint_fn = save_checkpoint_fn,
    #     resume_from_log = resume_from_log,
    #     reward_metric = reward_metric,
    #     logger = logger,
    #     verbose = True,
    #     show_progress = True,
    #     test_in_train = True,
    # )

    # result = OnpolicyTrainer(
    #     policy,
    #     train_collector,
    #     test_collector,
    #     epoch,
    #     step_per_epoch,
    #     repeat_per_collect,
    #     n_test_envs,
    #     batch_size,
    #     train_fn=train_fn,
    #     test_fn=test_fn,
    #     stop_fn=stop_fn,
    #     save_best_fn=save_best_fn,
    #     update_per_step=args.update_per_step,
    #     logger=logger,
    #     test_in_train=False,
    #     reward_metric=reward_metric
    # )
