
import gym
import gym.spaces
import numpy as np
import torch
import tianshou

from tianshou.trainer import OnpolicyTrainer
from tianshou.utils import TensorboardLogger
from tianshou.policy import PPOPolicy, MultiAgentPolicyManager, BasePolicy
from tianshou.data import Batch, ReplayBuffer
from tianshou.data import Collector, VectorReplayBuffer
from tianshou.env import DummyVectorEnv
from tianshou.utils.net.common import ActorCritic, Net
from tianshou.utils.net.continuous import ActorProb, Critic
from tianshou.utils.net.continuous import Actor, Critic

from torch.distributions import Distribution, Uniform

from gym_pybullet_drones.envs.BaseAviary import DroneModel
from gym_pybullet_drones.envs.multi_agent_rl.LeaderFollowerAviary import LeaderFollowerAviary
from gym_pybullet_drones.envs.multi_agent_rl.MeetAtHeightAviary import MeetAtHeightAviary
from gym_pybullet_drones.envs.single_agent_rl.BaseSingleAgentAviary import ActionType, ObservationType

def dict_to_flatten(dictionary: dict, n_drones: int, size: int) -> np.ndarray:
    array = np.array(list(dictionary.values()))
    flattened = array.reshape((n_drones * size,))
    return flattened

def flatten_to_dict(flattened: np.ndarray, n_drones: int, size: int) -> dict:
    array = flattened.reshape((n_drones, size))
    dictionary = { i: array[i] for i in range(n_drones)}
    return dictionary

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
    
    def step(self, action: np.ndarray):
        # Transform the flattened action array into an array of actions for each drone
        action_dict = flatten_to_dict(action, self.num_drones, self.rllib_env_act_size)

        # Use the underlying environment to perform the `step()` function
        observation_dict, reward_dict, terminated, _ = self.rllib_env.step(action_dict)
        
        # Transform the dictionary of drone observation into a flattened observation array
        observation = dict_to_flatten(observation_dict, self.num_drones, self.rllib_env_obs_size)
        
        # Transform the rewards into something useful?
        reward = 0 # TODO: this value might be very wrong?
        
        # I'm using the `info` return value to return the rewards for each agent
        return observation, reward, terminated["__all__"], reward_dict

    def reset(self):
        observation_dict = self.rllib_env.reset()
        observation = dict_to_flatten(observation_dict, self.num_drones, self.rllib_env_obs_size)
        return observation

    def render(self):
        return

    def close(self):
        self.rllib_env.close()
        return

class PolicyMA(BasePolicy):
    """
    This multiagent policy assumes that all agents act based on the same individual policy.
    """
    policy: BasePolicy
    num_drones: int
    obs_size: int
    act_size: int
    
    def __init__(self, agent_policy: BasePolicy, num_drones: int, obs_size: int, act_size: int,
                 observation_space = None, action_space = None, action_scaling = False, action_bound_method = "", lr_scheduler = None):
        super().__init__(observation_space, action_space, action_scaling, action_bound_method, lr_scheduler)
        self.policy = agent_policy
        self.num_drones = num_drones
        self.obs_size = obs_size
        self.act_size = act_size
    
    def forward(self, batch, state = None, **kwargs):
        print("forward")
        print(batch)

        # Get the results for each agent
        results = []
        for agent in range(self.num_drones):
            obs_i_s = self.obs_size * agent
            obs_i_e = obs_i_s + self.obs_size
            filtered_batch = Batch(
                obs = batch.obs[:,obs_i_s:obs_i_e],
                act = Batch(),
                info = batch.info,
            )
            out = self.policy(
                batch = filtered_batch,
                state = None if state is None else state[:,obs_i_s:obs_i_e],
                **kwargs
            )
            results.append(out.act)

        act_results = np.array([
            [result]
            for result in results
        ])

        print("finished forward")
        print(act_results)
        return Batch({
            "act" : act_results,
        })
    
    # def process_fn(self, batch: Batch, buffer: ReplayBuffer, indices: np.ndarray) -> Batch:
    #     print(batch)
    #     return super().process_fn(batch, buffer, indices)
    def process_fn(
        self, batch: Batch, buffer: ReplayBuffer, indice: np.ndarray
    ) -> Batch:
        print("process_fn")
        print(batch)
        """Dispatch batch data from obs.agent_id to every policy's process_fn.

        Save original multi-dimensional rew in "save_rew", set rew to the
        reward of each agent during their "process_fn", and restore the
        original reward afterwards.
        """
        results = {}
        # reward can be empty Batch (after initial reset) or nparray.
        has_rew = isinstance(buffer.rew, np.ndarray)
        if has_rew:  # save the original reward in save_rew
            # Since we do not override buffer.__setattr__, here we use _meta to
            # change buffer.rew, otherwise buffer.rew = Batch() has no effect.
            save_rew, buffer._meta.rew = buffer.rew, Batch()
        for agent, policy in self.policies.items():
            agent_index = np.nonzero(batch.obs.agent_id == agent)[0]
            if len(agent_index) == 0:
                results[agent] = Batch()
                continue
            tmp_batch, tmp_indice = batch[agent_index], indice[agent_index]
            if has_rew:
                tmp_batch.rew = tmp_batch.rew[:, self.agent_idx[agent]]
                buffer._meta.rew = save_rew[:, self.agent_idx[agent]]
            if not hasattr(tmp_batch.obs, "mask"):
                if hasattr(tmp_batch.obs, 'obs'):
                    tmp_batch.obs = tmp_batch.obs.obs
                if hasattr(tmp_batch.obs_next, 'obs'):
                    tmp_batch.obs_next = tmp_batch.obs_next.obs
            results[agent] = policy.process_fn(tmp_batch, buffer, tmp_indice)
        if has_rew:  # restore from save_rew
            buffer._meta.rew = save_rew
        return Batch(results)
    
    def learn(self, batch, **kwargs):
        print("learn")
        print(batch)
        return super().learn(batch, **kwargs)

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
    optim  = torch.optim.Adam(ActorCritic(actor, critic).parameters())
    dist   = torch.distributions.Categorical
    ppo_policy = PPOPolicy(actor = actor, critic = critic, optim = optim, dist_fn = dist)
    policy = PolicyMA(ppo_policy, 5, 3, 1)

    train_collector = Collector(
        policy,
        train_envs,
        VectorReplayBuffer(buffer_size, n_train_envs),
        exploration_noise = True,
        # reward_metric = rw
    )

    # test_collector = Collector(policy, test_envs, exploration_noise = True, reward_metric = rw)
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
