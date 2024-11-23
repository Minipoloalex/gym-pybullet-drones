import gymnasium as gym
import torch
import numpy as np
from tianshou.data import Collector, VectorReplayBuffer
from tianshou.env import DummyVectorEnv
from tianshou.policy import PPOPolicy
from tianshou.trainer import OnpolicyTrainer
from tianshou.utils.net.common import ActorCritic, Net
from tianshou.utils.net.continuous import ActorProb, Critic
from torch.distributions import Independent, Normal

# Import the environments from gym_pybullet_drones
from gym_pybullet_drones.envs.BaseAviary import BaseAviary, DroneModel
from gym_pybullet_drones.envs.multi_agent_rl.LeaderFollowerAviary import LeaderFollowerAviary
from gym_pybullet_drones.envs.single_agent_rl.BaseSingleAgentAviary import ActionType, ObservationType

# Environment parameters
NUM_DRONES = 2  # Define the number of drones as needed
AGGR_PHY_STEPS = 5  # This parameter is used in gym_pybullet_drones environments
OBS = ObservationType.KIN  # Type of observation
ACT = ActionType.RPM  # Type of action


# Create a multi-agent environment wrapper
class MultiAgentEnvWrapper(gym.Env):
    def __init__(self, env):
        self.env = env
        self.num_drones = self.env.NUM_DRONES
        # Combine observation and action spaces
        obs_space = self.env.observation_space[0]
        action_space = self.env.action_space[0]
        self.observation_space = gym.spaces.Box(
            low=np.tile(obs_space.low, self.num_drones),
            high=np.tile(obs_space.high, self.num_drones),
            dtype=obs_space.dtype
        )
        self.action_space = gym.spaces.Box(
            low=np.tile(action_space.low, self.num_drones),
            high=np.tile(action_space.high, self.num_drones),
            dtype=action_space.dtype
        )

    def reset(self):
        obs_dict = self.env.reset()  # Gymnasium might return only observation
        if isinstance(obs_dict, tuple):
            # Handle case where reset returns (observation, info)
            obs_dict, _ = obs_dict
        # Concatenate observations from multiple drones into a single vector
        obs_concat = np.concatenate([obs_dict[i] for i in range(self.num_drones)])
        return obs_concat, {}

    def step(self, action):
        action_dim = self.env.action_space[0].shape[0]
        actions = {i: action[i * action_dim:(i + 1) * action_dim] for i in range(self.num_drones)}

        obs_dict, rewards, done, info = self.env.step(actions)

        obs_concat = np.concatenate([obs_dict[i] for i in range(self.num_drones)])

        reward = np.mean(list(rewards.values()))

        terminated_env = any(done.values())
        truncated_env = False  
        info = {str(k): v for k, v in info.items()}
        return obs_concat, reward, terminated_env, truncated_env, info

    def render(self, mode='human'):
        return self.env.render(mode)

    def close(self):
        return self.env.close()

# Create function to make the environment
def make_env():
    env = LeaderFollowerAviary(num_drones=NUM_DRONES, aggregate_phy_steps=AGGR_PHY_STEPS, obs=OBS, act=ACT)
    env = MultiAgentEnvWrapper(env) 
    return env



# Create instances of the training and testing environments
train_envs = DummyVectorEnv([make_env for _ in range(8)])
test_envs = DummyVectorEnv([make_env for _ in range(8)])

# Define policy network
state_shape = train_envs.observation_space[0].shape
action_shape = train_envs.action_space[0].shape

print(state_shape)
print(action_shape)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

max_action = torch.tensor(train_envs.action_space[0].shape, dtype=torch.float32, device=device)

net_a = Net(state_shape=state_shape, hidden_sizes=[256, 256], device=device)
actor = ActorProb(
    net_a, action_shape=action_shape, max_action=max_action, device=device
).to(device)
net_c = Net(state_shape=state_shape, hidden_sizes=[256, 256], device=device)
critic = Critic(net_c, device=device).to(device)
actor_critic = ActorCritic(actor, critic)
optim = torch.optim.Adam(actor_critic.parameters(), lr=3e-4)

# Define distribution function
def dist(*logits):
    return Independent(Normal(*logits), 1)

# PPO Policy
policy = PPOPolicy(
    actor,
    critic,
    optim,
    dist_fn=dist,
    action_space=train_envs.action_space[0],
    deterministic_eval=True,
)

print(train_envs)
print(test_envs)

# Collectors
train_collector = Collector(policy, train_envs)
test_collector = Collector(policy, test_envs)

# Trainer
result = OnpolicyTrainer(
    policy=policy,
    train_collector=train_collector,
    test_collector=test_collector,
    max_epoch=80,
    step_per_epoch=10000,
    repeat_per_collect=4,
    episode_per_test=10,
    batch_size=256,
    step_per_collect=2000,
    stop_fn=lambda mean_rewards: mean_rewards >= 200,
).run()

print(f'Finished training! Use {result["duration"]}')

# Test the trained policy
policy.eval()
test_collector.reset()
result = test_collector.collect(n_episode=1, render=1 / 240)
print(f'Final reward: {result["rews"].mean()}, length: {result["lens"].mean()}')

# Close environments
train_envs.close()
test_envs.close()
