import gymnasium as gym
import torch
import numpy as np
from tianshou.data import Collector
from tianshou.env import DummyVectorEnv
from tianshou.policy import PPOPolicy
from tianshou.trainer import OnpolicyTrainer
from tianshou.utils.net.common import ActorCritic, Net
from tianshou.utils.net.continuous import ActorProb, Critic
from torch.distributions import Independent, Normal

from gym_pybullet_drones.envs.BaseAviary import DroneModel
from gym_pybullet_drones.envs.multi_agent_rl.LeaderFollowerAviary import LeaderFollowerAviary
from gym_pybullet_drones.envs.single_agent_rl.BaseSingleAgentAviary import ActionType, ObservationType

NUM_DRONES = 2
AGGR_PHY_STEPS = 5
OBS = ObservationType.KIN
ACT = ActionType.ONE_D_RPM

class MultiAgentEnvWrapper(gym.Env):
    def __init__(self, env):
        self.env = env
        self.num_drones = self.env.NUM_DRONES
        obs_space = self.env.observation_space[0]
        action_space = self.env.action_space[0]
        self.observation_space = obs_space
        self.action_space = action_space
        self.max_steps = int(self.env.EPISODE_LEN_SEC * self.env.SIM_FREQ)
        self.step_counter = 0

    def reset(self):
        obs_dict = self.env.reset()
        self.step_counter = 0
        return obs_dict[0], {}

    def step(self, action):
        actions_dict = {i: action[0] for i in range(2)}
        obs_dict, rewards, dones, infos = self.env.step(actions_dict)
        self.step_counter += 1
        terminated_env = self.step_counter >= self.max_steps
        truncated_env = False
        reward = np.mean(list(rewards.values()))
        return obs_dict[0], reward, terminated_env, truncated_env, {}

    def render(self, mode='human'):
        return self.env.render(mode)

    def close(self):
        return self.env.close()

def make_env():
    env = LeaderFollowerAviary(
        num_drones=NUM_DRONES,
        aggregate_phy_steps=AGGR_PHY_STEPS,
        obs=OBS,
        act=ACT
    )
    env = MultiAgentEnvWrapper(env)
    return env



train_envs = DummyVectorEnv([make_env for _ in range(16)])
test_envs = DummyVectorEnv([make_env for _ in range(8)])

env = make_env()
obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.shape[0]
state_shape = (obs_dim,)
action_shape = (act_dim,)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

max_action = torch.tensor(env.action_space.high, dtype=torch.float32, device=device)

net_a = Net(state_shape=state_shape, hidden_sizes=[256, 256], device=device)
actor = ActorProb(
    net_a, action_shape=action_shape, max_action=max_action, device=device
).to(device)
net_c = Net(state_shape=state_shape, hidden_sizes=[256, 256], device=device)
critic = Critic(net_c, device=device).to(device)
actor_critic = ActorCritic(actor, critic)
optim = torch.optim.Adam(actor_critic.parameters(), lr=3e-4)

def dist(*logits):
    return Independent(Normal(*logits), 1)

policy = PPOPolicy(
    actor,
    critic,
    optim,
    dist_fn=dist,
    action_space=env.action_space,
    deterministic_eval=True,
)

train_collector = Collector(policy, train_envs)
test_collector = Collector(policy, test_envs)

result = OnpolicyTrainer(
    policy=policy,
    train_collector=train_collector,
    test_collector=test_collector,
    max_epoch=80,
    step_per_epoch=10000,
    repeat_per_collect=4,
    episode_per_test=10,
    batch_size=256,
    step_per_collect=512,
    stop_fn=lambda mean_rewards: mean_rewards >= -10,
).run()

print(f'Finished training! Use {result["duration"]}')

policy.eval()
test_collector.reset()
result = test_collector.collect(n_episode=1, render=1 / 240)
print(f'Final reward: {result["rews"].mean()}, length: {result["lens"].mean()}')

train_envs.close()
test_envs.close()
