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

# Importar os ambientes do gym_pybullet_drones
from gym_pybullet_drones.envs.multi_agent_rl.FlockAviary import FlockAviary
from gym_pybullet_drones.envs.multi_agent_rl.LeaderFollowerAviary import LeaderFollowerAviary
from gym_pybullet_drones.envs.multi_agent_rl.MeetupAviary import MeetupAviary
from gym_pybullet_drones.envs.single_agent_rl.BaseSingleAgentAviary import ActionType, ObservationType
from gym_pybullet_drones.utils.utils import sync
from gym.spaces import Box

# Parâmetros do ambiente
NUM_DRONES = 2  # Defina o número de drones conforme necessário
AGGR_PHY_STEPS = 5  # Este parâmetro é usado nos ambientes do gym_pybullet_drones
OBS = ObservationType.KIN  # Tipo de observação
ACT = ActionType.RPM # Tipo de ação

# Crie um wrapper para transformar o ambiente multiagente em um ambiente de agente único
class MultiAgentEnvWrapper(gym.Env):
    def __init__(self, env):
        self.env = env
        self.num_drones = self.env.NUM_DRONES
        # Combine os espaços de observação e ação
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
        obs_dict, _ = self.env.reset()
        obs_concat = np.concatenate([obs_dict[i] for i in range(self.num_drones)])
        info = {}
        return obs_concat, info

    def step(self, action):
        # Dividir a ação para cada agente
        action_dim = self.env.action_space[0].shape[0]
        actions = {}
        for i in range(self.num_drones):
            start = i * action_dim
            end = (i + 1) * action_dim
            actions[i] = action[start:end]
        # Chamar o método step do ambiente original
        obs_dict, rewards, terminated, truncated, info = self.env.step(actions)
        obs_concat = np.concatenate([obs_dict[i] for i in range(self.num_drones)])
        reward = np.mean(list(rewards.values()))
        # Retornar 'terminated' e 'truncated' como booleanos agregados
        terminated_env = any(terminated.values())
        truncated_env = any(truncated.values())
        return obs_concat, reward, terminated_env, truncated_env, info





    def render(self, mode='human'):
        return self.env.render(mode)

    def close(self):
        return self.env.close()

# Função para criar o ambiente
def make_env():
    env = LeaderFollowerAviary(
        num_drones=NUM_DRONES,
        aggregate_phy_steps=AGGR_PHY_STEPS,
        obs=OBS,
        act=ACT,
    )
    env = MultiAgentEnvWrapper(env)  # Aplica o wrapper

    return env
env = make_env()
# Criar instâncias do ambiente
train_envs = DummyVectorEnv([make_env for _ in range(8)])
test_envs = DummyVectorEnv([make_env for _ in range(8)])


state_shape = env.observation_space.shape or env.observation_space.n
action_shape = env.action_space.shape or env.action_space.n

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

# Função de distribuição
def dist(*logits):
    return Independent(Normal(*logits), 1)

# Política PPO
policy = PPOPolicy(
    actor,
    critic,
    optim,
    dist_fn=dist,
    action_space=env.action_space,
    deterministic_eval=True,
)

# Coletores
train_collector = Collector(policy, train_envs)
test_collector = Collector(policy, test_envs)

# Treinador
result = OnpolicyTrainer(
    policy=policy,
    train_collector=train_collector,
    test_collector=test_collector,
    max_epoch=100,
    step_per_epoch=10000,
    repeat_per_collect=4,
    episode_per_test=10,
    batch_size=256,
    step_per_collect=2000,
    stop_fn=lambda mean_rewards: mean_rewards >= 200,
).run()

print(f'Finished training! Use {result["duration"]}')

# Testar a política treinada
policy.eval()
test_collector.reset()
result = test_collector.collect(n_episode=1, render=1 / env.env.env.SIM_FREQ)
print(f'Final reward: {result["rews"].mean()}, length: {result["lens"].mean()}')

# Fechar os ambientes
train_envs.close()
test_envs.close()
