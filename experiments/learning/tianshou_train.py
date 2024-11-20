import os
import time
import argparse
import numpy as np
from datetime import datetime
from tianshou.env import DummyVectorEnv
from tianshou.data import Collector, VectorReplayBuffer
from tianshou.policy import PPOPolicy
from tianshou.trainer import onpolicy_trainer
from tianshou.utils.net.common import ActorCritic
from tianshou.utils.net.continuous import ActorProb, Critic
import torch
import torch.nn as nn
from torch.optim import Adam

from gym_pybullet_drones.envs.multi_agent_rl.LeaderFollowerAviary import LeaderFollowerAviary
from gym_pybullet_drones.envs.multi_agent_rl.FlockAviary import FlockAviary
from gym_pybullet_drones.envs.multi_agent_rl.MeetupAviary import MeetupAviary
from gym_pybullet_drones.envs.single_agent_rl.BaseSingleAgentAviary import ActionType, ObservationType
from gym_pybullet_drones.envs.BaseAviary import DroneModel
import shared_constants

# Configurar os argumentos
parser = argparse.ArgumentParser()
parser.add_argument('--num_drones', type=int, default=2, help='Number of drones')
parser.add_argument('--env', type=str, default='leaderfollower', choices=['leaderfollower', 'flock', 'meetup'])
parser.add_argument('--obs', type=str, default='kin', choices=['kin', 'rgb'])
parser.add_argument('--act', type=str, default='one_d_rpm', choices=['one_d_rpm', 'rpm'])
args = parser.parse_args()

# Função para criar o ambiente
def make_env():
    if args.env == 'leaderfollower':
        env = LeaderFollowerAviary(
            num_drones=args.num_drones,
            aggregate_phy_steps=shared_constants.AGGR_PHY_STEPS,
            obs=ObservationType[args.obs.upper()],
            act=ActionType[args.act.upper()],
        )
    elif args.env == 'flock':
        env = FlockAviary(
            num_drones=args.num_drones,
            aggregate_phy_steps=shared_constants.AGGR_PHY_STEPS,
            obs=ObservationType[args.obs.upper()],
            act=ActionType[args.act.upper()],
        )
    elif args.env == 'meetup':
        env = MeetupAviary(
            num_drones=args.num_drones,
            aggregate_phy_steps=shared_constants.AGGR_PHY_STEPS,
            obs=ObservationType[args.obs.upper()],
            act=ActionType[args.act.upper()],
        )
    else:
        raise ValueError("Unknown environment type")
    return env

# Criar os ambientes de treino e teste
train_envs = DummyVectorEnv([make_env for _ in range(1)])  # 8 instâncias do ambiente
test_envs = DummyVectorEnv([make_env for _ in range(1)])   # 4 instâncias do ambiente

# Dimensões de observação e ação
dummy_env = make_env()  # Testar o ambiente com uma instância
state_shape = dummy_env.observation_space[0].shape
action_shape = dummy_env.action_space[0].shape[0]
max_action = dummy_env.action_space[0].high[0]

# Redes neurais para o ator e o crítico
class Net(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_sizes):
        super().__init__()
        layers = []
        last_size = input_dim
        for size in hidden_sizes:
            layers.append(nn.Linear(last_size, size))
            layers.append(nn.ReLU())
            last_size = size
        layers.append(nn.Linear(last_size, output_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

actor = ActorProb(
    Net(state_shape[0], 64, [128, 128]), action_shape, max_action=max_action
).to("cpu")
critic = Critic(
    Net(state_shape[0], 1, [128, 128])
).to("cpu")
actor_critic = ActorCritic(actor, critic)
optim = Adam(actor_critic.parameters(), lr=3e-4)

# Política
policy = PPOPolicy(
    actor,
    critic,
    optim,
    dist_fn=torch.distributions.TanhNormal,
    discount_factor=0.99,
    gae_lambda=0.95,
    max_grad_norm=0.5,
    vf_coef=0.5,
    ent_coef=0.01,
    reward_normalization=True,
    action_scaling=True,
    action_bound_method="clip",
)

# Replay buffer e coleta de dados
train_collector = Collector(policy, train_envs, VectorReplayBuffer(20000, len(train_envs)))
test_collector = Collector(policy, test_envs)

# Treinamento
result = onpolicy_trainer(
    policy,
    train_collector,
    test_collector,
    max_epoch=10,
    step_per_epoch=1000,
    repeat_per_collect=2,
    episode_per_test=5,
    batch_size=64,
)

# Salvar o modelo
torch.save(policy.state_dict(), "ppo_policy.pth")