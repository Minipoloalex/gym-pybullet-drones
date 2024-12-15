import gymnasium as gym
import torch
import numpy as np
from tianshou.data import Collector
from tianshou.env import DummyVectorEnv
from tianshou.policy import RandomPolicy, PPOPolicy, MultiAgentPolicyManager
from tianshou.trainer import OnpolicyTrainer
from tianshou.utils.net.common import ActorCritic, Net
from tianshou.utils.net.continuous import ActorProb, Critic
from torch.distributions import Independent, Normal

from gym_pybullet_drones.envs.BaseAviaryTS import DroneModel
from gym_pybullet_drones.envs.multi_agent_rl.LeaderFollowerAviaryTS import LeaderFollowerAviary
from gym_pybullet_drones.envs.single_agent_rl.BaseSingleAgentAviary import ActionType, ObservationType
from gym_pybullet_drones.utils.LoggerTS import Logger
NUM_DRONES = 2
AGGR_PHY_STEPS = 5
OBS = ObservationType.KIN
ACT = ActionType.ONE_D_RPM

class MultiAgentEnvWrapper(gym.Env):
    """Wrapper for the multi-agent environment to integrate it with the Gymnasium framework.

    This class converts a multi-agent environment into a single agent environment compatible with Gymnasium's
    interface by simplifying the action and observation spaces and updating the interaction flow.

    Attributes
    ----------
    env : gym.Env
        The environment to be wrapped.
    num_drones : int
        Number of drones in the environment.
    observation_space : spaces.Box
        The observation space of the environment.
    action_space : spaces.Box
        The action space of the environment.
    max_steps : int
        The maximum number of steps per episode.
    step_counter : int
        The current step in the episode.
    """

    def __init__(self, env):
        """Initializes the environment wrapper.

        Parameters
        ----------
        env : gym.Env
            The environment to be wrapped.
        """
        self.env = env        
        self.SIM_FREQ = 240
        self.AGGR_PHY_STEPS = 1
        self.agents = [0, 1]
        self.agent_idx = 0
        obs_space = self.env.observation_space[0]
        action_space = self.env.action_space[0]
        self.observation_space = obs_space
        self.action_space = action_space
        self.max_steps = int(self.env.EPISODE_LEN_SEC * self.env.SIM_FREQ)
        self.step_counter = 0

    def reset(self):
        """Resets the environment.

        This method resets the environment and prepares it for a new episode.
        
        Returns
        -------
        tuple
            The initial observation and an empty info dictionary.
        """
        obs_dict = self.env.reset()
        self.step_counter = 0
        return obs_dict[0], {}

    def step(self, action):
        """Steps the environment with the provided action.

        This method applies the given action to all drones in the environment, 
        collects the resulting observations, rewards, done flags, and other info.
        
        Parameters
        ----------
        action : np.ndarray
            The action for the environment, with shape (num_drones, action_dim).

        Returns
        -------
        tuple
            A tuple containing the observations, reward, termination flag, 
            truncation flag, and an empty info dictionary.
        """
        actions_dict = {i: action[0] for i in range(2)}
        obs_dict, rewards, dones, infos = self.env.step(actions_dict)
        self.step_counter += 1
        terminated_env = self.step_counter >= self.max_steps
        truncated_env = False
        reward = np.mean(list(rewards.values()))
        for j in range(NUM_DRONES):
                logger.log(drone=j,
                           timestamp=i/env.SIM_FREQ,
                          state= np.hstack([obs_dict[j][0:3], np.zeros(4), obs_dict[j][3:15], np.resize(actions_dict[j], (4))]),
                          control=np.zeros(12),
                          reward=reward
                          )
        return obs_dict[0], reward, terminated_env, truncated_env, {}

    def render(self, mode='human'):
        """Renders the environment.

        This method renders the environment using the specified mode (default is 'human').
        
        Parameters
        ----------
        mode : str, optional
            The mode in which to render (default is 'human').

        Returns
        -------
        object
            The rendered output, or None if no rendering is performed.
        """
        return self.env.render(mode)

    def close(self):
        """Closes the environment.

        This method closes the environment and frees up any resources being used.
        """
        return self.env.close()

def make_env():
    """Creates a multi-agent environment for training.

    This function sets up the environment for the LeaderFollowerAviary with two drones.
    
    Returns
    -------
    MultiAgentEnvWrapper
        A wrapped version of the LeaderFollowerAviary environment.
    """
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
logger = Logger(logging_freq_hz=int(env.SIM_FREQ/env.AGGR_PHY_STEPS),
                    num_drones=NUM_DRONES
                    )

obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.shape[0]
state_shape = (obs_dim,)
action_shape = (act_dim,)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the maximum action value (for normalization)
max_action = torch.tensor(env.action_space.high, dtype=torch.float32, device=device)

# Define neural network architectures for the actor and critic
net_a = Net(state_shape=state_shape, hidden_sizes=[256, 256], device=device)
actor = ActorProb(
    net_a, action_shape=action_shape, max_action=max_action, device=device
).to(device)
net_c = Net(state_shape=state_shape, hidden_sizes=[256, 256], device=device)
critic = Critic(net_c, device=device).to(device)
actor_critic = ActorCritic(actor, critic)

# Define the optimizer for training the actor-critic model
optim = torch.optim.Adam(actor_critic.parameters(), lr=3e-4)

def stop_fn(mean_rewards: float) -> bool:
        if env.spec.reward_threshold:
            return mean_rewards >= env.spec.reward_threshold
        return False

def dist(*logits):
    """Creates a distribution for continuous actions.

    This function creates a distribution from the output of the actor network.

    Parameters
    ----------
    logits : tuple
        The logits (mean, std) from the actor network.

    Returns
    -------
    torch.distributions.Independent
        An independent distribution object for continuous actions.
    """
    return Independent(Normal(*logits), 1)

# Define policies for each drone
policies = {}
for i in range(NUM_DRONES):
    policies[i] = PPOPolicy(
        actor=actor,
        critic=critic,
        optim=optim,
        dist_fn=dist,
        action_space=train_envs.action_space,
        deterministic_eval=True
    )

# Collectors for training and testing
train_collector = Collector(policies[0], train_envs)
test_collector = Collector(policies[1], test_envs)

# Training the model using the OnpolicyTrainer
result = OnpolicyTrainer(
    policy=policies[0],
    train_collector=train_collector,
    test_collector=test_collector,
    max_epoch=10,
    step_per_epoch=20000,
    repeat_per_collect=4,
    episode_per_test=16,
    batch_size=256,
    step_per_collect=16,
).run()

print(f'Finished training! Use {result["duration"]}')

# Evaluate the trained policy
test_collector.reset()
result = test_collector.collect(n_episode=1, render=1 / 240)
print(f'Final reward: {result["rews"].mean()}, length: {result["lens"].mean()}')
# logger.save_as_csv("ma") # Optional CSV save
logger.plot()
# Close the environments
train_envs.close()
test_envs.close()
