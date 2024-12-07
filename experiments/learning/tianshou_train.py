
import gym
import gym.spaces
import numpy as np

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

if __name__ == "__main__":
    env = WrapperMA(num_drones = 5)
    obs = env.reset()
    print()
    print(obs)
    print()
    obs, rewards, done, info = env.step(env.action_space.sample())
    print()
    print(obs)
    print()
    print(rewards)
    print()
    print(done)
    print()
    print(info)
    print()
