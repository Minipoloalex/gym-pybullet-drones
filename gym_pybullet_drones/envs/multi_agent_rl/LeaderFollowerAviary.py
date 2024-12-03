import math
import numpy as np
from gymnasium import spaces
from ray.rllib.env.multi_agent_env import MultiAgentEnv, ENV_STATE

from gym_pybullet_drones.envs.BaseAviary import DroneModel, Physics, BaseAviary

from gym_pybullet_drones.envs.BaseAviary import DroneModel, Physics, BaseAviary
from gym_pybullet_drones.envs.single_agent_rl.BaseSingleAgentAviary import ActionType, ObservationType
from gym_pybullet_drones.envs.multi_agent_rl.BaseMultiagentAviary import BaseMultiagentAviary

class LeaderFollowerAviary(BaseMultiagentAviary):
    """
    Multi-agent Reinforcement Learning (RL) environment for the leader-follower problem.
    This environment models a scenario where one drone acts as the leader and others follow.
    
    The drones are rewarded based on their position relative to each other, and their task is to maintain
    specific positions relative to the leader drone.
    """

    def __init__(self,
                 drone_model: DroneModel=DroneModel.CF2X,
                 num_drones: int=2,
                 neighbourhood_radius: float=np.inf,
                 initial_xyzs=None,
                 initial_rpys=None,
                 physics: Physics=Physics.PYB,
                 freq: int=240,
                 aggregate_phy_steps: int=1,
                 gui=False,
                 record=False, 
                 obs: ObservationType=ObservationType.KIN,
                 act: ActionType=ActionType.RPM):
        """Initialization of a multi-agent RL environment.

            Inherits from `BaseMultiagentAviary`, which is a superclass for multi-agent RL problems in gym.

            Attributes:
            -----------
            drone_model : DroneModel
                The type of drone to be used in the environment. Default is `DroneModel.CF2X` (detailed in an .urdf file in folder `assets`).
            num_drones : int
                The number of drones in the aviary. Default is 2.
            neighbourhood_radius : float
                The radius used for computing adjacency matrix, in meters, for the drones. Default is infinity.
            initial_xyzs : ndarray or None
                Initial positions of drones in 3D space. If None, the positions are initialized randomly.
            initial_rpys : ndarray or None
                Initial roll, pitch, yaw orientations of drones. If None, the orientations are initialized randomly.
            physics : Physics
                The physics engine used for simulating drone dynamics. Default is `Physics.PYB` (PyBullet).
            freq : int
                Frequency at which the physics engine steps. Default is 240Hz.
            aggregate_phy_steps : int
                The number of physics steps per `step()` call.
            gui : bool
                Whether to display the graphical user interface (GUI) for PyBullet. Default is `False`.
            record : bool
                Whether to record a video of the simulation. Default is `False`.
            obs : ObservationType
                The type of observation space (kinematic or vision-based). Default is `ObservationType.KIN`.
            act : ActionType
                The type of action space (RPM, thrust/torques, or waypoints). Default is `ActionType.RPM`.

        """
        super().__init__(drone_model=drone_model,
                         num_drones=num_drones,
                         neighbourhood_radius=neighbourhood_radius,
                         initial_xyzs=initial_xyzs,
                         initial_rpys=initial_rpys,
                         physics=physics,
                         freq=freq,
                         aggregate_phy_steps=aggregate_phy_steps,
                         gui=gui,
                         record=record, 
                         obs=obs,
                         act=act
                         )

    ################################################################################
    
    def _computeReward(self):
        """
            Computes the reward for each drone in the environment.

            The reward is based on the distance of each drone from its target position:
            - The leader drone is rewarded based on how close it is to a target position.
            - The follower drones are rewarded based on how close they are to their respective target positions.

            Returns:
            --------
            dict[int, float]
                A dictionary where keys are drone indices and values are the computed rewards.
        """
        rewards = {}
        states = np.array([self._getDroneStateVector(i) for i in range(self.NUM_DRONES)])

        # Reward for the leader drone
        rewards[0] = -1 * np.linalg.norm(np.array([0, 0, 0.5]) - states[0, 0:3])**2

        # rewards[1] = -1 * np.linalg.norm(np.array([states[1, 0], states[1, 1], 0.5]) - states[1, 0:3])**2 # DEBUG WITH INDEPENDENT REWARD 
        
        # Reward for follower drones
        for i in range(1, self.NUM_DRONES):
            rewards[i] = -(1/self.NUM_DRONES) * np.linalg.norm(np.array([states[i, 0], states[i, 1], states[0, 2]]) - states[i, 0:3])**2
        return rewards

    ################################################################################
    
    def _computeDone(self):
        """
            Determines if the episode is done based on the simulation time.

            An episode ends when the simulation time exceeds the predefined episode length.

            Returns:
            --------
            dict[int | "__all__", bool]
                A dictionary where each key corresponds to a drone and indicates whether the episode is done for that drone.
                Additionally, a key `"__all__"` is included to indicate if the entire episode is done.
        """

        bool_val = True if self.step_counter/self.SIM_FREQ > self.EPISODE_LEN_SEC else False
        done = {i: bool_val for i in range(self.NUM_DRONES)}
        done["__all__"] = bool_val # True if True in done.values() else False
        return done

    ################################################################################
    
    def _computeInfo(self):
        """
            Returns additional information about the current state of the environment.
            
            Currently not used, but a placeholder for future implementation.

            Returns:
            --------
            dict[int, dict]
                A dictionary where keys are drone indices and the values are empty dictionaries (unused).
        """
        return {i: {} for i in range(self.NUM_DRONES)}

    ################################################################################

    def _clipAndNormalizeState(self, state):
        """
            Normalizes and clips the state of a drone to ensure it falls within predefined limits.
            
            The normalization ensures that the state values are mapped to the range [-1, 1].

            Parameters:
            -----------
            state : ndarray
                A 20-dimensional array representing the state of a single drone.
            
            Returns:
            --------
            ndarray
                A 20-dimensional array containing the clipped and normalized state of the drone.
        """
        MAX_LIN_VEL_XY = 3 
        MAX_LIN_VEL_Z = 1

        MAX_XY = MAX_LIN_VEL_XY*self.EPISODE_LEN_SEC
        MAX_Z = MAX_LIN_VEL_Z*self.EPISODE_LEN_SEC

        MAX_PITCH_ROLL = np.pi # Full range

        clipped_pos_xy = np.clip(state[0:2], -MAX_XY, MAX_XY)
        clipped_pos_z = np.clip(state[2], 0, MAX_Z)
        clipped_rp = np.clip(state[7:9], -MAX_PITCH_ROLL, MAX_PITCH_ROLL)
        clipped_vel_xy = np.clip(state[10:12], -MAX_LIN_VEL_XY, MAX_LIN_VEL_XY)
        clipped_vel_z = np.clip(state[12], -MAX_LIN_VEL_Z, MAX_LIN_VEL_Z)

        if self.GUI:
            self._clipAndNormalizeStateWarning(state,
                                               clipped_pos_xy,
                                               clipped_pos_z,
                                               clipped_rp,
                                               clipped_vel_xy,
                                               clipped_vel_z
                                               )

        normalized_pos_xy = clipped_pos_xy / MAX_XY
        normalized_pos_z = clipped_pos_z / MAX_Z
        normalized_rp = clipped_rp / MAX_PITCH_ROLL
        normalized_y = state[9] / np.pi # No reason to clip
        normalized_vel_xy = clipped_vel_xy / MAX_LIN_VEL_XY
        normalized_vel_z = clipped_vel_z / MAX_LIN_VEL_XY
        normalized_ang_vel = state[13:16]/np.linalg.norm(state[13:16]) if np.linalg.norm(state[13:16]) != 0 else state[13:16]

        norm_and_clipped = np.hstack([normalized_pos_xy,
                                      normalized_pos_z,
                                      state[3:7],
                                      normalized_rp,
                                      normalized_y,
                                      normalized_vel_xy,
                                      normalized_vel_z,
                                      normalized_ang_vel,
                                      state[16:20]
                                      ]).reshape(20,)

        return norm_and_clipped
    
    ################################################################################
    
    def _clipAndNormalizeStateWarning(self,
                                      state,
                                      clipped_pos_xy,
                                      clipped_pos_z,
                                      clipped_rp,
                                      clipped_vel_xy,
                                      clipped_vel_z,
                                      ):
        """
            Prints warnings if the state values are outside the expected clipping range during debugging.

            This method is invoked if a value exceeds the set clipping boundaries, providing feedback for debugging.

            Parameters:
            -----------
            state : ndarray
                The original state of the drone.
            clipped_pos_xy : ndarray
                The clipped values for the drone's position in the XY plane.
            clipped_pos_z : ndarray
                The clipped value for the drone's Z position.
            clipped_rp : ndarray
                The clipped values for the drone's roll and pitch.
            clipped_vel_xy : ndarray
                The clipped values for the drone's velocity in the XY plane.
            clipped_vel_z : ndarray
                The clipped value for the drone's Z velocity.
        """
        if not(clipped_pos_xy == np.array(state[0:2])).all():
            print("[WARNING] it", self.step_counter, "in LeaderFollowerAviary._clipAndNormalizeState(), clipped xy position [{:.2f} {:.2f}]".format(state[0], state[1]))
        if not(clipped_pos_z == np.array(state[2])).all():
            print("[WARNING] it", self.step_counter, "in LeaderFollowerAviary._clipAndNormalizeState(), clipped z position [{:.2f}]".format(state[2]))
        if not(clipped_rp == np.array(state[7:9])).all():
            print("[WARNING] it", self.step_counter, "in LeaderFollowerAviary._clipAndNormalizeState(), clipped roll/pitch [{:.2f} {:.2f}]".format(state[7], state[8]))
        if not(clipped_vel_xy == np.array(state[10:12])).all():
            print("[WARNING] it", self.step_counter, "in LeaderFollowerAviary._clipAndNormalizeState(), clipped xy velocity [{:.2f} {:.2f}]".format(state[10], state[11]))
        if not(clipped_vel_z == np.array(state[12])).all():
            print("[WARNING] it", self.step_counter, "in LeaderFollowerAviary._clipAndNormalizeState(), clipped z velocity [{:.2f}]".format(state[12]))
