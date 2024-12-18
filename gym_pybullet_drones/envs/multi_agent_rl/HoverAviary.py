import math
import numpy as np
from gym import spaces
from ray.rllib.env.multi_agent_env import MultiAgentEnv, ENV_STATE

from gym_pybullet_drones.envs.BaseAviary import DroneModel, Physics, BaseAviary

from gym_pybullet_drones.envs.BaseAviary import DroneModel, Physics, BaseAviary
from gym_pybullet_drones.envs.single_agent_rl.BaseSingleAgentAviary import ActionType, ObservationType
from gym_pybullet_drones.envs.multi_agent_rl.BaseMultiagentAviary import BaseMultiagentAviary

class HoverAviary(BaseMultiagentAviary):
    """Multi-agent RL problem: Hover at a given position."""

    ################################################################################

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
                 act: ActionType=ActionType.RPM,
                 is_test_env: bool = False):
        """Initialization of a multi-agent RL environment.

        Using the generic multi-agent RL superclass.

        Parameters
        ----------
        drone_model : DroneModel, optional
            The desired drone type (detailed in an .urdf file in folder `assets`).
        num_drones : int, optional
            The desired number of drones in the aviary.
        neighbourhood_radius : float, optional
            Radius used to compute the drones' adjacency matrix, in meters.
        initial_xyzs: ndarray | None, optional
            (NUM_DRONES, 3)-shaped array containing the initial XYZ position of the drones.
        initial_rpys: ndarray | None, optional
            (NUM_DRONES, 3)-shaped array containing the initial orientations of the drones (in radians).
        physics : Physics, optional
            The desired implementation of PyBullet physics/custom dynamics.
        freq : int, optional
            The frequency (Hz) at which the physics engine steps.
        aggregate_phy_steps : int, optional
            The number of physics steps within one call to `BaseAviary.step()`.
        gui : bool, optional
            Whether to use PyBullet's GUI.
        record : bool, optional
            Whether to save a video of the simulation in folder `files/videos/`.
        obs : ObservationType, optional
            The type of observation space (kinematic information or vision)
        act : ActionType, optional
            The type of action space (1 or 3D; RPMS, thurst and torques, or waypoint with PID control)

        """
        initial_xyzs = np.array([(0.5, 0.5, 0.5), (-0.5, -0.5, 0.5)])
        self.TARGETS = [(0.5, 0.5, 0.5), (-0.5, -0.5, 0.5)]
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
        
        self.OWN_OBS_VEC_SIZE = 19

        self.is_test_env = is_test_env

    ################################################################################

    def _computeReward(self):
        """Computes the current reward value(s).

        Returns
        -------
        dict[int, float]
            The reward value for each drone.
        """

        rewards = {}
        for i in range(self.NUM_DRONES):
            state = self._getDroneStateVector(i)
            distance = calculate_distance_sq(state[0:3], self.TARGETS[i])
            rewards[i] = -distance
            z = state[2]
            zt = self.TARGETS[i][2]
            zdist_sq = (zt - z) * (zt - z)
            if z <= zt:
                rewards[i] -= zdist_sq * 10
            else:
                rewards[i] -= zdist_sq * 2
            # if z <= 0.1:
            #     rewards[i] += -10
            # if z >= 1:
            #     rewards[i] += -10

        return rewards

    ################################################################################
    
    def _computeDone(self):
        """Computes the current done value(s).

        Returns
        -------
        dict[int | "__all__", bool]
            Dictionary with the done value of each drone and 
            one additional boolean value for key "__all__".

        """
        bool_val = True if self.step_counter/self.SIM_FREQ > self.EPISODE_LEN_SEC else False
        done = {i: bool_val for i in range(self.NUM_DRONES)}
        done["__all__"] = bool_val # True if True in done.values() else False
        return done

    ################################################################################
    
    def _computeInfo(self):
        """Computes the current info dict(s).

        Unused.

        Returns
        -------
        dict[int, dict[]]
            Dictionary of empty dictionaries.

        """
        info = lambda i: self._getDroneStateVector(i) if self.is_test_env else {}
        return {i: info(i) for i in range(self.NUM_DRONES)}

    ################################################################################

    def _clipAndNormalizeState(self,
                               state
                               ):
        """Normalizes a drone's state to the [-1,1] range.

        Parameters
        ----------
        state : ndarray
            (20,)-shaped array of floats containing the non-normalized state of a single drone.

        Returns
        -------
        ndarray
            (20,)-shaped array of floats containing the normalized state of a single drone.

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

    def _clipAndNormalizeTarget(self, target):
        """Normalizes a drone's target to the [-1,1] range."""
        MAX_LIN_VEL_XY = 3 
        MAX_LIN_VEL_Z = 1

        MAX_XY = MAX_LIN_VEL_XY*self.EPISODE_LEN_SEC
        MAX_Z = MAX_LIN_VEL_Z*self.EPISODE_LEN_SEC

        clipped_target_xy = np.clip(target[0:2], -MAX_XY, MAX_XY)
        clipped_target_z = np.clip(target[2], 0, MAX_Z)

        if self.GUI:
            if not(clipped_target_xy == np.array(target[0:2])).all():
                print("[WARNING] init in FigureAviary._clipAndNormalizeTarget(), clipped xy target position [{:.2f} {:.2f}]".format(target[0], target[1]))
            if not(clipped_target_z == np.array(target[2])).all():
                print("[WARNING] init in FigureAviary._clipAndNormalizeTarget(), clipped z target position [{:.2f}]".format(target[2]))

        normalized_target_xy = clipped_target_xy / MAX_XY
        normalized_target_z = clipped_target_z / MAX_Z

        norm_and_clipped = np.hstack([normalized_target_xy, normalized_target_z]).reshape(3,)
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
        """Debugging printouts associated to `_clipAndNormalizeState`.

        Print a warning if values in a state vector is out of the clipping range.
        
        """
        if not(clipped_pos_xy == np.array(state[0:2])).all():
            print("[WARNING] it", self.step_counter, "in FigureAviary._clipAndNormalizeState(), clipped xy position [{:.2f} {:.2f}]".format(state[0], state[1]))
        if not(clipped_pos_z == np.array(state[2])).all():
            print("[WARNING] it", self.step_counter, "in FigureAviary._clipAndNormalizeState(), clipped z position [{:.2f}]".format(state[2]))
        if not(clipped_rp == np.array(state[7:9])).all():
            print("[WARNING] it", self.step_counter, "in FigureAviary._clipAndNormalizeState(), clipped roll/pitch [{:.2f} {:.2f}]".format(state[7], state[8]))
        if not(clipped_vel_xy == np.array(state[10:12])).all():
            print("[WARNING] it", self.step_counter, "in FigureAviary._clipAndNormalizeState(), clipped xy velocity [{:.2f} {:.2f}]".format(state[10], state[11]))
        if not(clipped_vel_z == np.array(state[12])).all():
            print("[WARNING] it", self.step_counter, "in FigureAviary._clipAndNormalizeState(), clipped z velocity [{:.2f}]".format(state[12]))

    ################################################################################

    def _observationSpace(self):
        """Returns the observation space of the environment.

        Returns
        -------
        dict[int, ndarray]
            A Dict with NUM_DRONES entries indexed by Id in integer format,
            each a Box() of shape (15 + 4*(NUM_DRONES - 1),).

        """
        if self.OBS_TYPE == ObservationType.RGB:
            print("[ERROR] in FigureAviary._observationSpace()")
            raise NotImplementedError()
        elif self.OBS_TYPE == ObservationType.KIN:
            ############################################################
            xy_boundary  = [-1, 1]
            z_boundary   = [ 0, 1]
            v_boundary   = [-1, 1]
            rpy_boundary = [-1, 1]
            quat_boundary= [-1, 1]
            w_boundary   = [-1, 1]
            d_boundary   = [ 0, 1]
            u_boundary   = [-1, 1]
            
            low_boundaries = []
            high_boundaries = []
            for i, boundaries in enumerate([low_boundaries, high_boundaries]):
                boundaries += [xy_boundary [i], xy_boundary [i], z_boundary  [i],
                               v_boundary  [i], v_boundary  [i], v_boundary  [i],
                               quat_boundary[i], quat_boundary[i], quat_boundary[i], quat_boundary[i],
                               rpy_boundary[i], rpy_boundary[i], rpy_boundary[i],
                               w_boundary  [i], w_boundary  [i], w_boundary  [i],
                               u_boundary  [i], u_boundary  [i], u_boundary  [i]]

            return spaces.Dict({ i: spaces.Box(
                low=np.array(low_boundaries),
                high=np.array(high_boundaries),
                dtype=np.float32) for i in range(self.NUM_DRONES) })
            ############################################################
        else:
            print("[ERROR] in FigureAviary._observationSpace()")

    ################################################################################

    def _computeObs(self):
        """Returns the current observation of the environment.

        Returns
        -------
        dict[int, ndarray]
            A Dict with NUM_DRONES entries indexed by Id in integer format

        """
        if self.OBS_TYPE == ObservationType.RGB:
            print("[ERROR] in FigureAviary._computeObs()")
            raise NotImplementedError()

        elif self.OBS_TYPE == ObservationType.KIN:
            ############################################################
            obs = np.zeros((self.NUM_DRONES, self.OWN_OBS_VEC_SIZE))
            states = np.array([self._clipAndNormalizeState(self._getDroneStateVector(i)) for i in range(self.NUM_DRONES)])

            for i in range(self.NUM_DRONES):
                obs[i, :] = np.hstack([states[i,  0: 3], # x y z
                                       states[i, 10:13], # vx vy vz
                                       states[i, 3:7]  , # quaternions
                                       states[i,  7:10], # r p y
                                       states[i, 13:16], # wx wy wz
                                       self.TARGETS[i] , # tx ty tz
                                      ])
            
            return {i: obs[i, :] for i in range(self.NUM_DRONES)}
            ############################################################
        else:
            print("[ERROR] in FigureAviary._computeObs()")

    ################################################################################

def calculate_distance_sq(pos1, pos2):
    return (pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2 + (pos1[2] - pos2[2])**2
