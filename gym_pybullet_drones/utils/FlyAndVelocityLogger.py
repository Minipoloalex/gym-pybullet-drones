import os
import time
from datetime import datetime
from cycler import cycler
import numpy as np
import matplotlib.pyplot as plt

os.environ['KMP_DUPLICATE_LIB_OK']='True'

class Logger(object):
    """A class for logging and visualization.

    Stores, saves to file, and plots the kinematic information and RPMs
    of a simulation with one or more drones.

    """

    ################################################################################

    def __init__(self,
                 logging_freq_hz: int,
                 num_drones: int=1,
                 duration_sec: int=0
                 ):
        """Logger class __init__ method.

        Parameters
        ----------
        logging_freq_hz : int
            Logging frequency in Hz.
        num_drones : int, optional
            Number of drones.
        duration_sec : int, optional
            Used to preallocate the log arrays (improves performance).

        """
        self.LOGGING_FREQ_HZ = logging_freq_hz
        self.NUM_DRONES = num_drones
        self.PREALLOCATED_ARRAYS = False if duration_sec == 0 else True
        self.counters = np.zeros(num_drones)
        self.timestamps = np.zeros((num_drones, duration_sec*self.LOGGING_FREQ_HZ))
        self.states = np.zeros((num_drones, 16, duration_sec*self.LOGGING_FREQ_HZ)) #### 16 states: pos_x,
                                                                                                  # pos_y,
                                                                                                  # pos_z,
                                                                                                  # vel_x,
                                                                                                  # vel_y,
                                                                                                  # vel_z,
                                                                                                  # roll,
                                                                                                  # pitch,
                                                                                                  # yaw,
                                                                                                  # ang_vel_x,
                                                                                                  # ang_vel_y,
                                                                                                  # ang_vel_z,
                                                                                                  # rpm0,
                                                                                                  # rpm1,
                                                                                                  # rpm2,
                                                                                                  # rpm3
        #### Note: this is not the same order nor length ###########
        self.controls = np.zeros((num_drones, 12, duration_sec*self.LOGGING_FREQ_HZ)) #### 12 control targets: pos_x,
                                                                                                             # pos_y,
                                                                                                             # pos_z,
                                                                                                             # vel_x, 
                                                                                                             # vel_y,
                                                                                                             # vel_z,
                                                                                                             # roll,
                                                                                                             # pitch,
                                                                                                             # yaw,
                                                                                                             # ang_vel_x,
                                                                                                             # ang_vel_y,
                                                                                                             # ang_vel_z

    ################################################################################

    def log(self,
            drone: int,
            timestamp,
            state,
            control=np.zeros(12)
            ):
        """Logs entries for a single simulation step, of a single drone.

        Parameters
        ----------
        drone : int
            Id of the drone associated to the log entry.
        timestamp : float
            Timestamp of the log in simulation clock.
        state : ndarray
            (20,)-shaped array of floats containing the drone's state.
        control : ndarray, optional
            (12,)-shaped array of floats containing the drone's control target.

        """
        if drone < 0 or drone >= self.NUM_DRONES or timestamp < 0 or len(state) != 20 or len(control) != 12:
            print("[ERROR] in Logger.log(), invalid data")
        current_counter = int(self.counters[drone])
        #### Add rows to the matrices if a counter exceeds their size
        if current_counter >= self.timestamps.shape[1]:
            self.timestamps = np.concatenate((self.timestamps, np.zeros((self.NUM_DRONES, 1))), axis=1)
            self.states = np.concatenate((self.states, np.zeros((self.NUM_DRONES, 16, 1))), axis=2)
            self.controls = np.concatenate((self.controls, np.zeros((self.NUM_DRONES, 12, 1))), axis=2)
        #### Advance a counter is the matrices have overgrown it ###
        elif not self.PREALLOCATED_ARRAYS and self.timestamps.shape[1] > current_counter:
            current_counter = self.timestamps.shape[1]-1
        #### Log the information and increase the counter ##########
        self.timestamps[drone, current_counter] = timestamp
        self.states[drone, :, current_counter] = np.hstack([state[0:3], state[10:13], state[7:10], state[13:20]])
        self.controls[drone, :, current_counter] = control
        self.counters[drone] = current_counter + 1

    ################################################################################

    def save(self):
        """Save the logs to file.
        """
        with open(os.path.dirname(os.path.abspath(__file__))+"/../../files/logs/save-flight-"+datetime.now().strftime("%m.%d.%Y_%H.%M.%S")+".npy", 'wb') as out_file:
            np.save(out_file, self.timestamps)
            np.save(out_file, self.states)
            np.save(out_file, self.controls)

    ################################################################################

    def save_as_csv(self,
                    comment: str=""
                    ):
        """Save the logs---on your Desktop---as comma separated values.

        Parameters
        ----------
        comment : str, optional
            Added to the foldername.

        """
        csv_dir = os.environ.get('HOME')+"/Desktop/save-flight-"+comment+"-"+datetime.now().strftime("%m.%d.%Y_%H.%M.%S")
        if not os.path.exists(csv_dir):
            os.makedirs(csv_dir+'/')
        t = np.arange(0, self.timestamps.shape[1]/self.LOGGING_FREQ_HZ, 1/self.LOGGING_FREQ_HZ)
        for i in range(self.NUM_DRONES):
            with open(csv_dir+"/x"+str(i)+".csv", 'wb') as out_file:
                np.savetxt(out_file, np.transpose(np.vstack([t, self.states[i, 0, :]])), delimiter=",")
            with open(csv_dir+"/y"+str(i)+".csv", 'wb') as out_file:
                np.savetxt(out_file, np.transpose(np.vstack([t, self.states[i, 1, :]])), delimiter=",")
            with open(csv_dir+"/z"+str(i)+".csv", 'wb') as out_file:
                np.savetxt(out_file, np.transpose(np.vstack([t, self.states[i, 2, :]])), delimiter=",")
            ####
            with open(csv_dir+"/r"+str(i)+".csv", 'wb') as out_file:
                np.savetxt(out_file, np.transpose(np.vstack([t, self.states[i, 6, :]])), delimiter=",")
            with open(csv_dir+"/p"+str(i)+".csv", 'wb') as out_file:
                np.savetxt(out_file, np.transpose(np.vstack([t, self.states[i, 7, :]])), delimiter=",")
            with open(csv_dir+"/ya"+str(i)+".csv", 'wb') as out_file:
                np.savetxt(out_file, np.transpose(np.vstack([t, self.states[i, 8, :]])), delimiter=",")
            ####
            with open(csv_dir+"/rr"+str(i)+".csv", 'wb') as out_file:
                rdot = np.hstack([0, (self.states[i, 6, 1:] - self.states[i, 6, 0:-1]) * self.LOGGING_FREQ_HZ ])
                np.savetxt(out_file, np.transpose(np.vstack([t, rdot])), delimiter=",")
            with open(csv_dir+"/pr"+str(i)+".csv", 'wb') as out_file:
                pdot = np.hstack([0, (self.states[i, 7, 1:] - self.states[i, 7, 0:-1]) * self.LOGGING_FREQ_HZ ])
                np.savetxt(out_file, np.transpose(np.vstack([t, pdot])), delimiter=",")
            with open(csv_dir+"/yar"+str(i)+".csv", 'wb') as out_file:
                ydot = np.hstack([0, (self.states[i, 8, 1:] - self.states[i, 8, 0:-1]) * self.LOGGING_FREQ_HZ ])
                np.savetxt(out_file, np.transpose(np.vstack([t, ydot])), delimiter=",")
            ###
            with open(csv_dir+"/vx"+str(i)+".csv", 'wb') as out_file:
                np.savetxt(out_file, np.transpose(np.vstack([t, self.states[i, 3, :]])), delimiter=",")
            with open(csv_dir+"/vy"+str(i)+".csv", 'wb') as out_file:
                np.savetxt(out_file, np.transpose(np.vstack([t, self.states[i, 4, :]])), delimiter=",")
            with open(csv_dir+"/vz"+str(i)+".csv", 'wb') as out_file:
                np.savetxt(out_file, np.transpose(np.vstack([t, self.states[i, 5, :]])), delimiter=",")
            ####
            with open(csv_dir+"/wx"+str(i)+".csv", 'wb') as out_file:
                np.savetxt(out_file, np.transpose(np.vstack([t, self.states[i, 9, :]])), delimiter=",")
            with open(csv_dir+"/wy"+str(i)+".csv", 'wb') as out_file:
                np.savetxt(out_file, np.transpose(np.vstack([t, self.states[i, 10, :]])), delimiter=",")
            with open(csv_dir+"/wz"+str(i)+".csv", 'wb') as out_file:
                np.savetxt(out_file, np.transpose(np.vstack([t, self.states[i, 11, :]])), delimiter=",")
            ####
            with open(csv_dir+"/rpm0-"+str(i)+".csv", 'wb') as out_file:
                np.savetxt(out_file, np.transpose(np.vstack([t, self.states[i, 12, :]])), delimiter=",")
            with open(csv_dir+"/rpm1-"+str(i)+".csv", 'wb') as out_file:
                np.savetxt(out_file, np.transpose(np.vstack([t, self.states[i, 13, :]])), delimiter=",")
            with open(csv_dir+"/rpm2-"+str(i)+".csv", 'wb') as out_file:
                np.savetxt(out_file, np.transpose(np.vstack([t, self.states[i, 14, :]])), delimiter=",")
            with open(csv_dir+"/rpm3-"+str(i)+".csv", 'wb') as out_file:
                np.savetxt(out_file, np.transpose(np.vstack([t, self.states[i, 15, :]])), delimiter=",")
            ####
            with open(csv_dir+"/pwm0-"+str(i)+".csv", 'wb') as out_file:
                np.savetxt(out_file, np.transpose(np.vstack([t, (self.states[i, 12, :] - 4070.3) / 0.2685])), delimiter=",")
            with open(csv_dir+"/pwm1-"+str(i)+".csv", 'wb') as out_file:
                np.savetxt(out_file, np.transpose(np.vstack([t, (self.states[i, 13, :] - 4070.3) / 0.2685])), delimiter=",")
            with open(csv_dir+"/pwm2-"+str(i)+".csv", 'wb') as out_file:
                np.savetxt(out_file, np.transpose(np.vstack([t, (self.states[i, 14, :] - 4070.3) / 0.2685])), delimiter=",")
            with open(csv_dir+"/pwm3-"+str(i)+".csv", 'wb') as out_file:
                np.savetxt(out_file, np.transpose(np.vstack([t, (self.states[i, 15, :] - 4070.3) / 0.2685])), delimiter=",")

    ################################################################################
    
    def plot(self, pwm=False):
        """Logs entries for a single simulation step, of a single drone.

        Parameters
        ----------
        pwm : bool, optional
            If True, converts logged RPM into PWM values (for Crazyflies).

        """
        rows = 8
        cols = 2
        #### Loop over colors and line styles ######################
        plt.rc('axes', prop_cycle=(cycler('color', ['red', 'blue', 'green', 'gray']) + cycler('linestyle', ['-', '--', ':', '-.'])))
        fig, axs = plt.subplots(rows, cols)
        t = np.arange(0, self.timestamps.shape[1]/self.LOGGING_FREQ_HZ, 1/self.LOGGING_FREQ_HZ)

        #### Column ################################################
        col = 0

        #### XYZ ###################################################
        row = 0
        for j in range(self.NUM_DRONES):
            axs[row, col].plot(t, self.states[j, 0, :], label="drone_"+str(j))
        axs[row, col].set_xlabel('time (s)')
        axs[row, col].set_ylabel('x (m)')

        row = 1
        for j in range(self.NUM_DRONES):
            axs[row, col].plot(t, self.states[j, 1, :], label="drone_"+str(j))
        axs[row, col].set_xlabel('time (s)')
        axs[row, col].set_ylabel('y (m)')

        row = 2
        for j in range(self.NUM_DRONES):
            axs[row, col].plot(t, self.states[j, 2, :], label="drone_"+str(j))
        axs[row, col].set_xlabel('time (s)')
        axs[row, col].set_ylabel('z (m)')

        #### RPY ###################################################
        row = 3
        for j in range(self.NUM_DRONES):
            axs[row, col].plot(t, self.states[j, 6, :], label="drone_"+str(j))
        axs[row, col].set_xlabel('time (s)')
        axs[row, col].set_ylabel('r (rad)')
        row = 4
        for j in range(self.NUM_DRONES):
            axs[row, col].plot(t, self.states[j, 7, :], label="drone_"+str(j))
        axs[row, col].set_xlabel('time (s)')
        axs[row, col].set_ylabel('p (rad)')
        row = 5
        for j in range(self.NUM_DRONES):
            axs[row, col].plot(t, self.states[j, 8, :], label="drone_"+str(j))
        axs[row, col].set_xlabel('time (s)')
        axs[row, col].set_ylabel('j (rad)')

        #### Ang Vel ###############################################
        col = 1
        row = 3
        for j in range(self.NUM_DRONES):
            axs[row, col].plot(t, self.states[j, 9, :], label="drone_"+str(j))
        axs[row, col].set_xlabel('time (s)')
        axs[row, col].set_ylabel('wx')
        row = 4
        for j in range(self.NUM_DRONES):
            axs[row, col].plot(t, self.states[j, 10, :], label="drone_"+str(j))
        axs[row, col].set_xlabel('time (s)')
        axs[row, col].set_ylabel('wy')
        row = 5
        for j in range(self.NUM_DRONES):
            axs[row, col].plot(t, self.states[j, 11, :], label="drone_"+str(j))
        axs[row, col].set_xlabel('time (s)')
        axs[row, col].set_ylabel('wz')


        #### Column ################################################
        col = 1

        #### Velocity ##############################################
        row = 0
        for j in range(self.NUM_DRONES):
            axs[row, col].plot(t, self.states[j, 3, :], label="drone_"+str(j))
        axs[row, col].set_xlabel('time (s)')
        axs[row, col].set_ylabel('vx (m/s)')
        row = 1
        for j in range(self.NUM_DRONES):
            axs[row, col].plot(t, self.states[j, 4, :], label="drone_"+str(j))
        axs[row, col].set_xlabel('time (s)')
        axs[row, col].set_ylabel('vy (m/s)')
        row = 2
        for j in range(self.NUM_DRONES):
            axs[row, col].plot(t, self.states[j, 5, :], label="drone_"+str(j))
        axs[row, col].set_xlabel('time (s)')
        axs[row, col].set_ylabel('vz (m/s)')

        ### This IF converts RPM into PWM for all drones ###########
        #### except drone_0 (only used in examples/compare.py) #####
        for j in range(self.NUM_DRONES):
            for i in range(12,16):
                if pwm and j > 0:
                    self.states[j, i, :] = (self.states[j, i, :] - 4070.3) / 0.2685

        #### RPMs ##################################################
        col = 0
        row = 6
        for j in range(self.NUM_DRONES):
            axs[row, col].plot(t, self.states[j, 12, :], label="drone_"+str(j))
        axs[row, col].set_xlabel('time (s)')
        if pwm:
            axs[row, col].set_ylabel('PWM0')
        else:
            axs[row, col].set_ylabel('RPM0')
        col = 1
        for j in range(self.NUM_DRONES):
            axs[row, col].plot(t, self.states[j, 13, :], label="drone_"+str(j))
        axs[row, col].set_xlabel('time (s)')
        if pwm:
            axs[row, col].set_ylabel('PWM1')
        else:
            axs[row, col].set_ylabel('RPM1')
        col = 0
        row = 7
        for j in range(self.NUM_DRONES):
            axs[row, col].plot(t, self.states[j, 14, :], label="drone_"+str(j))
        axs[row, col].set_xlabel('time (s)')
        if pwm:
            axs[row, col].set_ylabel('PWM2')
        else:
            axs[row, col].set_ylabel('RPM2')
        col = 1
        for j in range(self.NUM_DRONES):
            axs[row, col].plot(t, self.states[j, 15, :], label="drone_"+str(j))
        axs[row, col].set_xlabel('time (s)')
        if pwm:
            axs[row, col].set_ylabel('PWM3')
        else:
            axs[row, col].set_ylabel('RPM3')

        #### Drawing options #######################################
        for i in range (rows):
            for j in range (cols):
                axs[i, j].grid(True)
                axs[i, j].legend().remove()
                # axs[i, j].legend(loc='upper right',
                #          frameon=True
                #          )
        fig.subplots_adjust(left=0.2,
                            bottom=0.075,
                            right=0.8,
                            top=0.98,
                            wspace=0.25,
                            hspace=0.34
                            )
        fig.legend(["drone_1", "drone_2", "drone_3", "drone_4"], loc="lower center", ncol=4)

        plt.savefig("test.svg", format="svg")
        plt.show()
