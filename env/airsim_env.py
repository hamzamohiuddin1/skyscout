import airsim
import numpy as np
import time

class AirSimEnv:
    def __init__(self, ip="127.0.0.1", port=41451):
        """
        Initialize the AirSim environment.
        
        Args:
            ip (str): IP address of the AirSim server
            port (int): Port number of the AirSim server
        """
        self.client = airsim.MultirotorClient(ip=ip, port=port)
        self.client.confirmConnection()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        
    def reset(self):
        """
        Reset the simulation to initial state.
        """
        self.client.reset()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        self.client.takeoffAsync().join()
        return self._get_observation()
    
    def step(self, action):
        """
        Take a step in the environment.
        
        Args:
            action (dict): Dictionary containing control commands
                Expected keys: 'vx', 'vy', 'vz', 'yaw_rate'
        """
        # Execute the action
        self.client.moveByVelocityAsync(
            action['vx'],
            action['vy'],
            action['vz'],
            duration=0.1,
            drivetrain=airsim.DrivetrainType.MaxDegreeOfFreedom,
            yaw_mode=airsim.YawMode(False, action['yaw_rate'])
        ).join()
        
        # Get the new state
        observation = self._get_observation()
        done = self._check_done()
        reward = self._calculate_reward()
        
        return observation, reward, done, {}
    
    def _get_observation(self):
        """
        Get the current state of the drone.
        """
        state = self.client.getMultirotorState()
        position = state.kinematics_estimated.position
        orientation = state.kinematics_estimated.orientation
        velocity = state.kinematics_estimated.linear_velocity
        
        return {
            'position': np.array([position.x_val, position.y_val, position.z_val]),
            'orientation': np.array([orientation.x_val, orientation.y_val, orientation.z_val, orientation.w_val]),
            'velocity': np.array([velocity.x_val, velocity.y_val, velocity.z_val])
        }
    
    def _check_done(self):
        """
        Check if the episode is done.
        """
        # Add your termination conditions here
        return False
    
    def _calculate_reward(self):
        """
        Calculate the reward for the current state.
        """
        # Add your reward function here
        return 0.0
    
    def close(self):
        """
        Clean up the environment.
        """
        self.client.enableApiControl(False)
        self.client.armDisarm(False)
