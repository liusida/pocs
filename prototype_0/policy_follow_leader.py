# Author: Sida
# Description:
#   Let vehicle 0 to be the leader and move randomly, and other vehicle moving towards the leader's current position.

import time
import pickle
from pathlib import Path

import numpy as np

from policy import Policy


class Policy_Follow_Leader(Policy):
    def __init__(self, world, dim_obs=3, dim_action=2, seed=0, num_vehicles=None):
        self.world = world
        self.dim_obs = dim_obs
        self.dim_action = dim_action
        self.num_vehicles = num_vehicles
        if seed is None:
            self.seed = int(time.time())
        else:
            self.seed = seed
        np.random.seed(seed)  # generate a new system every time

    def relative_distance(self, x):
        if x>0.5:
            x -= 1
        elif x<-0.5:
            x += 1
        return x

    def get_action(self, obs):
        """
        obs:
            refer to world.get_obs().
        """
        if self.num_vehicles is None:
            self.num_vehicles = obs.shape[0] - 1
        obs = obs[0,:] # only use the first line
        action = np.zeros([self.num_vehicles,2])
        # Leader
        action[0,0] += (np.random.random() - 0.5)*0.1
        action[0,1] = 0.1
        # Followers
        for i in np.arange(1,self.num_vehicles):
            # follow the one before me (Warning: This is not local information anymore.)
            dx = obs[(i-1)*4]-obs[i*4]
            dy = obs[(i-1)*4+1]-obs[i*4+1]
            dx = self.relative_distance(dx)
            dy = self.relative_distance(dy)
            distance = dx*dx + dy*dy
            
            current_angle = obs[i*4+2]
            target_angle = np.arctan2( dy, dx )
            target_angle = np.pi/2 - target_angle
            action[i,0] = - current_angle + target_angle
            
            if distance > 0.005: # don't fall too far behind
                action[i,1] = action[i-1,1]
            else:
                action[i,1] = 0.1 * (1 - i/self.num_vehicles)
        return action
