# Author: Caitlin
# Description:
#   Random behavior - null model
import numpy as np
from utils import log

class Policy_Random:
    def __init__(self, world, dim_obs=3, dim_action=2):
        self.world = world
        self.dim_obs = dim_obs
        self.dim_action = dim_action
        self.num_vehicles = None


    def get_action(self, obs):
        if self.num_vehicles is None:
            self.num_vehicles = obs.shape[0] - 1

        action = np.zeros([self.num_vehicles, 2])

        # random angle
        action[:,0] = np.random.random(action[:,0].shape) * 2 * np.pi

        # random velocity
        action[:,1] = np.random.random(action[:,1].shape) * 0.2

        return action
