import numpy as np
from utils import log

class Policy:
    def __init__(self, world, dim_obs=3, dim_action=2, num_vehicles=None):
        self.world = world
        self.dim_obs = dim_obs
        self.dim_action = dim_action
        self.num_vehicles = num_vehicles

    def get_action_circle(self, obs):
        if self.num_vehicles is None:
            self.num_vehicles = obs.shape[0] - 1
        ret = np.zeros([self.num_vehicles, 2])
        ret[:,0] = -0.01
        ret[:,1] = 0.1
        return ret

    def get_action(self, obs):
        # demo for get_action_from_pos_x_pos_y_velocity
        if self.num_vehicles is None:
            self.num_vehicles = obs.shape[0] - 1
        ret = np.zeros([self.num_vehicles, 3])
        ret[:,2] = 0.1
        ret = self.world.get_action_from_pos_x_pos_y_velocity(ret)
        return ret
