# Author: Caitlin
# Description:
# This is an implementation of vanilla boids
# Refer to:
#   http://www.red3d.com/cwr/boids/
#   https://doi.org/10.1145/37402.37406


import numpy as np
from policy import Policy


class Policy_Boids_Vanilla(Policy):
    def get_action(self, obs):
        """
        obs:
            refer to world.get_obs().
        """
        if self.num_vehicles is None:
            self.num_vehicles = obs.shape[0] - 1
        ret = np.zeros([self.num_vehicles, 2])
        # TODO: implement the boids
        return ret
