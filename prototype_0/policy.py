import numpy as np
from utils import log

class Policy:
    def __init__(self, dim_obs=3, dim_action=2):
        self.dim_obs = dim_obs
        self.dim_action = dim_action
        self.num_vehicles = None
        self.weights = None
        self.bias = None
        self.init_params()

    def init_params(self):
        self.weights = np.random.random([self.dim_obs, self.dim_action]) * 2 - 1
        self.bias = np.random.random([self.dim_action]) * 2 - 1
        log("weights")
        log(self.weights)
        log("bias")
        log(self.bias)

    def get_action(self, obs):
        # A demostration of considering the relative position of neighbors
        # From closest to farthest
        if self.num_vehicles is None:
            self.num_vehicles = obs.shape[0]
        obs = obs.reshape([self.num_vehicles, self.num_vehicles, 3])
        distances = obs[:,:,0] ** 2 + obs[:,:,1] ** 2
        neighbors = np.argsort(distances, axis=1)

        new_obs = []
        for i in range(self.num_vehicles):
            obs_of_neighbors = obs[i, neighbors[i], :]
            new_obs.append(obs_of_neighbors)
        new_obs = np.array(new_obs).reshape([self.num_vehicles, -1])

        s = np.dot(new_obs, self.weights) + self.bias
        s /= self.num_vehicles  # Normalization
        return s
