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
            self.num_vehicles = obs.shape[0] - 1
        obs = obs.reshape([self.num_vehicles + 1, self.num_vehicles, 4])
        obs = obs[1:,:,:]
        inverse_obs = (0.5 - np.abs(obs)) * ((obs>=0).astype(int)*2-1)
        distances = obs[:,:,0] ** 2 + obs[:,:,1] ** 2
        neighbors = np.argsort(distances, axis=1)

        new_obs = []
        for i in range(self.num_vehicles):
            obs_of_neighbors = inverse_obs[i, neighbors[i], :]
            new_obs.append(obs_of_neighbors)
        new_obs = np.array(new_obs).reshape([self.num_vehicles, -1])

        s = np.dot(new_obs, self.weights) + self.bias
        s /= self.num_vehicles  # Normalization
        return s

class Policy_Boids(Policy):
    radius = 0.001
    def get_action(self, obs):
        if self.num_vehicles is None:
            self.num_vehicles = obs.shape[0]
        ret = np.zeros([self.num_vehicles, 2])
        # TODO: implement the boids
        return ret


class Policy_Stationary(Policy):
    def get_action(self, obs):
        if self.num_vehicles is None:
            self.num_vehicles = obs.shape[0]
        ret = np.zeros([self.num_vehicles, 2])
        return ret