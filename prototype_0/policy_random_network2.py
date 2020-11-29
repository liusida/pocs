# Author: Caitlin
# Description:
#   Random single layer, fully connected feed forward neural network
#   Inputs: Position of neighbors of individual (maybe all others in the swarm? closest 4 neighbors?)
#           i.e. only first row of obs
#   Outputs: Angle offset and velocity for inidividual to move at the next time step

import numpy as np
from utils import log
import time

class Policy_Random_Network2:
    def __init__(self, world, dim_obs=4, dim_action=2, seed=None):
        self.world = world
        self.dim_obs = dim_obs
        self.dim_action = dim_action
        self.num_vehicles = None
        self.perception = 0.25

        if seed is None:
            seed = int(time.time()) # generate a new system every time
        self.seed = seed
        np.random.seed(seed)

        self.model = self.init_model()

        print(f"Random network with seed {seed}")

    def init_model(self):
        print(self.dim_obs)
        print(self.num_vehicles)
        return np.random.random((self.dim_obs*self.num_vehicles,self.dim_action))*2-1

    def relu(self):
        pass

    def forward_pass(self):
        # execute the network
        pass

    def get_action(self, obs):
        print(self.model.shape)

        obs = obs[0,:]
        for i in np.arange(self.num_vehicles): 
            curr_x = obs[i*4]
            curr_y = obs[i*4+1] 
            current_angle = obs[i*4+2]
            current_veclocity = obs[i*4+3]

            neigh = self.find_nearest_neighbors(obs, i)
            neigh=neigh.flatten()
            print(neigh.shape)

        action = np.zeros([self.num_vehicles, 2])
        return action

    def find_nearest_neighbors(self, obs, vehicle):
        '''
        returns matrix of nearest neighbors of vehicle within self.neighborhood_dist
            neigh = (# neighbors x 4)
        '''
        x1 = obs[vehicle*4] 
        y1 = obs[vehicle*4+1] 
        others = np.concatenate((obs[:4*(vehicle)],obs[4*(vehicle)+4:]))
        neigh = []
        for i in range(self.num_vehicles-1):
            x2 = others[i*4]
            y2 = others[i*4+1]   

            d = np.sqrt((x2-x1)**2 + (y2-y1)**2)

            if d<self.perception:
                neigh.append(others[i*4:i*4+4])

        return np.array(neigh)



    