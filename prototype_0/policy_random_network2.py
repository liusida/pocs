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
        self.perception = 0.15

        # if seed is None:
        #     seed = int(time.time()) # generate a new system every time
        # self.seed = seed
        # np.random.seed(seed)

        self.model = self.init_model()

        print(f"Random network with seed {seed}")

    def init_model(self):
        return np.random.random((self.dim_obs,self.dim_action))*2-1

    def relu(self, x):
        return np.maximum(np.zeros(x.shape),x)

    def forward_pass(self, inputs):
        # execute forward pass of network

        weights = self.model[:len(inputs),:]

        output = np.dot(inputs,weights)
        output = self.relu(output)

        return output

    def get_action(self, obs):
        if self.num_vehicles is None:
            self.num_vehicles = obs.shape[0] - 1

        action = np.zeros([self.num_vehicles, 2])

        obs = obs[0,:]
        for i in np.arange(self.num_vehicles): 

            neigh = self.find_nearest_neighbors(obs, i)
            neigh=neigh.flatten()
            
            act = self.forward_pass(inputs=neigh)
            action[i,0]=act[0]*0.1
            action[i,1]=(act[1]+1)*0.1

        return action

    def find_nearest_neighbors(self, obs, vehicle):
        '''
        returns matrix of nearest neighbors of vehicle within self.perception (including self information)
            neigh = (# neighbors + 1 x 4)
        '''
        self_info = obs[vehicle*4:vehicle*4+4] 
        x1 = obs[vehicle*4] 
        y1 = obs[vehicle*4+1]
        others = np.concatenate((obs[:4*(vehicle)],obs[4*(vehicle)+4:]))
        neigh = []
        neigh.append(self_info)
        for i in range(self.num_vehicles-1):
            x2 = others[i*4]
            y2 = others[i*4+1]   

            d = np.sqrt((x2-x1)**2 + (y2-y1)**2)

            if d<self.perception:
                neigh.append(others[i*4:i*4+4])

        return np.array(neigh)



    