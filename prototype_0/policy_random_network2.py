# Author: Caitlin
# Description:
#   Random single layer, fully connected feed forward neural network
#   Inputs: Position of neighbors of individual (maybe all others in the swarm? closest 4 neighbors?)
#           i.e. only first row of obs
#   Outputs: Angle offset and velocity for inidividual to move at the next time step

import numpy as np
from utils import log

class Policy_Random_Network2:
    def __init__(self, world, dim_obs=4, dim_action=2, num_vehicles=None):
        self.world = world
        self.dim_obs = dim_obs
        self.dim_action = dim_action
        self.num_vehicles = None
        self.perception = 0.25

        self.model = self.init_model()

    def init_model():
        # single fully connected layer
        # inputs: position of all other individuals in the swarm
        # outputs: action for each individual
        # return model (i.e. weights)
        pass

    def relu():
        pass

    def forward_pass():
        # execute the network
        pass

    def get_action(self, obs):
        if self.num_vehicles is None:
            self.num_vehicles = obs.shape[0] - 1
        
        obs = obs[0,:]



    