# Author: Caitlin
# Description:
# This is an implementation of vanilla boids
# Refer to:
#   http://www.red3d.com/cwr/boids/
#   https://doi.org/10.1145/37402.37406


import numpy as np
from policy import Policy


class Policy_Boids_Vanilla(Policy):
    def __init__(self, world, dim_obs=3, dim_action=2, seed=2):
        self.world = world
        self.dim_obs = dim_obs
        self.dim_action = dim_action
        self.num_vehicles=None

        self.neighborhood_dist = 0.1  # arbitrary

        if seed is None:
            self.seed = int(time.time())
        else:
            self.seed = seed

        np.random.seed(self.seed)

    def get_action(self, obs):
        """
        obs:
            refer to world.get_obs().
        """

        if self.num_vehicles is None:
            self.num_vehicles = obs.shape[0] - 1
            
        action = np.zeros([self.num_vehicles, 2])

        obs = obs[0,:]

        for i in np.arange(self.num_vehicles): 
            curr_x = obs[i*4] * self.world.width
            curr_y = obs[i*4+1] * self.world.height
            current_angle = obs[i*4+2]

            neigh = self.find_nearest_neighbors(obs, i)

            if neigh.size !=0:
                separate_request = self.separation(neigh, i, curr_x, curr_y)
                align_request = self.alignment(neigh, i)
                cohere_request = self.cohesion(neigh, i)

                action[i,0] = - current_angle + np.mean([separate_request, align_request, cohere_request])

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

            if d<self.neighborhood_dist:
                neigh.append(others[i*4:i*4+4])

        return np.array(neigh)


    def separation(self, neigh, vehicle, curr_x, curr_y):
        xs = np.power(neigh[:,0]-curr_x,2) 
        ys = np.power(neigh[:,0]-curr_y,2)
        d = np.sqrt(xs + ys) # length 1 x n

        weighted_x = np.sum(neigh[:,0]/d)/np.sum(d)
        weighted_y = np.sum(neigh[:,1]/d)/np.sum(d)

        target_angle = np.arctan2( weighted_y, weighted_x )

        return target_angle

    def alignment(self, neigh, vehicle):
        target_angle = np.mean(neigh[:,2])
        return target_angle

    def cohesion(self, neigh, vehicle):
        mean_x = np.mean(neigh[:,0])
        mean_y = np.mean(neigh[:,1])

        target_angle = np.arctan2( mean_y, mean_x )

        return target_angle