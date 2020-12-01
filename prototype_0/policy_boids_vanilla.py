# Author: Caitlin
# Description:
# This is an implementation of vanilla boids
# Refer to:
#   http://www.red3d.com/cwr/boids/
#   https://doi.org/10.1145/37402.37406


import numpy as np
from policy import Policy

from sklearn.preprocessing import normalize

class Policy_Boids_Vanilla(Policy):
    def __init__(self, world, dim_obs=3, dim_action=2):

        self.world = world
        self.dim_obs = dim_obs
        self.dim_action = dim_action
        self.num_vehicles = None

        self.neighborhood_dist = 0.15  # arbitrary

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
            curr_x = obs[i*4]
            curr_y = obs[i*4+1] 
            current_angle = obs[i*4+2]
            current_velocity = obs[i*4+3]

            neigh = self.find_nearest_neighbors(obs, i)

            if neigh.size !=0:
                separate_request = self.separation(neigh, i, curr_x, curr_y)
                align_request = self.alignment(neigh, i, curr_x, curr_y)
                cohere_request = self.cohesion(neigh, i, curr_x, curr_y)
                
                # I don't know why method A and B produce very different values for steering. But by printing them out so we can see which method we are using.
                # You can remove them once the agents are stable.
                if separate_request is not None:
                    action[i,0] = np.mean([0.25*align_request, 0.25*cohere_request[0], 0.5*separate_request[0]]) - current_angle
                    if i==0:
                        print(f"A {action[i,0]:.03f} current_angle {current_angle:.03f}")
                    action[i,1] = np.mean([separate_request[1], cohere_request[1]]) * 5
                else:
                    action[i,0] = np.mean([align_request, cohere_request[0]]) - current_angle
                    if i==0:
                        print(f"B {action[i,0]:.03f} current_angle {current_angle:.03f}")
                    action[i,1] = cohere_request[1]

            else:  # continue in same direction/with same speed
                action[i,0] = 0
                if i==0:
                    print(f"C {action[i,0]:.03f} current_angle {current_angle:.03f}")
                action[i,1] = current_velocity

        return action

    # you could use this distance function in find_nearest_neighbors() to calculate the distance of two point, it covers the edge case.
    # I copied from metric_hse.py
    def distance(self, point1, point2):
        d = np.abs(point1 - point2)
        if d[0]>0.5:
            d[0] = 1-d[0]
        if d[1]>0.5:
            d[1] = 1-d[1]
        return np.sqrt(d[0]*d[0] + d[1]*d[1])

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
        # avoid collisions with neighbors that are too close

        thresh = .25*self.neighborhood_dist
        too_close = []
        for n in neigh:
            if np.linalg.norm([n[0]-curr_x, n[1]-curr_y])<thresh:
                too_close.append(n)

        if len(too_close) > 0:
            too_close = np.array(too_close)

            mean_x = np.mean(too_close[:,0])
            mean_y = np.mean(too_close[:,1])

            x_vec = curr_x - mean_x
            y_vec = curr_y - mean_y

            target_angle = np.arctan2(y_vec, x_vec)

            if target_angle < 0:
                target_angle = np.pi/2 + np.abs(target_angle)
            else:
                target_angle = np.pi/2 - target_angle

            target_speed = np.linalg.norm([x_vec,y_vec])

            return target_angle, target_speed
        return None

    def alignment(self, neigh, vehicle, curr_x, curr_y):
        # align direction with nearest neighbors

        target_angle = np.mean(neigh[:,2])
        return target_angle

    def cohesion(self, neigh, vehicle, curr_x, curr_y):

        # centroid of local flock
        mean_x = np.mean(neigh[:,0])
        mean_y = np.mean(neigh[:,1])

        x_vec = mean_x - curr_x
        y_vec = mean_y - curr_y

        target_angle = np.arctan2(y_vec, x_vec) 
        if target_angle < 0:
            target_angle = np.pi/2 + np.abs(target_angle)
        else:
            target_angle = np.pi/2 - target_angle

        target_speed = np.linalg.norm([x_vec, y_vec])

        return target_angle, target_speed