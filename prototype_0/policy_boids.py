# Author: Atoosa
# Description:
# This is an implementation of boids  algorithm
# References:
#   http://www.vergenet.net/~conrad/boids/pseudocode.html
#   https://www.labri.fr/perso/nrougier/from-python-to-numpy/#fluid-dynamics
#   http://www.red3d.com/cwr/boids/


import numpy as np
from policy import Policy
import math


class Policy_Boids(Policy):
    def __init__(self, world, dim_obs=3, dim_action=2, seed=2):
        self.world = world
        self.dim_obs = dim_obs
        self.dim_action = dim_action
        self.num_vehicles=None

        self.neighborhood_dist = 0.1  # arbitrary

        # if seed is None:
        #     self.seed = int(time.time())
        # else:
        #     self.seed = seed

        # np.random.seed(self.seed)

    def get_action(self, obs):
        """
        obs:
            refer to world.get_obs().
        """
        if self.num_vehicles is None:
            self.num_vehicles = obs.shape[0] - 1
            
        action = np.zeros([self.num_vehicles, 2])
        
        # only need the ansolute observations
        obs = obs[0,:]
        for i in np.arange(self.num_vehicles): 
            curr_x = obs[i*4]
            curr_y = obs[i*4+1]
            current_angle = obs[i*4+2]
            current_velocity = obs[i*4+3]
            
            v1 = self.rule1(obs, i)
            v2 = self.rule2(obs, i)
            v3 = self.rule3(obs, i)
            
            # angle(steering): the first two rules determine the angle, prioritizing the collision avoidance
            # there are not any boids that are too close
            if v2 == -100:
                # there are not any boids nearby to follow either
                if v1 ==  -100:
                    # no steering, continue in the same direction
                    action[i,0] = 0
                else:
                    # steer towards the center of the nearby boids
                    # 0.1 is just a regularization term, we want to change the direction slowly 
                    action[i,0] = 0.1*v1 - current_angle
            else:
                # steer away from the center of the nearby boids
                # 0.5 is again a regularization term, but it's higher than before, because we don't want any collisions.
                action[i,0] = 0.5*v2 - current_angle
            
            # velocity
            # maximum speed is assumed to be 0.2, don't want the boids to move too fast
            if v3 != 0:
                # move with the same speed as your neighbors
                action[i,1] = v3 if v3<2 else 2
            else:
                action[i,1] = 0.1 #current_velocity : bug: stay still because the initial velocities are zero
            
        return action
    
    # Calculating distance between two points, dealing with the edge of the world
    # Using Star's implementation
    def distance(self, point1, point2):
        d = np.abs(np.array(point1) - np.array(point2))
        if d[0]>0.5:
            d[0] = 1-d[0]
        if d[1]>0.5:
            d[1] = 1-d[1]
        return np.sqrt(d[0]*d[0] + d[1]*d[1])
    
    def rule1(self, obs, vehicle, dist=0.1): #0.05
    # Boids try to fly towards the centre of mass of neighboring boids.
    # 1: Find the neighbors within a radius of 0.1 and calculate the mean of their positions
    # 2: Change the angle of the current boid towards the center
        
        angle_sum = 0
        x_sum = 0
        y_sum = 0
        
        curr_x = obs[vehicle*4]
        curr_y = obs[vehicle*4+1]
        
        count = 0
        for i in np.arange(self.num_vehicles):
            x = obs[i*4]
            y = obs[i*4+1]
            angle = obs[i*4+2]
            
            # for the neighbors of vehicle:
            if i != vehicle and self.distance([curr_x, curr_y], [x, y]) < dist:
                x_sum = x_sum + x
                y_sum = y_sum + y
                angle_sum = angle_sum + angle
                count = count + 1
        
        if count != 0:
            mean_angle = angle_sum / count
            mean_x = x_sum / count
            mean_y = y_sum / count
            
            desired_angle = math.atan2(mean_y-curr_y, mean_x-curr_x)
            
            # the output of atan2 is between -pi and pi but our angle is between 0 and 2pi
            if desired_angle<0:
                desired_angle = desired_angle + 2*math.pi
            
            return  desired_angle
        
        # if there are no boids around
        return -100
    
    def rule2(self, obs, vehicle, dist=0.0005):
        # Boids try to keep a small distance away from other objects (including other boids).
        # 1: Find the nearest neighbors
        # 2: Set the angle to stay away from them
        
        curr_x = obs[vehicle*4]
        curr_y = obs[vehicle*4+1]
        current_angle = obs[vehicle*4+2]
        
        angle_sum = 0
        x_sum = 0
        y_sum = 0
        
        count = 0
        for i in np.arange(self.num_vehicles):
            if i != vehicle:
                x = obs[i*4]
                y = obs[i*4+1]
                angle = obs[i*4+2]
                
                if self.distance([curr_x, curr_y], [x, y]) < dist:
                    x_sum = x_sum + x
                    y_sum = y_sum + y
                    angle_sum = angle_sum + angle
                    count = count + 1
        
        if count != 0:
            mean_angle = angle_sum / count
            mean_x = x_sum / count
            mean_y = y_sum / count
            
            # repulsive
            desired_angle = -1 * math.atan2(mean_y-curr_y, mean_x-curr_x)
            
            # the output of atan2 is between -pi and pi but our angle is between 0 and 2pi
            if desired_angle<0:
                desired_angle = desired_angle + 2*math.pi
            
            return desired_angle
        
        # if there are no boids around
        return -100
    
    def rule3(self, obs, vehicle, dist=0.2):
        #Boids try to match velocity with near boids.
        
        velocity_sum = 0
        
        curr_x = obs[vehicle*4]
        curr_y = obs[vehicle*4+1]
        
        count = 0
        for i in np.arange(self.num_vehicles):
            x = obs[i*4]
            y = obs[i*4+1]
            velocity = obs[i*4+3]
            
            # for the neighbors of vehicle:
            if i != vehicle and self.distance([curr_x, curr_y], [x, y]) < dist: #math.sqrt((x-curr_x)**2+(y-curr_y)**2)<dist:
                velocity_sum = velocity_sum + velocity
                count = count + 1
        
        out = 0
        if count != 0:
            out = velocity_sum / count
        
        return out