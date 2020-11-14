import numpy as np


class Vehicle:
    def __init__(self, world):
        self.world = world
        self.dim_obs = 4    # pos_x, pos_y, angle
        self.dim_action = 2  # steering, velocity_offset

    def reset(self):
        r = (np.random.random([3])+1) * 0.5
        self.pos_x = r[0]       # [0 ~ 1)
        self.pos_y = r[1]       # [0 ~ 1)
        self.angle = r[2] * 2 * np.pi              # [0 ~ 2 pi)
        self.velocity = 0      # [0 ~ infty)

    def step(self, action):
        """action = [steering, velocity]"""
        steering, velocity = action
        self.angle += steering
        self.angle = self.angle % (2 * np.pi)
        self.velocity = velocity
        if self.velocity < 0:
            self.velocity = 0
        # update position after updating angle and velocity, the vehicle can respond faster
        self.pos_x += np.sin(self.angle) * self.velocity * self.world.dt
        self.pos_y += np.cos(self.angle) * self.velocity * self.world.dt

        if not 0 <= self.pos_x <= 1:
            self.pos_x = self.pos_x % 1
        if not 0 <= self.pos_y <= 1:
            self.pos_y = self.pos_y % 1

    def get_obs(self):
        # Normalized observations (0,1)
        return [self.pos_x, self.pos_y, self.angle, self.velocity]


class World:
    def __init__(self, seed=0):
        np.random.seed(seed)

        self.dt = 0.01       # delat time, step size
        self.time_step = 0

        self.dim_obs = 0
        self.dim_action = 0

        self.width = 1000
        self.height = 1000
        self.vehicles = []

    def init_vehicles(self, num):
        self.num_vehicles = num
        v = None
        self.vehicles = []
        for i in range(num):
            v = Vehicle(self)
            self.vehicles.append(v)
        self.dim_obs = (0 + num) * v.dim_obs  # if change obs, this should change
        self.dim_action = v.dim_action

    def step(self, action):
        assert action.shape[0] == self.num_vehicles, "Should be one vehicle per item in action"
        self.time_step += 1
        for i, vehicle in enumerate(self.vehicles):
            vehicle.step(action[i])
        
        info = {}
        return self.get_obs(), info

    def reset(self):
        for i, vehicle in enumerate(self.vehicles):
            vehicle.reset()
        return self.get_obs()

    def get_obs(self):
        """
        Return:
            obs.shape = [num_vehicles + 1, num_vehicles * vehicles.dim_obs]
            First line is the positions, angles and velocities of all vehicles
            Second line to the last line are relative positions, angles, and velocities of each vehicle.
                and they are sorted from nearest to farthest.
            Positions are normalized to [0,1]
            Angles are not normalized, in [0, 2 pi]
            Velocities are normalized according to min(windowWidth, windowHeight)
        """
        all_members = []
        for vehicle in self.vehicles:
            all_members.append(vehicle.get_obs())
        all_members = np.array(all_members)
        # return all_members
        ret = []
        ret.append(all_members.flatten())
        for i, vehicle in enumerate(self.vehicles):  # if change obs, this should change
            all_members_v = all_members - vehicle.get_obs()
            all_members_v[:, :2] = (all_members_v[:, :2] + 0.5) % 1. - 0.5  # make screen continuous
            ret.append(all_members_v.flatten())
        self.current_obs = np.array(ret)
        return self.current_obs

    def get_absolute_obs(self):
        all_members = []
        for vehicle in self.vehicles:
            all_members.append(vehicle.get_obs())
        return np.array(all_members)

    def get_action_from_pos_x_pos_y_velocity(self, a):
        assert a.shape == (self.num_vehicles, 3)
        target_pos = a[:,:2]
        obs = self.current_obs[0,:]
        obs = obs.reshape([self.num_vehicles, 4])
        current_pos = obs[:,:2]
        current_angle = obs[:,2]
        offset = target_pos-current_pos
        target_angle = np.arctan2(offset[:,0], offset[:,1])

        ret = np.zeros([self.num_vehicles, 2])
        ret[:,0] = target_angle - current_angle
        ret[:,1] = a[:,2]
        return ret

    def get_action_from_angle_velocity_offset(self, a):
        assert a.shape == (self.num_vehicles, 2)
        ret = np.zeros([self.num_vehicles, 2])
        ret[:,0] = a[:,0]
        ret[:,1] = a[:,1] + 0.1
        return ret

#   y-direction
#   |        /
#   | angle /
#   |      /
#   |     /
#   |    /
#   |   /
#   |  /
#   | /
#   +------------------>  x-direction
#  Origin