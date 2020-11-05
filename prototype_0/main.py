import time
import threading
import numpy as np
from p5 import *  # pip install p5

from numpy.lib.function_base import angle
np.random.seed(0)


class Vehicle:
    def __init__(self, world):
        self.world = world
        self.dim_obs = 3    # pos_x, pos_y, angle
        self.dim_action = 2  # steering, acceleration
        self.reset()

    def reset(self):
        self.pos_x = 0      # [0 ~ world.width]
        self.pos_y = 0      # [0 ~ world.height]
        self.angle = 0      # [0 ~ 2 pi)
        self.velocity = 0   # [0 ~ infty)

    def step(self, action):
        steering, acceleration = action
        steering *= 0.001 # Definitly need Normalization
        acceleration *= 1.0
        self.angle += steering
        self.angle = self.angle % (2 * np.pi)
        self.velocity += acceleration
        self.velocity = np.min([self.world.max_velocity, self.velocity])
        # update position after updating angle and velocity, the vehicle can respond faster
        self.pos_x += np.sin(self.angle) * self.velocity * self.world.dt
        self.pos_y += np.cos(self.angle) * self.velocity * self.world.dt

        self.pos_x = self.pos_x % self.world.width
        self.pos_y = self.pos_y % self.world.height

    def get_obs(self):
        return [self.pos_x, self.pos_y, self.angle]


class World:
    def __init__(self):
        self.dt = 0.1       # delat time, step size
        self.max_velocity = 100.0

        self.dim_obs = 0
        self.dim_action = 0

        self.width = 1000
        self.height = 1000
        self.vehicles = []

    def init_vehicles(self, num):
        v = None
        self.vehicles = []
        for i in range(num):
            v = Vehicle(self)
            self.vehicles.append(v)
        self.dim_obs = 1 + (1 + num) * v.dim_obs
        self.dim_action = v.dim_action

    def step(self, action):
        for i, vehicle in enumerate(self.vehicles):
            vehicle.step(action[i])
        return self.get_obs()

    def reset(self):
        for i, vehicle in enumerate(self.vehicles):
            vehicle.reset()
            vehicle.pos_x = 100 + i * 10
            vehicle.pos_y = 100
        return self.get_obs()

    def get_obs(self):
        """ observation = [ vehicle itself, all members ] x n
        """
        all_members = []
        for vehicle in self.vehicles:
            all_members.append(vehicle.get_obs())
        all_members = np.array(all_members).flatten()
        ret = []
        for i, vehicle in enumerate(self.vehicles):
            ret.append(np.concatenate([[i], vehicle.get_obs(), all_members]))
        return np.array(ret)


class Policy:
    def __init__(self, dim_obs=3, dim_action=2):
        self.dim_obs = dim_obs
        self.dim_action = dim_action
        self.weights = None
        self.bias = None
        self.init_params()

    def init_params(self):
        self.weights = np.random.random([self.dim_obs, self.dim_action])
        self.bias = np.random.random([self.dim_action])
        print("weights")
        print(self.weights)
        print("bias")
        print(self.bias)

    def get_action(self, obs):
        return np.dot(obs, self.weights) + self.bias


class Simulation(threading.Thread):
    def run(self):
        global g_obs
        g_world = World()
        g_world.init_vehicles(10)

        g_policy = Policy(dim_obs=g_world.dim_obs, dim_action=g_world.dim_action)

        obs = g_world.reset()

        while True:
            action = g_policy.get_action(obs)
            obs = g_world.step(action)
            g_obs = obs
            time.sleep(0.1)


# P5 interface

def setup():
    size(1000, 1000)
    no_stroke()

def draw():
    global g_obs
    background(27, 73, 98)
    all_vehicles = g_obs[0,4:]
    all_vehicles = all_vehicles.reshape([-1,3])
    for v in all_vehicles:
        draw_vehicle( *v )
    print(all_vehicles)

def draw_vehicle(pos_x, pos_y, angel):
    p1 = [0, 10]
    p2 = [-3, -5]
    p3 = [+3, -5]
    p4 = [-1, +5]
    p5 = [+1, +5]
    with push_matrix():
        with push_style():
            translate(pos_x, pos_y)
            rotate(-angel)
            scale(1.3)
            fill(Color(136, 177, 112))
            triangle(p1, p2, p3)
            fill(Color(162, 184, 167))
            triangle(p1, p4, p5)


if __name__ == "__main__":
    g_obs = None
    sim = Simulation()
    sim.start()
    # after start simulation thread, start to draw using p5
    run()
