import time
import threading
import argparse
import numpy as np
from p5 import *  # pip install p5

from numpy.lib.function_base import angle
np.random.seed(0)

def log(*args):
    if False:
        print(*args)

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
        steering = steering * 0.1
        acceleration = acceleration * 10
        self.angle += steering
        self.angle = self.angle % (2 * np.pi)
        self.velocity = self.world.default_velocity + acceleration
        if self.velocity < 0:
            self.velocity = 0
        # update position after updating angle and velocity, the vehicle can respond faster
        self.pos_x += np.sin(self.angle) * self.velocity * self.world.dt
        self.pos_y += np.cos(self.angle) * self.velocity * self.world.dt

        self.pos_x = self.pos_x % self.world.width
        self.pos_y = self.pos_y % self.world.height

    def get_obs(self):
        # Normalized observations (0,1)
        return [self.pos_x / self.world.width, self.pos_y / self.world.height, self.angle / 2 / np.pi]


class World:
    def __init__(self):
        self.dt = 0.1       # delat time, step size
        self.default_velocity = 100.0

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
        self.dim_obs = (1 + num) * v.dim_obs
        self.dim_action = v.dim_action

    def step(self, action):
        for i, vehicle in enumerate(self.vehicles):
            vehicle.step(action[i])
        self.calculate_metrics()
        return self.get_obs()

    def reset(self):
        for i, vehicle in enumerate(self.vehicles):
            vehicle.reset()
            vehicle.pos_x = 100 + i * 20
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
            ret.append(np.concatenate([vehicle.get_obs(), all_members]))
        return np.array(ret)

    def calculate_metrics(self):
        global g_metrics
        g_metrics[0] = self.vehicles[0].pos_x / self.width # TODO: for demostration. Should be entropy or something.


class Policy:
    def __init__(self, dim_obs=3, dim_action=2):
        self.dim_obs = dim_obs
        self.dim_action = dim_action
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
        s = np.dot(obs, self.weights) + self.bias
        s /= len(obs) # Normalization
        return s


class Simulation(threading.Thread):
    def run(self):
        global g_obs, g_world, args
        g_world = World()
        g_world.init_vehicles(args.num_vehicles)

        g_policy = Policy(dim_obs=g_world.dim_obs, dim_action=g_world.dim_action)

        obs = g_world.reset()

        while True:
            action = g_policy.get_action(obs)
            obs = g_world.step(action)
            g_obs = obs
            time.sleep(0.01)


# P5 interface

def setup():
    size(g_world.width, g_world.height)
    no_stroke()

def draw():
    hack_check_window_size()
    background(27, 73, 98)
    text(f"Metrics: {g_metrics[0]:.03f}", 10, 10)
    all_vehicles = g_obs[0,3:]
    all_vehicles = all_vehicles.reshape([-1,3])
    for v in all_vehicles:
        draw_vehicle( *v )

def hack_check_window_size():
    """ I use tile in Linux, so window size changes after it opens. """
    global g_world
    if g_world.width != p5.sketch.size[0] and g_world.height != p5.sketch.size[1]:
        g_world.width = p5.sketch.size[0]
        g_world.height = p5.sketch.size[1]

def draw_vehicle(pos_x, pos_y, angel):
    p1 = [0, 10]
    p2 = [-3, -5]
    p3 = [+3, -5]
    p4 = [-1, +5]
    p5 = [+1, +5]
    with push_matrix():
        with push_style():
            translate(pos_x * g_world.width, pos_y * g_world.height)
            rotate(-angel*2*np.pi)
            scale(1.3)
            fill(Color(136, 177, 112))
            triangle(p1, p2, p3)
            fill(Color(162, 184, 167))
            triangle(p1, p4, p5)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--num-vehicles", type=int, default=10, help="Number of vehicles")
    args = parser.parse_args()
    g_obs = None
    g_world = None
    g_metrics = [0.]
    print("Press Ctrl+C twice to quit...")
    sim = Simulation()
    sim.start()
    # after start simulation thread, start to draw using p5
    run()
    
