import time
import threading
import argparse
import numpy as np
np.random.seed(0)
from p5 import *  # pip install p5

from policy import Policy, Policy_Boids
from utils import log
from world import World

class Simulation(threading.Thread):
    def run(self):
        global g_obs, g_world, g_metrics, args
        g_world = World()
        g_world.init_vehicles(args.num_vehicles)

        g_policy = Policy(dim_obs=g_world.dim_obs, dim_action=g_world.dim_action)

        obs = g_world.reset()

        while True:
            action = g_policy.get_action(obs)
            obs, info = g_world.step(action)
            g_metrics[0] = info["metrics"]
            # set g_obs for visualization
            g_obs = g_world.get_absolute_obs()
            time.sleep(0.01)


# P5 interface

def setup():
    size(g_world.width, g_world.height)
    no_stroke()


def draw():
    hack_check_window_size()
    background(27, 73, 98)
    draw_info()
    all_vehicles = g_obs
    for i, v in enumerate(all_vehicles):
        draw_vehicle(*v, vehicle_id=i)


def draw_info():
    global g_last_step
    step_per_frame = g_world.time_step - g_last_step
    g_last_step = g_world.time_step
    with push_matrix():
        translate(10, 10)
        text(f"Metrics: {g_metrics[0]:.03f}", 0, 0)
        text(f"Step per frame: {step_per_frame}", 0, 15)


def hack_check_window_size():
    """ I use tile in Linux, so window size changes after it opens. """
    global g_world
    if g_world.width != p5.sketch.size[0] and g_world.height != p5.sketch.size[1]:
        g_world.width = p5.sketch.size[0]
        g_world.height = p5.sketch.size[1]


def draw_vehicle(pos_x, pos_y, angel, vehicle_id):
    p1 = [0, 10]
    p2 = [-3, -5]
    p3 = [+3, -5]
    p4 = [-1, +5]
    p5 = [+1, +5]
    with push_matrix():
        with push_style():
            translate(pos_x * g_world.width, pos_y * g_world.height)
            with push_matrix():
                rotate(-angel*2*np.pi)
                scale(1.3)
                fill(Color(136, 177, 112))
                triangle(p1, p2, p3)
                fill(Color(162, 184, 167))
                triangle(p1, p4, p5)
            text(f"{vehicle_id}", 0, -20)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--num-vehicles", type=int, default=10, help="Number of vehicles")
    args = parser.parse_args()
    g_obs = None
    g_world = None
    g_metrics = [0.]
    g_last_step = 0
    print("Press Ctrl+C twice to quit...")
    # Start Simulation Thread
    sim = Simulation()
    sim.start()
    # Start to draw using p5
    run()
