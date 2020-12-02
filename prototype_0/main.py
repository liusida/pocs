import time
import numbers
import threading
import argparse
import numpy as np
from p5 import *  # pip install p5
import matplotlib.pyplot as plt
import tqdm

from policy import Policy
from policy_boids_vanilla import Policy_Boids_Vanilla
from policy_random_network import Policy_Random_Network
from policy_follow_leader import Policy_Follow_Leader
from policy_random import Policy_Random
from policy_random_network2 import Policy_Random_Network2

from metric import Metric, MicroEntropyMetric, MacroEntropyMetric, MacroMicroEntropyMetric
from metric_hse import HSEMetric

import utils
from world import World

Policy_classes = {
    "Policy": Policy,
    "Policy_Boids_Vanilla": Policy_Boids_Vanilla,
    "Policy_Random_Network": Policy_Random_Network,
    "Policy_Follow_Leader": Policy_Follow_Leader,
    "Policy_Random": Policy_Random,
    "Policy_Random_Network2": Policy_Random_Network2
}

Metric_classes = {
    "Metric": Metric,
    "Micro_Entropy": MicroEntropyMetric,
    "Macro_Entropy": MacroEntropyMetric,
    "Macro_Micro_Entropy": MacroMicroEntropyMetric,
    "HSE": HSEMetric,
}

def sequence(max_element):
    num = 0
    while max_element < 0 or num < max_element :
        yield num
        num += 1

class Simulation(threading.Thread):
    save_history = False
    max_steps = -1
    def run(self):
        assert not self.save_history or self.max_steps > 0
        global g_obs, g_world, g_metrics, g_metrics_val, metric_history
        g_world = World(seed=0)
        g_world.init_vehicles(args.num_vehicles)

        g_metrics = Metric_classes[args.metric_class](world=g_world)

        g_policy = Policy_classes[args.policy_class](world=g_world, dim_obs=g_world.dim_obs, dim_action=g_world.dim_action)

        obs = g_world.reset()
        
        metric_history = dict()

        for step_id in tqdm.tqdm(sequence(self.max_steps)):
            action = g_policy.get_action(obs)
            obs, info = g_world.step(action)
            ret = g_metrics.get_metric()
            assert isinstance(ret, dict)
            # save the metrics to the history
            if step_id == 0 and self.save_history:
                for key in ret.keys():
                    metric_history[key] = np.zeros(shape=(self.max_steps))

            g_metrics_val = ret

            if args.blind:
                print(ret, g_world.time_step)
            # if isinstance(ret, numbers.Number):
            #     g_metrics_val[0] = ret # no extra info returned.
            #     g_metrics_val[1] = 0
            # else:
            #     g_metrics_val[0] = ret[0]
            #     g_metrics_val[1] = ret[1]
            if self.save_history:
                for key, val in ret.items():
                    metric_history[key][step_id] = val

            # set g_obs for visualization
            g_obs = g_world.get_absolute_obs()
            if not args.blind:
                time.sleep(0.0001)


# P5 interface

def setup():
    size(g_world.width, g_world.height)
    no_stroke()


def draw():
    # utils.reset_timer("Outside Draw Function")
    background(27, 73, 98)
    with push_style():
        no_fill()
        stroke(0,0,0)
        rect(0, 0, g_world.width, g_world.height)
    draw_info()
    all_vehicles = g_obs
    for i, v in enumerate(all_vehicles):
        v = v[:3] # first three observations are pos_x, pos_y, angle
        draw_vehicle(*v, vehicle_id=i)
    # utils.reset_timer("In Draw Function")

def draw_info():
    global g_last_step
    step_per_frame = g_world.time_step - g_last_step
    g_last_step = g_world.time_step
    text("Origin", 0, g_world.height)
    with push_matrix():
        translate(10, 10)
        line = 0
        text(f"Time Step: {g_world.time_step}", 0, line*15)
        line += 1
        for key, val in g_metrics_val.items():
            text(f"{key}: {val:.03f}", 0, line*15)
            line += 1
        # text(f"counts: {g_metrics_val[1]}", 0, line*15)
        # line += 1
        text(f"Step per frame: {step_per_frame}", 0, line*15)

def draw_vehicle(pos_x, pos_y, angle, vehicle_id):
    p1 = [0, -10]
    p2 = [-3, 5]
    p3 = [+3, 5]
    with push_matrix():
        with push_style():
            translate(pos_x * g_world.width, (1-pos_y) * g_world.height)
            with push_matrix():
                rotate(angle)
                scale(1.3)
                fill(Color(136, 177, 112))
                triangle(p1, p2, p3)
            text(f"{vehicle_id}", 0, -20)

#
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--num-vehicles", type=int, default=10, help="Number of vehicles")
    parser.add_argument("-p", "--policy-class", type=str, default="Policy", help="The name of the policy class you want to use. Choices: %s"%
                                                                                                                (', '.join(Policy_classes.keys())))
    parser.add_argument("-m", "--metric-class", type=str, default="Metric", help="The name of the metric class you want to use. Choices: %s"%
                                                                                                                (', '.join(Metric_classes.keys())))
    parser.add_argument("-b", "--blind",  action='store_true')
    parser.add_argument("-s", "--steps", type=int, default=1000)

    parser.add_argument("--report", action="store_true")

    args = parser.parse_args()

    if args.report:
        for p in Policy_classes.keys():
            args.policy_class = p
            g_obs = None
            g_world = None
            g_metrics = None
            g_metrics_val = dict()
            g_last_step = 0
            metric_history = None
            print(f"Simulating {p}...")
            # Start Simulation Thread
            sim = Simulation()
            sim.max_steps = args.steps
            sim.save_history = True
            sim.start()
            sim.join()
            if metric_history is not None:
                for key, val in metric_history.items():
                    plt.plot(metric_history[key][:], label=f"{p} {key}")

                # if args.metric_class=="HSE":
                # else:
                #     plt.plot(metric_history[:, 0], label=f"{p} Micro")
                #     plt.plot(metric_history[:, 1], label=f"{p} Macro")
        plt.legend()
        plt.ylim((0,1))
        plt.savefig("%s_%d_steps_%d.pdf"%(args.metric_class, args.steps, int(time.time())))
        plt.savefig("%s_%d_steps_%d.png"%(args.metric_class, args.steps, int(time.time())))
    else:
        g_obs = None
        g_world = None
        g_metrics = None
        g_metrics_val = dict()
        g_last_step = 0
        metric_history = None
        print("Press Ctrl+C twice to quit...")
        # Start Simulation Thread
        sim = Simulation()
        sim.start()
        if not args.blind:
            run()
            # Start to draw using p5
        else:
            sim.join()