from math import pi
from p5 import *  # pip install p5
import numpy as np

# Global variables
vehicles_pos = {}
vehicles_orientation = {}
bg = None

# P5 default callbacks


def setup():
    global vehicles_pos, vehicles_orientation, bg
    size(1920, 1200)
    no_stroke()
    # bg = load_image("bg.jpg")

    for v in range(10):
        vehicles_pos[v] = np.array([400., 0.])
        vehicles_orientation[v] = pi


def draw():
    background(27, 73, 98)
    # image(bg, (0, 0))
    # bg()
    apply_rules()
    status()
    draw_all_vehicles()

# Custom functions


def apply_rules():
    global vehicles_pos, vehicles_orientation
    velocity = 3.
    for v in vehicles_pos:
        vehicles_orientation[v] += random_gaussian(mean=0, std_dev=0.1)
        vehicles_pos[v][0] += sin(vehicles_orientation[v]) * velocity
        vehicles_pos[v][1] -= cos(vehicles_orientation[v]) * velocity

    # bring everyone on screen again
    for v in vehicles_pos:
        vehicles_pos[v][0] = vehicles_pos[v][0] % width
        vehicles_pos[v][1] = vehicles_pos[v][1] % height


def status():
    text(f"step:{frame_count}, fps:{frame_rate}", 0, 0)


def draw_all_vehicles():
    with push_style():
        no_stroke()
        for v in vehicles_pos:
            vehicle(vehicles_pos[v], vehicles_orientation[v])


def bg():
    with push_style():
        stroke(100)
        stripe_width = 30.
        for i in range(10):
            line(0, i*stripe_width*2, width, i*stripe_width*2)
            line(i*stripe_width*2, 0, i*stripe_width*2, height)


def vehicle(pos, orient):
    """
    draw a vehicle
    pos: (x,y)
    orient: degree in [0, 2*pi]
    """
    p1 = [0, -10]
    p2 = [-3, +5]
    p3 = [+3, +5]
    p4 = [-1, -5]
    p5 = [+1, -5]
    with push_matrix():
        with push_style():
            translate(*pos)
            rotate(orient)
            scale(1.3)
            fill(Color(136, 177, 112))
            triangle(p1, p2, p3)
            fill(Color(162, 184, 167))
            triangle(p1, p4, p5)


if __name__ == '__main__':
    run()
