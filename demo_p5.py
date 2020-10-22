from math import pi
from p5 import * # pip install p5
import numpy as np

vehicles_pos = {}
vehicles_orientation = {}

def setup():
    global vehicles_pos, vehicles_orientation
    size(1024, 768)
    no_stroke()
    for v in range(10):
        vehicles_pos[v] = np.array([400,0])
        vehicles_orientation[v] = pi

def apply_rules():
    global vehicles_pos, vehicles_orientation
    velocity = 2
    for v in vehicles_pos:
        vehicles_pos[v][0] += sin(vehicles_orientation[v]) * velocity
        vehicles_pos[v][1] -= cos(vehicles_orientation[v]) * velocity
        vehicles_orientation[v] += random_gaussian() * 0.05

def status():
    text(f"{frame_count}", 0, 0)

def draw_all_vehicles():
    for v in vehicles_pos:
        vehicle(vehicles_pos[v], vehicles_orientation[v])

def draw():
    background(5)
    
    apply_rules()
    status()
    draw_all_vehicles()

def vehicle(pos, orient, color=100):
    """
    draw a vehicle
    pos: (x,y)
    orient: degree in [0, 2*pi]
    """
    with push_matrix():
        translate(*pos)
        p1 = [0, -10]
        p2 = [-3, +5]
        p3 = [+3, +5]
        fill(color)
        rotate(orient)
        scale(2.0)
        triangle(p1, p2, p3)

if __name__ == '__main__':
    run()
