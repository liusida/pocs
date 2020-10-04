import numpy as np
import pybullet as p
from PyBulletWrapper.pybullet_wrapper.base import BaseWrapperPyBullet
from PyBulletWrapper.pybullet_wrapper.handy import HandyPyBullet

p = BaseWrapperPyBullet(p)
p = HandyPyBullet(p)

p.connectPy()
walker, = p.loadMJCF("./bodies/walker2d_0.xml")
num_joints = p.getNumJointsPy(bodyUniqueId=walker)
while True:
    
    p.stepSimulation()
    p.sleepPy(0.01)