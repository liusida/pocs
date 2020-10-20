import os, random
import numpy as np
import pybullet as p
import pybullet_data
from PyBulletWrapper.pybullet_wrapper.base import BaseWrapperPyBullet
from PyBulletWrapper.pybullet_wrapper.handy import HandyPyBullet

p = BaseWrapperPyBullet(p)
p = HandyPyBullet(p)

p.connectPy()
# walker, = p.loadMJCF("./bodies/walker2d_0.xml")
# num_joints = p.getNumJointsPy(bodyUniqueId=walker)

cars = []
for i in range(10):
    car = p.loadURDF(os.path.join(pybullet_data.getDataPath(), "racecar/racecar.urdf"))
    p.resetBasePositionAndOrientation(car, [0, i, 0], [0, 0, 0, 1])
    cars.append(car)

inactive_wheels = [3, 5, 7]
wheels = [2]
steering = [4, 6]

for wheel in inactive_wheels:
    for car in cars:
        p.setJointMotorControl2(car, wheel, p.VELOCITY_CONTROL, targetVelocity=0, force=0)


while True:

    for car in cars:
        for wheel in wheels:
            p.setJointMotorControl2(car,
                                wheel,
                                p.VELOCITY_CONTROL,
                                targetVelocity=random.random() * 10,
                                force=random.random() * 5)

        for steer in steering:
            p.setJointMotorControl2(car, steer, p.POSITION_CONTROL, targetPosition=random.random()-0.2)

    p.stepSimulation()
    p.sleepPy(0.01)
