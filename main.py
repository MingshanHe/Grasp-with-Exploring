#!/usr/bin/env python

import time
import os
import random
import threading
import argparse
import matplotlib.pyplot as plt
import numpy as np
import scipy as sc
import cv2
from collections import namedtuple
import torch
from torch.autograd import Variable
from robot import Robot
# from trainer import Trainer
from logger import Logger
import utils


def main(args):


    #!--------------- Setup options ---------------
    obj_mesh_dir    = os.path.abspath(args.obj_mesh_dir)  # Directory containing 3D mesh files (.obj) of objects to be added to simulation
    num_obj         = args.num_obj  # Number of objects to add to simulation
    workspace_limits = np.asarray([[-0.724, -0.276], [-0.224, 0.224], [-0.0001, 0.4]]) # Cols: min max, Rows: x y z (define workspace limits in robot coordinates)

    #? Set random seed
    # np.random.seed(random_seed)

    #? Initialize pick-and-place system (camera and robot)
    robot = Robot(obj_mesh_dir, num_obj, workspace_limits)




    robot.GoHome()

    robot.GoWork()

    robot.Explore(target_pose=[-0.724, 0.0, 0.04, np.pi/2, 0.0, np.pi/2], vel=0.005)


    # def collection_thread():
    #     threadLock = threading.Lock()
    #     while True:
    #         threadLock.acquire()
    #         force, torque = robot.get_force_sensor_data()
    #         threadLock.release()
    # collection_thread = threading.Thread(target=collection_thread)
    # collection_thread.daemon = True
    # collection_thread.start()
    # robot.move_to([-0.224, 0, 0.05], None) #[-0.724, -0.276], [-0.224, 0.224], [-0.0001, 0.4]
    # robot.move_to([-0.724, 0, 0.05], None) #[-0.724, -0.276], [-0.224, 0.224], [-0.0001, 0.4]
    # sensor_data, img = robot.explore()
    # logger.save_force_sensor_data(1,np.array(sensor_data))
    # logger.save_cognition_images(1, img)
    # logger.save_cognition_data(1,img)
    # Start main training/testing loop
    while True:
        # print('\n%s iteration: %d' % ('Testing' if is_testing else 'Training', trainer.iteration))
        # robot.move_to([-0.6, 0, 0.3], None) #[-0.724, -0.276], [-0.224, 0.224], [-0.0001, 0.4]
        time.sleep(1)
        # robot.move_to([-0.7, 0, 0.05], None)
    # force, torque = robot.get_force_sensor_data()
    # print("force:",force," torque:",torque)


if __name__ == '__main__':

    # Parse arguments
    parser = argparse.ArgumentParser(description='Train robotic agents to learn how to plan complementary pushing and grasping actions for manipulation with deep reinforcement learning in PyTorch.')

    # --------------- Setup options ---------------
    parser.add_argument('--obj_mesh_dir', dest='obj_mesh_dir', action='store', default='objects/blocks',                  help='directory containing 3D mesh files (.obj) of objects to be added to simulation')
    parser.add_argument('--num_obj', dest='num_obj', type=int, action='store', default=10,                                help='number of objects to add to simulation')

    # Run main program with specified arguments
    args = parser.parse_args()
    main(args)
