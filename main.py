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
from trainer import Trainer
from logger import Logger
import utils


def main(args):


    #!--------------- Setup options ---------------
    is_sim          = args.is_sim # Run in simulation?
    obj_mesh_dir    = os.path.abspath(args.obj_mesh_dir) if is_sim else None # Directory containing 3D mesh files (.obj) of objects to be added to simulation
    num_obj         = args.num_obj if is_sim else None # Number of objects to add to simulation
    tcp_host_ip     = args.tcp_host_ip if not is_sim else None # IP and port to robot arm as TCP client (UR5)
    tcp_port        = args.tcp_port if not is_sim else None
    rtc_host_ip     = args.rtc_host_ip if not is_sim else None # IP and port to robot arm as real-time client (UR5)
    rtc_port        = args.rtc_port if not is_sim else None
    if is_sim:
        workspace_limits = np.asarray([[-0.724, -0.276], [-0.224, 0.224], [-0.0001, 0.4]]) # Cols: min max, Rows: x y z (define workspace limits in robot coordinates)
    else:
        workspace_limits = np.asarray([[0.3, 0.748], [-0.224, 0.224], [-0.255, -0.1]]) # Cols: min max, Rows: x y z (define workspace limits in robot coordinates)
    heightmap_resolution = args.heightmap_resolution # Meters per pixel of heightmap
    random_seed     = args.random_seed
    force_cpu       = args.force_cpu

    #!------------- Algorithm options -------------
    method                  = args.method # 'reactive' (supervised learning) or 'reinforcement' (reinforcement learning ie Q-learning)
    push_rewards            = args.push_rewards if method == 'reinforcement' else None  # Use immediate rewards (from change detection) for pushing?
    future_reward_discount  = args.future_reward_discount
    experience_replay       = args.experience_replay # Use prioritized experience replay?
    heuristic_bootstrap     = args.heuristic_bootstrap # Use handcrafted grasping algorithm when grasping fails too many times in a row?
    explore_rate_decay      = args.explore_rate_decay
    grasp_only              = args.grasp_only

    #!-------------- Testing options --------------
    is_testing          = args.is_testing
    max_test_trials     = args.max_test_trials # Maximum number of test runs per case/scenario
    test_preset_cases   = args.test_preset_cases
    test_preset_file    = os.path.abspath(args.test_preset_file) if test_preset_cases else None

    #!------ Pre-loading and logging options ------
    load_snapshot       = args.load_snapshot # Load pre-trained snapshot of model?
    snapshot_file       = os.path.abspath(args.snapshot_file)  if load_snapshot else None
    continue_logging    = args.continue_logging # Continue logging from previous session
    continue_logging    = False
    logging_directory   = os.path.abspath(args.logging_directory) if continue_logging else os.path.abspath('logs')
    save_visualizations = args.save_visualizations # Save visualizations of FCN predictions? Takes 0.6s per training step if set to True

    #? Set random seed
    np.random.seed(random_seed)

    #* Initialize pick-and-place system (camera and robot)
    robot = Robot(is_sim, obj_mesh_dir, num_obj, workspace_limits,
                    tcp_host_ip, tcp_port, rtc_host_ip, rtc_port,
                    is_testing, test_preset_cases, test_preset_file)

    #? Initialize data logger
    logger = Logger(continue_logging, logging_directory)
    logger.save_camera_info(robot.cam_intrinsics, robot.cam_pose, robot.cam_depth_scale) # Save camera intrinsics and pose
    logger.save_heightmap_info(workspace_limits, heightmap_resolution) # Save heightmap parameters

    #? Initialize variables for heuristic bootstrapping and exploration probability
    no_change_count = [2, 2] if not is_testing else [0, 0]
    explore_prob    = 0.5 if not is_testing else 0.0

    # Quick hack for nonlocal memory between threads in Python 2
    nonlocal_variables = {  'executing_action' : False,
                            'primitive_action' : None,
                            'best_pix_ind' : None,
                            'push_success' : False,
                            'grasp_success' : False}

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
    sensor_data, img = robot.explore()
    logger.save_force_sensor_data(1,np.array(sensor_data))
    logger.save_cognition_images(1, img)
    logger.save_cognition_data(1,img)
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
    parser.add_argument('--is_sim', dest='is_sim', action='store_true', default=False,                                    help='run in simulation?')
    parser.add_argument('--obj_mesh_dir', dest='obj_mesh_dir', action='store', default='objects/blocks',                  help='directory containing 3D mesh files (.obj) of objects to be added to simulation')
    parser.add_argument('--num_obj', dest='num_obj', type=int, action='store', default=10,                                help='number of objects to add to simulation')
    parser.add_argument('--tcp_host_ip', dest='tcp_host_ip', action='store', default='100.127.7.223',                     help='IP address to robot arm as TCP client (UR5)')
    parser.add_argument('--tcp_port', dest='tcp_port', type=int, action='store', default=30002,                           help='port to robot arm as TCP client (UR5)')
    parser.add_argument('--rtc_host_ip', dest='rtc_host_ip', action='store', default='100.127.7.223',                     help='IP address to robot arm as real-time client (UR5)')
    parser.add_argument('--rtc_port', dest='rtc_port', type=int, action='store', default=30003,                           help='port to robot arm as real-time client (UR5)')
    parser.add_argument('--heightmap_resolution', dest='heightmap_resolution', type=float, action='store', default=0.002, help='meters per pixel of heightmap')
    parser.add_argument('--random_seed', dest='random_seed', type=int, action='store', default=1234,                      help='random seed for simulation and neural net initialization')
    parser.add_argument('--cpu', dest='force_cpu', action='store_true', default=False,                                    help='force code to run in CPU mode')

    # ------------- Algorithm options -------------
    parser.add_argument('--method', dest='method', action='store', default='reinforcement',                               help='set to \'reactive\' (supervised learning) or \'reinforcement\' (reinforcement learning ie Q-learning)')
    parser.add_argument('--push_rewards', dest='push_rewards', action='store_true', default=False,                        help='use immediate rewards (from change detection) for pushing?')
    parser.add_argument('--future_reward_discount', dest='future_reward_discount', type=float, action='store', default=0.5)
    parser.add_argument('--experience_replay', dest='experience_replay', action='store_true', default=False,              help='use prioritized experience replay?')
    parser.add_argument('--heuristic_bootstrap', dest='heuristic_bootstrap', action='store_true', default=False,          help='use handcrafted grasping algorithm when grasping fails too many times in a row during training?')
    parser.add_argument('--explore_rate_decay', dest='explore_rate_decay', action='store_true', default=False)
    parser.add_argument('--grasp_only', dest='grasp_only', action='store_true', default=False)

    # -------------- Testing options --------------
    parser.add_argument('--is_testing', dest='is_testing', action='store_true', default=False)
    parser.add_argument('--max_test_trials', dest='max_test_trials', type=int, action='store', default=30,                help='maximum number of test runs per case/scenario')
    parser.add_argument('--test_preset_cases', dest='test_preset_cases', action='store_true', default=False)
    parser.add_argument('--test_preset_file', dest='test_preset_file', action='store', default='test-10-obj-01.txt')

    # ------ Pre-loading and logging options ------
    parser.add_argument('--load_snapshot', dest='load_snapshot', action='store_true', default=False,                      help='load pre-trained snapshot of model?')
    parser.add_argument('--snapshot_file', dest='snapshot_file', action='store')
    parser.add_argument('--continue_logging', dest='continue_logging', action='store_true', default=False,                help='continue logging from previous session?')
    parser.add_argument('--logging_directory', dest='logging_directory', action='store')
    parser.add_argument('--save_visualizations', dest='save_visualizations', action='store_true', default=False,          help='save visualizations of FCN predictions?')

    # Run main program with specified arguments
    args = parser.parse_args()
    main(args)
