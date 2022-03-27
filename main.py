import time
import select
import struct
import time
import os
import argparse
import numpy as np
from ur5e import UR5E



def main(args):
    # --------------- Setup options ---------------
    host_ip = args.host_ip
    use_sim = args.use_sim
    iterations = args.iterations
    train_axis = args.train_axis
    if use_sim:
        print("[SETTING PARAM INFO]: Total Training Iterations: %d" % (iterations))
        for i in range(iterations):
            print("[PARAM INFO]: %s iteration: %d" % ('Training', i))
            #? Initialize pick-and-place system (camera and robot)
            ur5e = UR5E(host_ip,use_simulation=True, train_axis=train_axis)

            # Test:----------------------------------------------------
            # ur5e.Go((-0.276, 0.0, 0.04, np.pi/2, np.pi/2, np.pi/2))
            # ur5e.Go((-0.276, 0.0, 0.04, np.pi/2, np.pi/3, np.pi/2))
            # ur5e.Go((-0.276, 0.0, 0.04, np.pi/2, -np.pi/3, np.pi/2))
            # ---------------------------------------------------------

            ur5e.GoHome()

            ur5e.GoWork()

            ur5e.Explore()

            ur5e.Train(use_heuristic=True)
            print("--------------------------------------------------")
    else:
        ur5e = UR5E(host_ip,use_simulation=False, train_axis=train_axis)

        time.sleep(1)

        ur5e.GoHome()

        ur5e.GoWork()

        ur5e.Explore()

        while True :
            time.sleep(0.1)  #sleep first since the robot may not have processed the command yet
            if ur5e.is_program_running():
                break

if __name__ == '__main__':

    # Parse arguments
    parser = argparse.ArgumentParser(description='Train robotic agents to learn how to plan complementary pushing and grasping actions for manipulation with deep reinforcement learning in PyTorch.')

    # --------------- Setup options ---------------
    parser.add_argument('--use_sim',    dest='use_sim',                 action='store_true', default=False,         help='run in simulation?')
    parser.add_argument('--train_axis', dest='train_axis',              action='store', default='x y',              help='Axis to train')
    parser.add_argument('--iterations', dest='iterations', type=int,    action='store', default=50,                 help='Iterations to train')

    parser.add_argument('--host_ip',    dest='host_ip',                 action='store', default='192.168.1.101',    help='IP address to robot arm (UR5E)')

    # Run main program with specified arguments
    args = parser.parse_args()
    main(args)
