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
    iterations = args.iterations

    print("[SETTING PARAM INFO]: Total Training Iterations: %d" % (iterations))
    for i in range(iterations):
        print("[PARAM INFO]: %s iteration: %d" % ('Training', i))
        #? Initialize pick-and-place system (camera and robot)
        ur5e = UR5E()

        # Test:----------------------------------------------------
        # ur5e.Go((-0.276, 0.0, 0.04, np.pi/2, np.pi/2, np.pi/2))
        # ur5e.Go((-0.276, 0.0, 0.04, np.pi/2, np.pi/3, np.pi/2))
        # ur5e.Go((-0.276, 0.0, 0.04, np.pi/2, -np.pi/3, np.pi/2))
        # ---------------------------------------------------------

        ur5e.GoHome()

        ur5e.GoWork()

        ur5e.Explore()

        ur5e.stop_sim()

        print("--------------------------------------------------")

if __name__ == '__main__':

    # Parse arguments
    parser = argparse.ArgumentParser(description='Train robotic agents to learn how to plan complementary pushing and grasping actions for manipulation with deep reinforcement learning in PyTorch.')

    # --------------- Setup options ---------------
    parser.add_argument('--iterations', dest='iterations', type=int,    action='store', default=50,                 help='Iterations to train')

    # Run main program with specified arguments
    args = parser.parse_args()

    main(args)
