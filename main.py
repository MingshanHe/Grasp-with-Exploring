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
    if use_sim:
        ur5e = UR5E(host_ip,use_simulation=True)
    else:
        ur5e = UR5E(host_ip,use_simulation=False)

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
    parser.add_argument('--use_sim', dest='use_sim', action='store_true', default=False,      help='run in simulation?')
    parser.add_argument('--host_ip', dest='host_ip', action='store', default='192.168.1.101', help='IP address to robot arm (UR5E)')

    # Run main program with specified arguments
    args = parser.parse_args()
    main(args)