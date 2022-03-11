from urx.robot import Robot
from utils import DataLogger
from utils import Filter
from trainer import NeuralNetwork
import numpy as np
import socket
import time

class UR5E(Robot):
    def __init__(self, host, use_rt=False, use_simulation=False):
        self.gripper_close()
        Robot.__init__(self, host, use_rt, use_simulation)
        #--------Config Setup--------#
        self.home_pose = [0.0, -0.3, 0.3, np.pi, 0.0, 0.0] # Joint Space

        self.workstart_pose = [0.0, -0.3, 0.135, np.pi, 0.0, 0.0]

        self.grasp_z = 0.135
        self.workspace_limits = [-0.15, 0.15, -0.45, -0.3, 0.1, 0.2]
        Robot.set_tcp(self, (0, 0, 0.1, 0, 0, 0))
        Robot.set_payload(self, 2, (0, 0, 0.1))

        self.Monitor = Robot.get_realtime_monitor(self)

        # Create Class
        self.datalogger = DataLogger()
        self.filter = Filter()
        self.nn = NeuralNetwork()

        self.tcp_Force = None
        self.tcp_Velocity = None
        self.Detected = False
        time.sleep(0.2)

    def GoHome(self):
        """
        Let the Robot move to
        the defined home pose
        """
        Robot.movel(self, self.home_pose, acc=0.02, vel=0.1)
        time.sleep(1)

    def GoWork(self):
        """
        Let the Robot move to
        the start pose of work
        """
        Robot.movel(self, self.workstart_pose, acc=0.01, vel=0.05)
        time.sleep(1)

    def Go(self, pose):
        """
        Let the Robot move to
        the input pose data
        """
        Robot.movel(self, pose, acc=0.01, vel=0.05)
        time.sleep(2)
        self.gripper_close()
        time.sleep(2)

    def DetectObject(self):
        """
        Check the tcp_force and
        return if detect the object
        """
        self.tcp_Force = self.Monitor.tcf_force()
        self.tcp_Force = self.filter.LowPassFilter(self.tcp_Force)

        # self.tcp_Velocity = self.Monitor.tcp_Velocity
        if((np.fabs(self.tcp_Force[0]) < 1.0)|(np.fabs(self.tcp_Force[1]) < 1.0)):
            return True
        else:
            self.datalogger.save_tcp_force(self.tcp_Force)
            self.Detected = True
            return False

    def Explore(self):
        """
        Expore and Grasp
        """
        while (self.DetectObject()):
            # Force = self.Monitor.tcf_force()
            Robot.speedl_tool(self, (0, 0.01, 0, 0, 0, 0),0.01,0.15)

        if self.Detected:
            pos = self.nn.forward(self.tcp_Force)
            predict_pos = Robot.getl(self)
            predict_pos[0] = predict_pos[0] - pos[0]
            predict_pos[1] = predict_pos[1] - pos[1]

            if((predict_pos[0]>self.workspace_limits[0]) & (predict_pos[0]<self.workspace_limits[1]) & (predict_pos[1]>self.workspace_limits[2]) & (predict_pos[1]<self.workspace_limits[3])):
                self.Grasp(pos, np.pi/2)
            else:
                print("Predicted Position is out of the workspace limit.")


    def Grasp(self, pos_data, ori_data):
        """
        Grasp Strategy
        """

        Robot.back(self, 0.2, acc=0.02, vel=0.1)
        time.sleep(1)

        trans = Robot.get_pose(self)  # get current transformation matrix (tool to base)
        trans.pos.x -= pos_data[0]
        trans.pos.y -= pos_data[1]
        trans.orient.rotate_zb(ori_data)
        Robot.set_pose(self,trans, acc=0.1, vel=0.1)  # apply the new pose
        time.sleep(1)

        self.gripper_open()
        time.sleep(1)

        Robot.back(self,-0.2, acc=0.01, vel=0.05)
        time.sleep(2)

        self.gripper_close()
        time.sleep(1)

        Robot.back(self, 0.2, acc=0.02, vel=0.1)

    def gripper_open(self):
        """
        open the onrobot RG2 gripper
        using the defined urscript
        """
        tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        tcp_socket.connect(("192.168.1.101", 30003))
        with open("urscripts/open.txt", "r") as f:
            tcp_command = f.read()
        tcp_socket.send(str.encode(tcp_command))  # 利用字符串的encode方法编码成bytes，默认为utf-8类型
        tcp_socket.close()
        time.sleep(2)

    def gripper_close(self):
        """
        close the onrobot RG2 gripper
        using the defined urscript
        """
        tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        tcp_socket.connect(("192.168.1.101", 30003))
        with open("urscripts/close.txt", "r") as f:
            tcp_command = f.read()
        tcp_socket.send(str.encode(tcp_command))  # 利用字符串的encode方法编码成bytes，默认为utf-8类型
        tcp_socket.close()
        time.sleep(2)