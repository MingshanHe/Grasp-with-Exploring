from urx.robot import Robot
from utils import DataLogger
from utils import Filter
from trainer import NeuralNetwork
from vrep_api import vrep
import numpy as np
import socket
import time
import struct
import os



# class Robot(object):
#     def __init__(self, obj_mesh_dir, num_obj, workspace_limits):



class UR5E(Robot):
    def __init__(self, host, use_rt=False, use_simulation=False):
        self.use_sim = use_simulation
        if self.use_sim:
            # Setup some params
            self.workspace_limits = np.asarray([[-0.724, -0.276], [-0.224, 0.224], [-0.0001, 0.4]])
            self.home_pose = [-0.276, 0.0, 0.30, np.pi/2, 0.0, np.pi/2]
            self.workstart_pose = [-0.276, 0.0, 0.04, np.pi/2, 0.0, np.pi/2]
            # Define colors for object meshes (Tableau palette)
            self.color_space = np.asarray([[78.0, 121.0, 167.0], # blue
                                            [89.0, 161.0, 79.0], # green
                                            [156, 117, 95], # brown
                                            [242, 142, 43], # orange
                                            [237.0, 201.0, 72.0], # yellow
                                            [186, 176, 172], # gray
                                            [255.0, 87.0, 89.0], # red
                                            [176, 122, 161], # purple
                                            [118, 183, 178], # cyan
                                            [255, 157, 167]])/255.0 #pink

            # Read files in object mesh directory
            self.obj_mesh_dir = 'simulation/objects/block'
            self.num_obj = 1
            self.mesh_list = os.listdir(self.obj_mesh_dir)

            # Randomly choose objects to add to scene
            self.obj_mesh_ind = np.random.randint(0, len(self.mesh_list), size=self.num_obj)
            self.obj_mesh_color = self.color_space[np.asarray(range(10)), :]

            # Make sure to have the server side running in V-REP:
            # in a child script of a V-REP scene, add following command
            # to be executed just once, at simulation start:
            #
            # simExtRemoteApiStart(19999)
            #
            # then start simulation, and run this program.
            #
            # IMPORTANT: for each successful call to simxStart, there
            # should be a corresponding call to simxFinish at the end!

            # MODIFY remoteApiConnections.txt

            # Connect to simulator
            vrep.simxFinish(-1) # Just in case, close all opened connections
            self.sim_client = vrep.simxStart('127.0.0.1', 19997, True, True, 5000, 5) # Connect to V-REP on port 19997
            if self.sim_client == -1:
                print('Failed to connect to simulation (V-REP remote API server). Exiting.')
                exit()
            else:
                print('Connected to simulation.')
                self.restart_sim()

            # Add objects to simulation environment
            self.add_objects()

            #? Initialize data logger
            logging_directory = os.path.abspath('logs')
            self.logger = Logger(logging_directory)

            #? Initialize filter
            self.filter = Filter()

            self.force_data = []
            self.torque_data = []
        else:
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

    def add_objects(self):
        if self.use_sim:
            # Add each object to robot workspace at x,y location and orientation (random or pre-loaded)
            self.object_handles = []
            sim_obj_handles = []
            for object_idx in range(len(self.obj_mesh_ind)):
                curr_mesh_file = os.path.join(self.obj_mesh_dir, self.mesh_list[self.obj_mesh_ind[object_idx]])

                curr_shape_name = 'block'
                drop_x = (self.workspace_limits[0][1] - self.workspace_limits[0][0] - 0.2) * np.random.random_sample() + self.workspace_limits[0][0] + 0.1
                drop_y = (self.workspace_limits[1][1] - self.workspace_limits[1][0] - 0.2) * np.random.random_sample() + self.workspace_limits[1][0] + 0.1
                # object_position = [drop_x, drop_y, 0.15]
                # object_orientation = [2*np.pi*np.random.random_sample(), 2*np.pi*np.random.random_sample(), 2*np.pi*np.random.random_sample()]
                object_position = [-0.5, 0, 0.15]
                object_orientation = [np.pi/2, 0, 0]

                object_color = [self.obj_mesh_color[object_idx][0], self.obj_mesh_color[object_idx][1], self.obj_mesh_color[object_idx][2]]
                ret_resp,ret_ints,ret_floats,ret_strings,ret_buffer = vrep.simxCallScriptFunction(self.sim_client, 'remoteApiCommandServer',vrep.sim_scripttype_childscript,'importShape',[0,0,255,0], object_position + object_orientation + object_color, [curr_mesh_file, curr_shape_name], bytearray(), vrep.simx_opmode_blocking)
                if ret_resp == 8:
                    print('Failed to add new objects to simulation. Please restart.')
                    exit()
                curr_shape_handle = ret_ints[0]
                self.object_handles.append(curr_shape_handle)
            time.sleep(2)
    def restart_sim(self):
        if self.use_sim:
            sim_ret, self.UR5_target_handle = vrep.simxGetObjectHandle(self.sim_client,'UR5_target',vrep.simx_opmode_blocking)
            sim_ret, self.Sensor_handle = vrep.simxGetObjectHandle(self.sim_client, 'UR5_connection', vrep.simx_opmode_blocking)
            vrep.simxSetObjectPosition(self.sim_client, self.UR5_target_handle, -1, (-0.5,0,0.3), vrep.simx_opmode_blocking)
            vrep.simxStopSimulation(self.sim_client, vrep.simx_opmode_blocking)
            vrep.simxStartSimulation(self.sim_client, vrep.simx_opmode_blocking)
            time.sleep(1)
            sim_ret, self.RG2_tip_handle = vrep.simxGetObjectHandle(self.sim_client, 'UR5_tip', vrep.simx_opmode_blocking)
            sim_ret, gripper_position = vrep.simxGetObjectPosition(self.sim_client, self.RG2_tip_handle, -1, vrep.simx_opmode_blocking)
            while gripper_position[2] > 0.4: # V-REP bug requiring multiple starts and stops to restart
                vrep.simxStopSimulation(self.sim_client, vrep.simx_opmode_blocking)
                vrep.simxStartSimulation(self.sim_client, vrep.simx_opmode_blocking)
                time.sleep(1)
                sim_ret, gripper_position = vrep.simxGetObjectPosition(self.sim_client, self.RG2_tip_handle, -1, vrep.simx_opmode_blocking)
    def GoHome(self):
        """
        Let the Robot move to
        the defined home pose
        """
        if self.use_sim:
            # sim_ret, UR5_target_handle = vrep.simxGetObjectHandle(self.sim_client,'UR5_target',vrep.simx_opmode_blocking)
            sim_ret, UR5_target_position = vrep.simxGetObjectPosition(self.sim_client, self.UR5_target_handle,-1,vrep.simx_opmode_blocking)
            sim_ret, UR5_target_orientation = vrep.simxGetObjectOrientation(self.sim_client, self.UR5_target_handle, -1, vrep.simx_opmode_blocking)

            # Compute gripper position and linear movement increments
            move_direction = np.asarray([self.home_pose[0] - UR5_target_position[0], self.home_pose[1] - UR5_target_position[1], self.home_pose[2] - UR5_target_position[2]])
            move_magnitude = np.linalg.norm(move_direction)
            move_step = 0.05*move_direction/move_magnitude
            num_move_steps = max(int(np.floor((move_direction[0]+1e-5)/(move_step[0]+1e-5))),
                                int(np.floor((move_direction[1]+1e-5)/(move_step[1]+1e-5))),
                                int(np.floor((move_direction[2]+1e-5)/(move_step[2]+1e-5))))

            # Compute gripper orientation and rotation increments
            rotate_direction = np.asarray([self.home_pose[3] - UR5_target_orientation[0], self.home_pose[4] - UR5_target_orientation[1], self.home_pose[5] - UR5_target_orientation[2]])
            rotate_magnitude = np.linalg.norm(rotate_direction)
            rotate_step = 0.05*rotate_direction/rotate_magnitude
            num_rotate_steps = max(int(np.floor((rotate_direction[0]+1e-5)/(rotate_step[0]+1e-5))),
                                int(np.floor((rotate_direction[1]+1e-5)/(rotate_step[1]+1e-5))),
                                int(np.floor((rotate_direction[2]+1e-5)/(rotate_step[2]+1e-5))))

            # Simultaneously move and rotate gripper
            for step_iter in range(max(num_move_steps, num_rotate_steps)):
                vrep.simxSetObjectPosition(self.sim_client,self.UR5_target_handle,-1,(UR5_target_position[0] + move_step[0]*min(step_iter,num_move_steps), UR5_target_position[1] + move_step[1]*min(step_iter,num_move_steps), UR5_target_position[2] + move_step[2]*min(step_iter,num_move_steps)),vrep.simx_opmode_blocking)
                vrep.simxSetObjectOrientation(self.sim_client, self.UR5_target_handle, -1, (UR5_target_orientation[0] + move_step[0]*min(step_iter,num_rotate_steps), UR5_target_orientation[1] + move_step[1]*min(step_iter,num_rotate_steps), UR5_target_orientation[2] + move_step[2]*min(step_iter,num_rotate_steps)), vrep.simx_opmode_blocking)
            vrep.simxSetObjectPosition(self.sim_client,self.UR5_target_handle,-1,(self.home_pose[0],self.home_pose[1],self.home_pose[2]),vrep.simx_opmode_blocking)
            vrep.simxSetObjectOrientation(self.sim_client, self.UR5_target_handle, -1, (self.home_pose[3],self.home_pose[4],self.home_pose[5]), vrep.simx_opmode_blocking)
            time.sleep(1)
        else:
            Robot.movel(self, self.home_pose, acc=0.02, vel=0.1)
            time.sleep(1)

    def GoWork(self):
        """
        Let the Robot move to
        the start pose of work
        """
        if self.use_sim:
            # sim_ret, UR5_target_handle = vrep.simxGetObjectHandle(self.sim_client,'UR5_target',vrep.simx_opmode_blocking)
            sim_ret, UR5_target_position = vrep.simxGetObjectPosition(self.sim_client, self.UR5_target_handle,-1,vrep.simx_opmode_blocking)
            sim_ret, UR5_target_orientation = vrep.simxGetObjectOrientation(self.sim_client, self.UR5_target_handle, -1, vrep.simx_opmode_blocking)

            # Compute gripper position and linear movement increments
            move_direction = np.asarray([self.workstart_pose[0] - UR5_target_position[0], self.workstart_pose[1] - UR5_target_position[1], self.workstart_pose[2] - UR5_target_position[2]])
            move_magnitude = np.linalg.norm(move_direction)
            move_step = 0.05*move_direction/move_magnitude
            num_move_steps = max(int(np.floor((move_direction[0]+1e-5)/(move_step[0]+1e-5))),
                                int(np.floor((move_direction[1]+1e-5)/(move_step[1]+1e-5))),
                                int(np.floor((move_direction[2]+1e-5)/(move_step[2]+1e-5))))

            # Compute gripper orientation and rotation increments
            rotate_direction = np.asarray([self.workstart_pose[3] - UR5_target_orientation[0], self.workstart_pose[4] - UR5_target_orientation[1], self.workstart_pose[5] - UR5_target_orientation[2]])
            rotate_magnitude = np.linalg.norm(rotate_direction)
            rotate_step = 0.05*rotate_direction/rotate_magnitude
            num_rotate_steps = max(int(np.floor((rotate_direction[0]+1e-5)/(rotate_step[0]+1e-5))),
                                int(np.floor((rotate_direction[1]+1e-5)/(rotate_step[1]+1e-5))),
                                int(np.floor((rotate_direction[2]+1e-5)/(rotate_step[2]+1e-5))))

            # Simultaneously move and rotate gripper
            for step_iter in range(max(num_move_steps, num_rotate_steps)):
                vrep.simxSetObjectPosition(self.sim_client,self.UR5_target_handle,-1,(UR5_target_position[0] + move_step[0]*min(step_iter,num_move_steps), UR5_target_position[1] + move_step[1]*min(step_iter,num_move_steps), UR5_target_position[2] + move_step[2]*min(step_iter,num_move_steps)),vrep.simx_opmode_blocking)
                vrep.simxSetObjectOrientation(self.sim_client, self.UR5_target_handle, -1, (UR5_target_orientation[0] + move_step[0]*min(step_iter,num_rotate_steps), UR5_target_orientation[1] + move_step[1]*min(step_iter,num_rotate_steps), UR5_target_orientation[2] + move_step[2]*min(step_iter,num_rotate_steps)), vrep.simx_opmode_blocking)
            vrep.simxSetObjectPosition(self.sim_client,self.UR5_target_handle,-1,(self.workstart_pose[0],self.workstart_pose[1],self.workstart_pose[2]),vrep.simx_opmode_blocking)
            vrep.simxSetObjectOrientation(self.sim_client, self.UR5_target_handle, -1, (self.workstart_pose[3],self.workstart_pose[4],self.workstart_pose[5]), vrep.simx_opmode_blocking)
            time.sleep(1)
        else:
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

    def Explore(self,vel=0.02):
        """
        Expore and Grasp
        """
        if self.use_sim:
            sim_ret, UR5_target_position = vrep.simxGetObjectPosition(self.sim_client, self.UR5_target_handle,-1,vrep.simx_opmode_blocking)
            sim_ret, UR5_target_orientation = vrep.simxGetObjectOrientation(self.sim_client, self.UR5_target_handle, -1, vrep.simx_opmode_blocking)

            # Compute gripper position and linear movement increments
            move_direction = np.asarray([target_pose[0] - UR5_target_position[0], target_pose[1] - UR5_target_position[1], target_pose[2] - UR5_target_position[2]])
            move_magnitude = np.linalg.norm(move_direction)
            move_step = vel*move_direction/move_magnitude
            num_move_steps = max(int(np.floor((move_direction[0]+1e-5)/(move_step[0]+1e-5))),
                                int(np.floor((move_direction[1]+1e-5)/(move_step[1]+1e-5))),
                                int(np.floor((move_direction[2]+1e-5)/(move_step[2]+1e-5))))

            # Compute gripper orientation and rotation increments
            rotate_direction = np.asarray([target_pose[3] - UR5_target_orientation[0], target_pose[4] - UR5_target_orientation[1], target_pose[5] - UR5_target_orientation[2]])
            rotate_magnitude = np.linalg.norm(rotate_direction)
            rotate_step = 0.05*rotate_direction/rotate_magnitude
            num_rotate_steps = max(int(np.floor((rotate_direction[0]+1e-5)/(rotate_step[0]+1e-5))),
                                int(np.floor((rotate_direction[1]+1e-5)/(rotate_step[1]+1e-5))),
                                int(np.floor((rotate_direction[2]+1e-5)/(rotate_step[2]+1e-5))))

            # Simultaneously move and rotate gripper
            for step_iter in range(max(num_move_steps, num_rotate_steps)):
                vrep.simxSetObjectPosition(self.sim_client,self.UR5_target_handle,-1,(UR5_target_position[0] + move_step[0]*min(step_iter,num_move_steps), UR5_target_position[1] + move_step[1]*min(step_iter,num_move_steps), UR5_target_position[2] + move_step[2]*min(step_iter,num_move_steps)),vrep.simx_opmode_blocking)
                vrep.simxSetObjectOrientation(self.sim_client, self.UR5_target_handle, -1, (UR5_target_orientation[0] + move_step[0]*min(step_iter,num_rotate_steps), UR5_target_orientation[1] + move_step[1]*min(step_iter,num_rotate_steps), UR5_target_orientation[2] + move_step[2]*min(step_iter,num_rotate_steps)), vrep.simx_opmode_blocking)
                sim_ret,state,forceVector,torqueVector=vrep.simxReadForceSensor(self.sim_client,self.Sensor_handle,vrep.simx_opmode_streaming)
                forceVector = self.filter.LowPassFilter(forceVector)
                # Output the force of XYZ
                self.force_data.append(forceVector)
                print(forceVector)
                # Output the torque of XYZ
                # print(torqueVector)
            self.logger.save_force_data(self.force_data)
            vrep.simxSetObjectPosition(self.sim_client,self.UR5_target_handle,-1,(target_pose[0],target_pose[1],target_pose[2]),vrep.simx_opmode_blocking)
            vrep.simxSetObjectOrientation(self.sim_client, self.UR5_target_handle, -1, (target_pose[3],target_pose[4],target_pose[5]), vrep.simx_opmode_blocking)
            time.sleep(1)
        else:
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
        if self.use_sim:
            gripper_motor_velocity = 0.5
            gripper_motor_force = 20
            sim_ret, RG2_gripper_handle = vrep.simxGetObjectHandle(self.sim_client, 'RG2_openCloseJoint', vrep.simx_opmode_blocking)
            sim_ret, gripper_joint_position = vrep.simxGetJointPosition(self.sim_client, RG2_gripper_handle, vrep.simx_opmode_blocking)
            vrep.simxSetJointForce(self.sim_client, RG2_gripper_handle, gripper_motor_force, vrep.simx_opmode_blocking)
            vrep.simxSetJointTargetVelocity(self.sim_client, RG2_gripper_handle, gripper_motor_velocity, vrep.simx_opmode_blocking)
            while gripper_joint_position < 0.03: # Block until gripper is fully open
                sim_ret, gripper_joint_position = vrep.simxGetJointPosition(self.sim_client, RG2_gripper_handle, vrep.simx_opmode_blocking)
        else:
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
        if self.use_sim:
            gripper_motor_velocity = -0.5
            gripper_motor_force = 100
            sim_ret, RG2_gripper_handle = vrep.simxGetObjectHandle(self.sim_client, 'RG2_openCloseJoint', vrep.simx_opmode_blocking)
            sim_ret, gripper_joint_position = vrep.simxGetJointPosition(self.sim_client, RG2_gripper_handle, vrep.simx_opmode_blocking)
            vrep.simxSetJointForce(self.sim_client, RG2_gripper_handle, gripper_motor_force, vrep.simx_opmode_blocking)
            vrep.simxSetJointTargetVelocity(self.sim_client, RG2_gripper_handle, gripper_motor_velocity, vrep.simx_opmode_blocking)
            gripper_fully_closed = False
            while gripper_joint_position > -0.045: # Block until gripper is fully closed
                sim_ret, new_gripper_joint_position = vrep.simxGetJointPosition(self.sim_client, RG2_gripper_handle, vrep.simx_opmode_blocking)
                # print(gripper_joint_position)
                if new_gripper_joint_position >= gripper_joint_position:
                    return gripper_fully_closed
                gripper_joint_position = new_gripper_joint_position
            gripper_fully_closed = True
        else:
            tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            tcp_socket.connect(("192.168.1.101", 30003))
            with open("urscripts/close.txt", "r") as f:
                tcp_command = f.read()
            tcp_socket.send(str.encode(tcp_command))  # 利用字符串的encode方法编码成bytes，默认为utf-8类型
            tcp_socket.close()
            time.sleep(2)