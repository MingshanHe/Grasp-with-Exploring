from logging import error
from turtle import pos
from urx.robot import Robot
from utils import Logger
from utils import Filter
from trainer import NeuralNetwork, Trainer
from vrep_api import vrep
import numpy as np
import socket
import time
import struct
import os

class UR5E(Robot):
    def __init__(self, host, use_rt=False, use_simulation=False, train_axis='x y'):
        """
        UR5E Class: Control the Robot
        CoppeliaSim(V-rep): vrep-api in Simulation
        urx(third party package): urx in Real World
        """
        self.use_sim = use_simulation
        self.train_axis = train_axis


        if (self.train_axis == 'x y'):
            print("[SETTING PARAM INFO]: Training in both 'X' and 'Y' axis.")
        elif (self.train_axis == 'x'):
            print("[SETTING PARAM INFO]: Training in only 'X' axis.")
        elif (self.train_axis =='y'):
            print("[SETTING PARAM INFO]: Training in only 'Y' axis.")
        else:
            error("Wrong Training Axis.")

        if self.use_sim:
            # Set up grasp params
            self.pre_grasp_high = 0.1
            self.grasp_high = 0.02

            # Setup some params
            self.workspace_limits = np.asarray([[-0.7, -0.3], [-0.2, 0.2], [-0.0001, 0.4]])

            self.home_pose = [-0.3, 0.0, 0.30, np.pi/2, 0.0, np.pi/2]

            self.put_pose  = [[-0.3, -0.3, self.pre_grasp_high, np.pi/2, 0.0, np.pi/2],
                            [-0.3, -0.3, self.grasp_high, np.pi/2, 0.0, np.pi/2]]

            self.workstart_pose = [[-0.3, 0.0, 0.1, np.pi/2, 0.0, np.pi/2],
                                [-0.7, 0.0, 0.1, np.pi/2, 0.0, np.pi/2],
                                [-0.500, -0.2, 0.1, np.pi/2, np.pi/2, np.pi/2],
                                [-0.500, 0.2, 0.1, np.pi/2, np.pi/2, np.pi/2]]

            self.explore_start_pose = [[-0.3, 0.0, 0.02, np.pi/2, 0.0, np.pi/2],
                                    [-0.7, 0.0, 0.02, np.pi/2, 0.0, np.pi/2],
                                    [-0.5, -0.2, 0.02, np.pi/2, np.pi/2, np.pi/2],
                                    [-0.5, 0.2, 0.02, np.pi/2, np.pi/2, np.pi/2]]

            self.explore_end_pose = [[-0.7, 0.0, 0.02, np.pi/2, 0.0, np.pi/2],
                                    [-0.3, 0.0, 0.02, np.pi/2, 0.0, np.pi/2],
                                    [-0.5, 0.2, 0.02, np.pi/2, np.pi/2, np.pi/2],
                                    [-0.5, -0.2, 0.02, np.pi/2, np.pi/2, np.pi/2]]



            self.detected_threshold = 3.0
            self.detect_iterations  = 4
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
            self.obj_mesh_dir = os.path.abspath('simBindings/objects/blocks')
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
                print('[ENVIRONMENT STATE]: Connected to simulation.')
                self.restart_sim()

            # Add objects to simulation environment
            self.add_objects()

            #? Initialize trainer
            # self.trainer = NeuralNetwork([2,4,3])
            self.trainer = Trainer(force_cpu=False)
            #? Initialize data logger
            logging_directory = os.path.abspath('logs')
            self.datalogger = Logger(logging_directory)

            #? Initialize filter
            self.forceFilter = Filter()
            self.torqueFilter = Filter()

            self.force_data = []
            self.torque_data = []
            self.Detected = False
            self.Check    = None

            # grasp_pose = grasp_predict_pose + current_pose
            self.grasp_predict_pose = None
            self.grasp_pose = [0.0, 0.0, 0.0]
            self.grasp_param = 0.1
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
            logging_directory = os.path.abspath('logs')
            self.datalogger = Logger(logging_directory)
            self.forceFilter = Filter()
            self.torqueFilter = Filter()
            self.nn = NeuralNetwork()

            self.tcp_Force = None
            self.tcp_Velocity = None
            self.Detected = False
            time.sleep(0.2)

    def add_objects(self):
        """
        Add random object automously
        Only in Simulation
        """
        if self.use_sim:
            # Add each object to robot workspace at x,y location and orientation (random or pre-loaded)
            self.object_handles = []
            sim_obj_handles = []
            for object_idx in range(len(self.obj_mesh_ind)):
                curr_mesh_file = os.path.join(self.obj_mesh_dir, self.mesh_list[self.obj_mesh_ind[object_idx]])

                curr_shape_name =  'shape_%02d' % object_idx
                drop_x = (self.workspace_limits[0][1] - self.workspace_limits[0][0] - 0.2) * np.random.random_sample() + self.workspace_limits[0][0] + 0.1
                drop_y = (self.workspace_limits[1][1] - self.workspace_limits[1][0] - 0.2) * np.random.random_sample() + self.workspace_limits[1][0] + 0.1
                #? Drop in Random position and orientation
                # object_position = [drop_x, drop_y, 0.15]
                # object_orientation = [2*np.pi*np.random.random_sample(), 2*np.pi*np.random.random_sample(), 2*np.pi*np.random.random_sample()]
                #? Drop in Fixed position and orientation
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
        """
        Restart the simulation
        """
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
                vrep.simxSetObjectOrientation(self.sim_client, self.UR5_target_handle, -1, (UR5_target_orientation[0] + rotate_step[0]*min(step_iter,num_rotate_steps), UR5_target_orientation[1] + rotate_step[1]*min(step_iter,num_rotate_steps), UR5_target_orientation[2] + rotate_step[2]*min(step_iter,num_rotate_steps)), vrep.simx_opmode_blocking)
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
            move_direction = np.asarray([self.workstart_pose[0][0] - UR5_target_position[0], self.workstart_pose[0][1] - UR5_target_position[1], self.workstart_pose[0][2] - UR5_target_position[2]])
            move_magnitude = np.linalg.norm(move_direction)
            move_step = 0.05*move_direction/move_magnitude
            num_move_steps = max(int(np.floor((move_direction[0]+1e-5)/(move_step[0]+1e-5))),
                                int(np.floor((move_direction[1]+1e-5)/(move_step[1]+1e-5))),
                                int(np.floor((move_direction[2]+1e-5)/(move_step[2]+1e-5))))

            # Compute gripper orientation and rotation increments
            rotate_direction = np.asarray([self.workstart_pose[0][3] - UR5_target_orientation[0], self.workstart_pose[0][4] - UR5_target_orientation[1], self.workstart_pose[0][5] - UR5_target_orientation[2]])
            rotate_magnitude = np.linalg.norm(rotate_direction)
            rotate_step = 0.05*rotate_direction/rotate_magnitude
            num_rotate_steps = max(int(np.floor((rotate_direction[0]+1e-5)/(rotate_step[0]+1e-5))),
                                int(np.floor((rotate_direction[1]+1e-5)/(rotate_step[1]+1e-5))),
                                int(np.floor((rotate_direction[2]+1e-5)/(rotate_step[2]+1e-5))))

            # Simultaneously move and rotate gripper
            for step_iter in range(max(num_move_steps, num_rotate_steps)):
                vrep.simxSetObjectPosition(self.sim_client,self.UR5_target_handle,-1,(UR5_target_position[0] + move_step[0]*min(step_iter,num_move_steps), UR5_target_position[1] + move_step[1]*min(step_iter,num_move_steps), UR5_target_position[2] + move_step[2]*min(step_iter,num_move_steps)),vrep.simx_opmode_blocking)
                vrep.simxSetObjectOrientation(self.sim_client, self.UR5_target_handle, -1, (UR5_target_orientation[0] + move_step[0]*min(step_iter,num_rotate_steps), UR5_target_orientation[1] + move_step[1]*min(step_iter,num_rotate_steps), UR5_target_orientation[2] + move_step[2]*min(step_iter,num_rotate_steps)), vrep.simx_opmode_blocking)
            vrep.simxSetObjectPosition(self.sim_client,self.UR5_target_handle,-1,(self.workstart_pose[0][0],self.workstart_pose[0][1],self.workstart_pose[0][2]),vrep.simx_opmode_blocking)
            vrep.simxSetObjectOrientation(self.sim_client, self.UR5_target_handle, -1, (self.workstart_pose[0][3],self.workstart_pose[0][4],self.workstart_pose[0][5]), vrep.simx_opmode_blocking)
            time.sleep(1)
        else:
            Robot.movel(self, self.workstart_pose, acc=0.01, vel=0.05)
            time.sleep(1)

    def Go(self, pose):
        """
        Let the Robot move to
        the input pose data
        """
        if self.use_sim:

            # sim_ret, UR5_target_handle = vrep.simxGetObjectHandle(self.sim_client,'UR5_target',vrep.simx_opmode_blocking)
            sim_ret, UR5_target_position = vrep.simxGetObjectPosition(self.sim_client, self.UR5_target_handle,-1,vrep.simx_opmode_blocking)
            sim_ret, UR5_target_orientation = vrep.simxGetObjectOrientation(self.sim_client, self.UR5_target_handle, -1, vrep.simx_opmode_blocking)

            # Compute gripper position and linear movement increments
            move_direction = np.asarray([pose[0] - UR5_target_position[0], pose[1] - UR5_target_position[1], pose[2] - UR5_target_position[2]])
            move_magnitude = np.linalg.norm(move_direction)
            move_step = 0.01*move_direction/move_magnitude
            num_move_steps = max(int(np.floor((move_direction[0]+1e-10)/(move_step[0]+1e-10))),
                                int(np.floor((move_direction[1]+1e-10)/(move_step[1]+1e-10))),
                                int(np.floor((move_direction[2]+1e-10)/(move_step[2]+1e-10))))

            # Compute gripper orientation and rotation increments
            rotate_direction = np.asarray([pose[3] - UR5_target_orientation[0], pose[4] - UR5_target_orientation[1], pose[5] - UR5_target_orientation[2]])
            rotate_magnitude = np.linalg.norm(rotate_direction)
            rotate_step = 0.05*rotate_direction/rotate_magnitude
            num_rotate_steps = max(int(np.floor((rotate_direction[0])/(rotate_step[0]+1e-10))),
                                int(np.floor((rotate_direction[1])/(rotate_step[1]+1e-10))),
                                int(np.floor((rotate_direction[2])/(rotate_step[2]+1e-10))))

            # Simultaneously move and rotate gripper
            for step_iter in range(num_move_steps):
                vrep.simxSetObjectPosition(self.sim_client,self.UR5_target_handle,-1,(UR5_target_position[0] + move_step[0]*min(step_iter,num_move_steps), UR5_target_position[1] + move_step[1]*min(step_iter,num_move_steps), UR5_target_position[2] + move_step[2]*min(step_iter,num_move_steps)),vrep.simx_opmode_blocking)
            vrep.simxSetObjectPosition(self.sim_client,self.UR5_target_handle,-1,(pose[0],pose[1],pose[2]),vrep.simx_opmode_blocking)
            for step_iter in range(num_rotate_steps):
                vrep.simxSetObjectOrientation(self.sim_client, self.UR5_target_handle, -1, (pose[3], UR5_target_orientation[1] + rotate_step[1]*min(step_iter,num_rotate_steps), pose[5]), vrep.simx_opmode_blocking)
            vrep.simxSetObjectOrientation(self.sim_client, self.UR5_target_handle, -1, (pose[3],pose[4],pose[5]), vrep.simx_opmode_blocking)
            time.sleep(1)
        else:
            Robot.movel(self, pose, acc=0.01, vel=0.05)
            time.sleep(2)
            self.gripper_close()
            time.sleep(2)

    def DetectObject(self):
        """
        Check the tcp_force and
        return if detect the object
        """
        if self.use_sim:
            sim_ret,state,forceVector,torqueVector = vrep.simxReadForceSensor(self.sim_client,self.Sensor_handle,vrep.simx_opmode_streaming)

            forceVector = self.forceFilter.LowPassFilter(forceVector)
            torqueVector = self.torqueFilter.LowPassFilter(torqueVector)
            # Output the force of XYZ
            if((np.fabs(forceVector[0]) < self.detected_threshold)&(np.fabs(forceVector[1]) < self.detected_threshold)):
                self.force_data = forceVector
                return True
            else:
                self.forceFilter.LowPassFilterClear()
                self.datalogger.save_force_data(forceVector)
                self.Detected = True
                return False
        else:
            self.tcp_Force = self.Monitor.tcf_force()
            self.tcp_Force = self.forceFilter.LowPassFilter(self.tcp_Force)

            # self.tcp_Velocity = self.Monitor.tcp_Velocity
            if((np.fabs(self.tcp_Force[0]) < 1.0)|(np.fabs(self.tcp_Force[1]) < 1.0)):
                return True
            else:
                self.datalogger.save_force_data(self.tcp_Force)
                self.Detected = True
                return False

    def Explore(self):
        """
        Expore and Grasp
        """
        if self.use_sim:
            for i in range(self.detect_iterations):
                self. Go(self.explore_start_pose[i])

                # Pre: close the gripper
                self.gripper_close()
                time.sleep(1)

                # Get Current end state
                sim_ret, UR5_target_position = vrep.simxGetObjectPosition(self.sim_client, self.UR5_target_handle,-1,vrep.simx_opmode_blocking)
                sim_ret, UR5_target_orientation = vrep.simxGetObjectOrientation(self.sim_client, self.UR5_target_handle, -1, vrep.simx_opmode_blocking)

                # Compute gripper position and linear movement increments
                move_direction = np.asarray([self.explore_end_pose[i][0] - UR5_target_position[0], self.explore_end_pose[i][1] - UR5_target_position[1], self.explore_end_pose[i][2] - UR5_target_position[2]])
                move_magnitude = np.linalg.norm(move_direction)
                move_step = 0.005*move_direction/move_magnitude
                num_move_steps = max(int(np.floor((move_direction[0]+1e-5)/(move_step[0]+1e-5))),
                                    int(np.floor((move_direction[1]+1e-5)/(move_step[1]+1e-5))),
                                    int(np.floor((move_direction[2]+1e-5)/(move_step[2]+1e-5))))

                # Compute gripper orientation and rotation increments
                rotate_direction = np.asarray([self.explore_end_pose[i][3] - UR5_target_orientation[0], self.explore_end_pose[i][4] - UR5_target_orientation[1], self.explore_end_pose[i][5] - UR5_target_orientation[2]])
                rotate_magnitude = np.linalg.norm(rotate_direction)
                rotate_step = 0.05*rotate_direction/rotate_magnitude
                num_rotate_steps = max(int(np.floor((rotate_direction[0])/(rotate_step[0]+1e-10))),
                                    int(np.floor((rotate_direction[1])/(rotate_step[1]+1e-10))),
                                    int(np.floor((rotate_direction[2])/(rotate_step[2]+1e-10))))

                # Simultaneously move and rotate gripper
                for step_iter in range(max(num_move_steps, num_rotate_steps)):
                    vrep.simxSetObjectPosition(self.sim_client,self.UR5_target_handle,-1,(UR5_target_position[0] + move_step[0]*min(step_iter,num_move_steps), UR5_target_position[1] + move_step[1]*min(step_iter,num_move_steps), UR5_target_position[2] + move_step[2]*min(step_iter,num_move_steps)),vrep.simx_opmode_blocking)
                    vrep.simxSetObjectOrientation(self.sim_client, self.UR5_target_handle, -1, (UR5_target_orientation[0] + rotate_step[0]*min(step_iter,num_rotate_steps), UR5_target_orientation[1] + rotate_step[1]*min(step_iter,num_rotate_steps), UR5_target_orientation[2] + rotate_step[2]*min(step_iter,num_rotate_steps)), vrep.simx_opmode_blocking)
                    if not self.DetectObject() :

                        # Read current pose (position & orientation)
                        sim_ret, UR5_target_position = vrep.simxGetObjectPosition(self.sim_client, self.UR5_target_handle,-1,vrep.simx_opmode_blocking)
                        sim_ret, UR5_target_orientation = vrep.simxGetObjectOrientation(self.sim_client, self.UR5_target_handle, -1, vrep.simx_opmode_blocking)

                        # Save the heatmap
                        self.heatmap = self.trainer.upate_heatmap(self.workspace_limits, (UR5_target_position[0], UR5_target_position[1]), self.force_data, UR5_target_orientation[1])
                        self.datalogger.save_heatmaps(self.heatmap)

                        # Touch something and z+ to pre grasp
                        # self.Go((UR5_target_position[0] + move_step[0]*min(step_iter,num_move_steps), UR5_target_position[1] + move_step[1]*min(step_iter,num_move_steps), self.pre_grasp_high, self.explore_start_pose[i][3], self.explore_start_pose[i][4],self.explore_start_pose[i][5]))

                        # Touch something and move to the start pose
                        self.Go(self.workstart_pose[i])
                        time.sleep(1)
                        break

                # Check the Object to Grasp
                if self.Detected:
                    self.Check = True
                    # # Read current pose (position & orientation)
                    # sim_ret, UR5_target_position = vrep.simxGetObjectPosition(self.sim_client, self.UR5_target_handle,-1,vrep.simx_opmode_blocking)
                    # sim_ret, UR5_target_orientation = vrep.simxGetObjectOrientation(self.sim_client, self.UR5_target_handle, -1, vrep.simx_opmode_blocking)

                    # # Save the heatmap
                    # self.heatmap = self.trainer.upate_heatmap(self.workspace_limits, (UR5_target_position[0], UR5_target_position[1]), self.force_data, UR5_target_orientation[1])
                    # self.datalogger.save_heatmaps(self.heatmap)

                    # Using forward to predict
                    # self.grasp_predict_pose = self.trainer.forward(np.asarray(([self.force_data[1]], [self.force_data[0]])))

                    # if (self.train_axis == 'x'):
                    #     self.grasp_predict_pose[1] = 0.0
                    # elif (self.train_axis=='y'):
                    #     self.grasp_predict_pose[0] = 0.0
                    # else:
                    #     self.grasp_predict_pose = self.grasp_predict_pose
                    # print("[PREDICT RESULT]: Trainer Predict Grasp Position: [{},{}]; Orientation: [{}]".format(self.grasp_predict_pose[0], self.grasp_predict_pose[1], self.grasp_predict_pose[2]))

                    # Read current pose (position & orientation)
                    # sim_ret, UR5_target_position = vrep.simxGetObjectPosition(self.sim_client, self.UR5_target_handle,-1,vrep.simx_opmode_blocking)
                    # sim_ret, UR5_target_orientation = vrep.simxGetObjectOrientation(self.sim_client, self.UR5_target_handle, -1, vrep.simx_opmode_blocking)

                    # Add Predict value to current pose
                    # self.grasp_pose[0] = self.grasp_param*self.grasp_predict_pose[0] + UR5_target_position[0]
                    # self.grasp_pose[1] = self.grasp_param*self.grasp_predict_pose[1] + UR5_target_position[1]
                    # self.grasp_pose[2] = (np.pi)*(self.grasp_predict_pose[2]+0.5) + UR5_target_orientation[1]
                    if ((i+1) == self.detect_iterations):
                        self.prev_heatmap = self.heatmap.copy()
                        break
                    else:
                        self.Go(self.workstart_pose[i+1])
                        continue
                    # self.Grasp(pos_data=(self.grasp_pose[0], self.grasp_pose[1]), ori_data=(np.pi/2, self.grasp_pose[2], np.pi/2))
                else:
                    print("[ENVIRONMENT STATE]: No Object to Grasp")
                    if not (self.Check):
                        self.Check = False
                    if ((i+1) == self.detect_iterations):
                        self.prev_heatmap = self.heatmap.copy()
                        break
                    else:
                        self.Go(self.workstart_pose[i+1])
                        continue
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
                    grasp_pose = (predict_pos[0],predict_pos[1],predict_pos[2], np.pi/2, 0.0, np.pi/2)
                    self.Grasp(grasp_pose)
                else:
                    print("Predicted Position is out of the workspace limit.")

    def Train(self, use_heuristic):

        grasp_predictions, state_feat = self.trainer.forward(self.heatmap, is_volatile=True)

        best_grasp_conf = np.max(grasp_predictions)
        print('[TRAINER INFO]: Primitive GRASP confidence scores: %f' % (best_grasp_conf))

        if use_heuristic:
            best_pix_ind = self.trainer.grasp_heuristic(self.heatmap)
            predicted_value = grasp_predictions[best_pix_ind]
        else:
            best_pix_ind = np.unravel_index(np.argmax(grasp_predictions), grasp_predictions.shape)
            predicted_value = np.max(grasp_predictions)
        self.prev_best_pix_ind = best_pix_ind
        print('[TRAINER INFO]: Action Grasp at (%d, %d, %d)' % (best_pix_ind[0], best_pix_ind[1], best_pix_ind[2]))
        best_rotation_angle = np.deg2rad(best_pix_ind[0]*(360.0/self.trainer.model.num_rotations))
        best_pix_x = best_pix_ind[2]
        best_pix_y = best_pix_ind[1]

        primitive_position = [best_pix_x * self.trainer.heatmap_resolution + self.workspace_limits[0][0], 
        best_pix_y * self.trainer.heatmap_resolution + self.workspace_limits[1][0]]

        self.Grasp(pos_data=(primitive_position[0], primitive_position[1]), ori_data=(np.pi/2, best_rotation_angle, np.pi/2))


    def Grasp(self, pos_data, ori_data):
        """
        Grasp Strategy
        """
        if self.use_sim:
            print("[PREDICT RESULT]: Desired Grasp Position: [{}, {}], Orientation: [{}].".format(pos_data[0], pos_data[1], ori_data[1]))
            backdata, taskcontinue = self.DesiredPositionScore(pos_data)
            if taskcontinue:
                # Open the Gripper
                self.gripper_open()
                time.sleep(1)

                # Go to the position and orientation above the object
                self.Go((pos_data[0], pos_data[1], 0.1, ori_data[0], ori_data[1], ori_data[2]))

                # Go to grasp
                self.Go((pos_data[0], pos_data[1], self.grasp_high, ori_data[0], ori_data[1], ori_data[2]))


                # self.trainer.update(np.asarray(([self.force_data[1]], [self.force_data[0]])),
                # np.asarray((self.grasp_predict_pose[0]/self.grasp_param+grasp_score[0],
                #             self.grasp_predict_pose[1]/self.grasp_param+grasp_score[1],
                #             self.grasp_predict_pose[2]+grasp_score[2])))

                # Close the Gripper
                self.gripper_close()
                

                self.Go(self.put_pose[0])

                self.Go(self.put_pose[1])

                self.gripper_open()

                # self.Explore()
                self.Check = input("Check if it is grasp: [True], [False]")

                if self.Check:
                    grasp_success = False
                    print("[IMPORTANT RESULT]: Nothing Grasped. TUT.TUT")
                else:
                    grasp_success = True
                    print("[IMPORTANT RESULT]: Nice Grasp!!! !^U^!")

                label_value, prev_reward_value = self.trainer.get_label_value(grasp_success)

                self.trainer.backprop(self.prev_heatmap, self.prev_best_pix_ind, label_value)

                time.sleep(1)
                # Pick the object up
                self.Go((pos_data[0], pos_data[1], self.pre_grasp_high, ori_data[0], ori_data[1], ori_data[2]))

            else:
                print("out of workspace limit, need to backprob to modify the params.")

        else:
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

    def DesiredPositionScore(self, data):
        """
        Score the desired position: If it is in the workspace limits
        """
        backdata = []
        task_continue = True

        if (data[0] > self.workspace_limits[0][1]):
            backdata.append(-10)
            task_continue = False
        elif (data[0] < self.workspace_limits[0][0]):
            backdata.append(+10)
            task_continue = False
        else:
            backdata.append(0)

        if (data[1] > self.workspace_limits[1][1]):
            backdata.append(-10)
            task_continue = False
        elif (data[1] < self.workspace_limits[1][0]):
            backdata.append(+10)
            task_continue = False
        else:
            backdata.append(0)
        return backdata, task_continue

    def PredictedGraspScore(self):
        """
        Score the predicted position
        """
        sim_ret,state,forceVector,torqueVector=vrep.simxReadForceSensor(self.sim_client,self.Sensor_handle,vrep.simx_opmode_streaming)
        forceVector = self.forceFilter.LowPassFilter(forceVector)
        torqueVector = self.torqueFilter.LowPassFilter(torqueVector)

        grasp_score = [0.0, 0.0, 0.0]

        if (np.fabs(forceVector[2]) > 8):
            print("[ITERATION RESULT]:  Need to change Angle")
            grasp_score[2] = np.random.rand() - 0.5
        else:
            print("[ITERATION RESULT]: Nice Grasp Angle.")
        if (torqueVector[1] < -1):
            print("[ITERATION RESULT]: Need to change position.")
            grasp_score[0] = - np.random.rand()/2
        elif (torqueVector[1] > 1):
            print("[ITERATION RESULT]: Need to change position.")
            grasp_score[0] = np.random.rand()/2
        else:
            print("[ITERATION RESULT]: Nice Grasp Position.")

        return grasp_score

    def gripper_open(self):
        """
        open the onrobot RG2 gripper
        using the defined urscript
        """
        if self.use_sim:
            ret_resp,ret_ints,ret_floats,ret_strings,ret_buffer = vrep.simxCallScriptFunction(
                self.sim_client, 'remoteApiCommandServer',vrep.sim_scripttype_childscript,
                'GripperOpen',[0], [0.0], "false", bytearray(), vrep.simx_opmode_blocking)
            time.sleep(1)
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
            ret_resp,ret_ints,ret_floats,ret_strings,ret_buffer = vrep.simxCallScriptFunction(
                self.sim_client, 'remoteApiCommandServer',vrep.sim_scripttype_childscript,
                'GripperClose',[0], [0.0], "false", bytearray(), vrep.simx_opmode_blocking)
            time.sleep(1)
        else:
            tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            tcp_socket.connect(("192.168.1.101", 30003))
            with open("urscripts/close.txt", "r") as f:
                tcp_command = f.read()
            tcp_socket.send(str.encode(tcp_command))  # 利用字符串的encode方法编码成bytes，默认为utf-8类型
            tcp_socket.close()
            time.sleep(2)