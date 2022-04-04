from logging import error
from turtle import pos
from urx.robot import Robot
from utils import Logger
from utils import Filter
from trainer import NeuralNetwork, Trainer
from graph  import Graph
from explore import FrontierSearch
from model import QLearningTable

from vrep_api import vrep
import numpy as np
import socket
import time
import struct
import os


class UR5E(Robot):
    def __init__(self):
        """
        UR5E Class: Control the Robot
        CoppeliaSim(V-rep): vrep-api in Simulation
        urx(third party package): urx in Real World
        """

        # Set up grasp params
        self.pre_grasp_high = 0.1
        self.grasp_high = 0.02

        # Setup some params
        self.workspace_limits = np.asarray([[-0.7, -0.3], [-0.2, 0.2], [-0.0001, 0.4]])

        self.home_pose = [-0.3, 0.0, 0.30, 0.0, 0.0, 0.0]

        self.put_pose  = [[-0.3, -0.3, self.pre_grasp_high, 0.0, 0.0, np.pi/2],
                        [-0.3, -0.3, self.grasp_high, 0.0, 0.0, np.pi/2]]

        self.workstart_pose = [[-0.3, 0.0, 0.1, 0.0, 0.0, 0.0],
                            [-0.7,    0.0, 0.1, 0.0, 0.0, 0.0],
                            [-0.500, -0.2, 0.1, 0.0, 0.0, 0.0],
                            [-0.500,  0.2, 0.1, 0.0, 0.0, 0.0]]

        self.explore_start_pose = [[-0.3, 0.0, self.grasp_high, 0.0, 0.0, 0.0],
                            [-0.7,    0.0, self.grasp_high, 0.0, 0.0, 0.0],
                            [-0.500, -0.2, self.grasp_high, 0.0, 0.0, np.pi/3],
                            [-0.500,  0.2, self.grasp_high, 0.0, 0.0, np.pi/3]]

        self.explore_end_pose = [[-0.7, 0.0, self.grasp_high, 0.0, 0.0, 0.0],
                            [-0.3,    0.0, self.grasp_high, 0.0, 0.0, 0.0],
                            [-0.500,  0.2, self.grasp_high, 0.0, 0.0, np.pi/3],
                            [-0.500, -0.2, self.grasp_high, 0.0, 0.0, np.pi/3]]

        self.detected_threshold = 2.0
        self.detect_iterations  = 5000
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
        self.unit = 0.02
        self.resolutions = 600
        self.trainer = Trainer(force_cpu=False)
        self.graph = Graph()
        self.frontierSearch = FrontierSearch(self.workspace_limits, self.resolutions)
        self.RL = QLearningTable(actions=list(range(self.frontierSearch.n_actions)))


        #? Initialize data logger
        logging_directory = os.path.abspath('logs')
        self.datalogger = Logger(logging_directory)

        #? Initialize filter
        self.forceFilter = Filter()
        self.torqueFilter = Filter()

        self.force_data = []
        self.torque_data = []
        self.Detected = False
        self.Detect_num = 0
        self.Check    = None

        # grasp_pose = grasp_predict_pose + current_pose
        self.grasp_predict_pose = None
        self.grasp_pose = [0.0, 0.0, 0.0]
        self.grasp_param = 0.1


    def add_objects(self):
        """
        Add random object automously
        Only in Simulation
        """

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
            object_position = [-0.5, 0, 0.2]
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

    def Go(self, pose):
        """
        Let the Robot move to
        the input pose data
        """
        sim_ret, UR5_target_position = vrep.simxGetObjectPosition(self.sim_client, self.UR5_target_handle,-1,vrep.simx_opmode_blocking)
        sim_ret, UR5_target_orientation = vrep.simxGetObjectOrientation(self.sim_client, self.UR5_target_handle, -1, vrep.simx_opmode_blocking)

        # Compute gripper position and linear movement increments
        move_direction = np.asarray([pose[0] - UR5_target_position[0], pose[1] - UR5_target_position[1], pose[2] - UR5_target_position[2]])
        move_magnitude = np.linalg.norm(move_direction)
        move_step = 0.01*move_direction/move_magnitude
        num_move_steps = max(int(np.floor((move_direction[0])/(move_step[0]+1e-5))),
                            int(np.floor((move_direction[1])/(move_step[1]+1e-5))),
                            int(np.floor((move_direction[2])/(move_step[2]+1e-5))))

        # Compute gripper orientation and rotation increments
        rotate_direction = np.asarray([pose[3] - UR5_target_orientation[0], pose[4] - UR5_target_orientation[1], pose[5] - UR5_target_orientation[2]])
        rotate_step = 0.05*rotate_direction
        num_rotate_steps = 20

        # Simultaneously move and rotate gripper
        for step_iter in range(num_move_steps):
            vrep.simxSetObjectPosition(self.sim_client,self.UR5_target_handle,-1,(UR5_target_position[0] + move_step[0]*min(step_iter,num_move_steps), UR5_target_position[1] + move_step[1]*min(step_iter,num_move_steps), UR5_target_position[2] + move_step[2]*min(step_iter,num_move_steps)),vrep.simx_opmode_blocking)
        vrep.simxSetObjectPosition(self.sim_client,self.UR5_target_handle,-1,(pose[0],pose[1],pose[2]),vrep.simx_opmode_blocking)
        for step_iter in range(num_rotate_steps):
            vrep.simxSetObjectOrientation(self.sim_client, self.UR5_target_handle, -1, (pose[3], UR5_target_orientation[1] + rotate_step[1]*min(step_iter,num_rotate_steps), pose[5]), vrep.simx_opmode_blocking)
        vrep.simxSetObjectOrientation(self.sim_client, self.UR5_target_handle, -1, (pose[3],pose[4],pose[5]), vrep.simx_opmode_blocking)
        time.sleep(1)

    def GoHome(self):
        """
        Let the Robot move to
        the defined home pose
        """
        self.Go(self.home_pose)

    def GoWork(self):
        """
        Let the Robot move to
        the start pose of work
        """
        self.Go(self.workstart_pose[0])

    def DetectObject(self):
        """
        Check the tcp_force and
        return if detect the object
        """

        sim_ret,state,forceVector,torqueVector = vrep.simxReadForceSensor(self.sim_client,self.Sensor_handle,vrep.simx_opmode_streaming)

        forceVector = self.forceFilter.LowPassFilter(forceVector)
        torqueVector = self.torqueFilter.LowPassFilter(torqueVector)
        # Output the force of XYZ
        if((np.fabs(forceVector[0]) > self.detected_threshold) or (np.fabs(forceVector[1]) > self.detected_threshold)):
            self.force_data = forceVector
            self.Detected = True
            self.Detect_num += 1
            return True
        else:
            # print(forceVector)
            # self.forceFilter.LowPassFilterClear()
            self.Detected = False
            return False

    def Explore(self):
        """
        Expore and Grasp
        """
        self. Go(self.explore_start_pose[0])
        for i in range(self.detect_iterations):

            # Pre: close the gripper
            self.gripper_close()
            time.sleep(1)

            # Get Current end state
            sim_ret, UR5_target_position = vrep.simxGetObjectPosition(self.sim_client, self.UR5_target_handle,-1,vrep.simx_opmode_blocking)
            sim_ret, UR5_target_orientation = vrep.simxGetObjectOrientation(self.sim_client, self.UR5_target_handle, -1, vrep.simx_opmode_blocking)

            # RL
            w2m_pos = self.frontierSearch.map.WorldToMap((UR5_target_position[0],UR5_target_position[1]))
            heatmap = self.frontierSearch.map.heatmap

            self.action = self.RL.choose_action(map_pos=w2m_pos, explore_complete=self.frontierSearch.map.explore_complete, resolutions=self.resolutions)

            move_pos = self.frontierSearch.step(action=self.action, current_pos=(UR5_target_position[0], UR5_target_position[1]), unit=self.unit)
            # Compute gripper position and linear movement increments
            move_direction = np.asarray([move_pos[0] - UR5_target_position[0], move_pos[1] - UR5_target_position[1], 0.0])
            move_magnitude = np.linalg.norm(move_direction)
            move_step = 0.0005*move_direction/(move_magnitude+1e-10)
            num_move_steps = max(int(np.floor((move_direction[0])/(move_step[0]+1e-10))),
                                int(np.floor((move_direction[1])/(move_step[1]+1e-10))),
                                int(np.floor((move_direction[2])/(move_step[2]+1e-10))))

            # Simultaneously move and rotate gripper
            for step_iter in range(num_move_steps):
                vrep.simxSetObjectPosition(self.sim_client,self.UR5_target_handle,-1,(UR5_target_position[0] + move_step[0]*min(step_iter,num_move_steps), UR5_target_position[1] + move_step[1]*min(step_iter,num_move_steps), UR5_target_position[2] + move_step[2]*min(step_iter,num_move_steps)),vrep.simx_opmode_blocking)

                # build new free heatmap
                self.frontierSearch.buildNewFree(
                    initial_cell=(UR5_target_position[0] + move_step[0]*min(step_iter,num_move_steps), UR5_target_position[1] + move_step[1]*min(step_iter,num_move_steps)),
                    initial_angle=UR5_target_orientation[2]
                )
                if self.DetectObject() :
                    # print("[ENVIRONMENT STATE]: Touch a Object")
                    self.reward = 100
                    self.RL.learn(s=w2m_pos,a=self.action,r=self.reward)
                    break

            # Check the Object to Grasp
            if self.Detected:
                sim_ret, UR5_target_position = vrep.simxGetObjectPosition(self.sim_client, self.UR5_target_handle,-1,vrep.simx_opmode_blocking)
                self.frontierSearch.buildNewFrontier(initial_cell=(UR5_target_position[0], UR5_target_position[1]),
                    initial_force=self.force_data, initial_angle=UR5_target_orientation[2])
                vrep.simxSetObjectPosition(self.sim_client,self.UR5_target_handle,-1,(UR5_target_position[0] - move_step[0]*min(step_iter,num_move_steps), UR5_target_position[1] - move_step[1]*min(step_iter,num_move_steps), UR5_target_position[2] - move_step[2]*min(step_iter,num_move_steps)),vrep.simx_opmode_blocking)
                self.datalogger.save_heatmaps(self.frontierSearch.map.heatmap)
                if self.Detect_num == 4:
                    print("[STRATEGY INFO]: Try to Grasp the object.")
                    grasp_point, grasp_angle = self.frontierSearch.grasp_point_angle()
                    self.Grasp(pos_data=grasp_point, ori_data=grasp_angle)

            else:
                vrep.simxSetObjectPosition(self.sim_client,self.UR5_target_handle,-1,(move_pos[0], move_pos[1], UR5_target_position[2]),vrep.simx_opmode_blocking)
                self.reward = 1
                self.RL.learn(s=w2m_pos,a=self.action,r=self.reward)

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

        print("[PREDICT RESULT]: Desired Grasp Position: [{}, {}], Orientation: [{}].".format(pos_data[0], pos_data[1], ori_data))
        # backdata, taskcontinue = self.DesiredPositionScore(pos_data)
        # if taskcontinue:
            # Open the Gripper
        self.gripper_open()
        time.sleep(1)

        # Go to the position and orientation above the object
        self.Go((pos_data[0], pos_data[1], 0.1, 0.0, 0.0, ori_data))

        # Go to grasp
        self.Go((pos_data[0], pos_data[1], self.grasp_high, 0.0, 0.0, ori_data))


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

        if self.Check == 'True':
            grasp_success = True
            print("[IMPORTANT RESULT]: Nice Grasp!!! !^U^!")

        elif self.Check == 'False':
            grasp_success = False
            print("[IMPORTANT RESULT]: Nothing Grasped. TUT.TUT")

        # label_value, prev_reward_value = self.trainer.get_label_value(grasp_success)

        # self.trainer.backprop(self.prev_heatmap, self.prev_best_pix_ind, label_value)

        time.sleep(1)
        # Pick the object up
        self.Go((pos_data[0], pos_data[1], self.pre_grasp_high, 0.0, 0.0, ori_data))



    def gripper_open(self):
        """
        open the onrobot RG2 gripper
        using the defined urscript
        """

        ret_resp,ret_ints,ret_floats,ret_strings,ret_buffer = vrep.simxCallScriptFunction(
            self.sim_client, 'remoteApiCommandServer',vrep.sim_scripttype_childscript,
            'GripperOpen',[0], [0.0], "false", bytearray(), vrep.simx_opmode_blocking)
        time.sleep(1)

    def gripper_close(self):
        """
        close the onrobot RG2 gripper
        using the defined urscript
        """

        ret_resp,ret_ints,ret_floats,ret_strings,ret_buffer = vrep.simxCallScriptFunction(
            self.sim_client, 'remoteApiCommandServer',vrep.sim_scripttype_childscript,
            'GripperClose',[0], [0.0], "false", bytearray(), vrep.simx_opmode_blocking)
        time.sleep(1)