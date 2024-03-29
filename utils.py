import time
import datetime
import os
import numpy as np
import cv2
# import seaborn as sns
import matplotlib.pyplot as plt
import struct
import math
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

#! Map Struct
import numpy as np
import scipy.ndimage
class Map():
    '''
    Map Class: pixel map class
    using workspace_limits and resolutions to create a pixel map
    '''
    def __init__(self,workspace_limits, resolutions):
        self.workspace_limits = workspace_limits
        self.resolutions = resolutions
        self.GOAL_STATE = []
        self.START_STATE = []
        # self.range_ = int(self.depth.shape[0]/50)

    def add_depth(self, depth):
        '''
        resize the depth map and create the heatmap
        '''
        self.depth = scipy.ndimage.zoom(depth, (self.resolutions[0]/depth.shape[0], self.resolutions[1]/depth.shape[1]))
        self.heatmap = np.zeros(self.resolutions)

    def MapToWorld(self, map_idx):
        '''
        using workspace limits to compute the world position of the map postion
        map_pos -> world_pos
        '''
        world_x = ((map_idx[0]+1) * np.fabs(self.workspace_limits[0][0]-self.workspace_limits[0][1]))/self.resolutions[0] + self.workspace_limits[0][0]
        world_y = ((map_idx[1]+1) * np.fabs(self.workspace_limits[1][0]-self.workspace_limits[1][1]))/self.resolutions[1] + self.workspace_limits[1][0]
        return([world_x, world_y])

    def WorldToMap(self, world_idx):
        '''
        using workspace limits to compute the map position of the world position
        world_pos -> map_pos
        '''
        map_x = int((world_idx[0] - self.workspace_limits[0][0]) * self.resolutions[0]/np.fabs(self.workspace_limits[0][0]-self.workspace_limits[0][1]))-1
        map_y = int((world_idx[1] - self.workspace_limits[1][0]) * self.resolutions[1]/np.fabs(self.workspace_limits[1][0]-self.workspace_limits[1][1]))-1
        return([map_x, map_y])

    def updatemap(self, x, y, value):
        '''
        Update the heatmap with given value
        '''
        if (x >= 0 and x < self.heatmap.shape[0]) and (y >=0 and y < self.heatmap.shape[1]):
            self.heatmap[x][y] =  value

    def updateFree(self, pos, angle):
        '''
        determine the position of the robot end,
        and update the heatmap with free value
        '''
        self.update_explore_complete(pos)
        for i in range(self.range_):
            for j in range(self.range_):
                #TODO: Add some if condition to judge in the map limits

                x = int(pos[0]+ (i * np.cos(angle) - j * np.sin(angle)))
                y = int(pos[1]+ (i * np.sin(angle) + j * np.cos(angle)))
                self.updatemap(x, y, 120)

                x = int(pos[0]- (i * np.cos(angle) - j * np.sin(angle)))
                y = int(pos[1]- (i * np.sin(angle) + j * np.cos(angle)))
                self.updatemap(x, y, 120)

                x = int(pos[0]+ (i * np.cos(angle) - j * np.sin(angle)))
                y = int(pos[1]- (i * np.sin(angle) + j * np.cos(angle)))
                self.updatemap(x, y, 120)

                x = int(pos[0]- (i * np.cos(angle) - j * np.sin(angle)))
                y = int(pos[1]+ (i * np.sin(angle) + j * np.cos(angle)))
                self.updatemap(x, y, 120)

    def updateFrontier(self, pos, angle):
        '''
        determine the position of the robot end,
        and update the heatmap with frontier value
        '''
        self.update_explore_complete(pos)
        for i in range(self.range_):
            for j in range(self.range_):
                #TODO: Add some if condition to judge in the map limits

                x = int(pos[0]+ (i * np.cos(angle) - j * np.sin(angle)))
                y = int(pos[1]+ (i * np.sin(angle) + j * np.cos(angle)))
                self.updatemap(x, y, 255)

                x = int(pos[0]+ (i * np.cos(angle) - j * np.sin(angle)))
                y = int(pos[1]- (i * np.sin(angle) + j * np.cos(angle)))
                self.heatmap[x][y] = 255

                x = int(pos[0]- (i * np.cos(angle) - j * np.sin(angle)))
                y = int(pos[1]- (i * np.sin(angle) + j * np.cos(angle)))
                self.updatemap(x, y, 255)

                x = int(pos[0]- (i * np.cos(angle) - j * np.sin(angle)))
                y = int(pos[1]+ (i * np.sin(angle) + j * np.cos(angle)))
                self.updatemap(x, y, 255)

    def step(self, action, current_pos):
        '''
        action: | UP | DOWN | LEFT | RIGHT |
        '''
        # act_pos = [current_pos[0],current_pos[1]]
        now_pos = self.WorldToMap(current_pos)
        if action == 0:
            now_pos[0] -= 1
        elif action == 1:
            now_pos[0] += 1
        elif action == 2: # x+
            now_pos[1] -= 1
        elif action == 3: # x-
            now_pos[1] += 1
        predict_pos = self.MapToWorld(now_pos)
        return predict_pos

class Logger():

    def __init__(self, logging_directory):
        """
        Logger Class: Saving Data
        """
        # Create directory to save data
        timestamp = time.time()
        timestamp_value = datetime.datetime.fromtimestamp(timestamp)

        self.base_directory = os.path.join(logging_directory, timestamp_value.strftime('%Y-%m-%d@%H-%M-%S'))
        # print('Creating data logging session: %s' % (self.base_directory))

        self.force_sensor_data_directory = os.path.join(self.base_directory, 'data', 'force-sensor-data')
        if not os.path.exists(self.force_sensor_data_directory):
            os.makedirs(self.force_sensor_data_directory)

        self.image_directory = os.path.join(self.base_directory, 'image')
        if not os.path.exists(self.image_directory):
            os.makedirs(self.image_directory)


    def save_force_data(self, force_data):
        '''
        save the force data value
        '''
        np.savetxt(os.path.join(self.force_sensor_data_directory, 'foce_data.csv'), force_data, delimiter=',')

    def save_heatmaps(self, heatmap):
        '''
        save the heatmaps numpy data to img
        '''
        # heatmap = cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR)
        # sns.set()
        # ax = sns.heatmap(heatmap)
        # plt.ion()
        # plt.pause(3)
        # plt.close()
        cv2.imwrite(os.path.join(self.image_directory, 'heatmap.png'), heatmap)

    def save_depthImg(self, depthImg):
        '''
        save the depth image
        '''
        cv2.imwrite(os.path.join(self.image_directory, 'depth.png'), depthImg)

    def save_colorImg(self, colorImg):
        '''
        save the color image with RGB
        '''
        cv2.imwrite(os.path.join(self.image_directory, 'color.png'), colorImg)

    def save_depthheatImg(self, heatImg):
        cv2.imwrite(os.path.join(self.image_directory, 'depthheat.png'), heatImg)


class Filter():
    def __init__(self):
        """
        Filter Class: Filter data
        """
        self.OldData = None
        self.NewData = None
        self.Initial = False

    def LowPassFilter(self, data, filterParam=0.2):
        '''
        Low Pass Filter Function
        '''
        if not (self.Initial):
            self.OldData = data
            self.Initial = True
            return data
        else:
            self.NewData = data
            return_ = []
            for i in range(len(self.NewData)):
                return_.append((1-filterParam)*self.OldData[i] + self.NewData[i])
            self.OldData = data
            # print(return_)
            return return_

    def LowPassFilterClear(self):
        self.OldData = None
        self.NewData = None
        self.Initial = False



# Get rotation matrix from euler angles
def euler2rotm(theta):
    '''
    Get rotation matrix from euler angles
    '''
    R_x = np.array([[1,         0,                  0                   ],
                    [0,         math.cos(theta[0]), -math.sin(theta[0]) ],
                    [0,         math.sin(theta[0]), math.cos(theta[0])  ]
                    ])
    R_y = np.array([[math.cos(theta[1]),    0,      math.sin(theta[1])  ],
                    [0,                     1,      0                   ],
                    [-math.sin(theta[1]),   0,      math.cos(theta[1])  ]
                    ])         
    R_z = np.array([[math.cos(theta[2]),    -math.sin(theta[2]),    0],
                    [math.sin(theta[2]),    math.cos(theta[2]),     0],
                    [0,                     0,                      1]
                    ])            
    R = np.dot(R_z, np.dot( R_y, R_x ))
    return R

def get_pointcloud(color_img, depth_img, camera_intrinsics):
    '''
    Get point Cloud function
    '''
    # Get depth image size
    im_h = depth_img.shape[0]
    im_w = depth_img.shape[1]

    # Project depth into 3D point cloud in camera coordinates
    pix_x,pix_y = np.meshgrid(np.linspace(0,im_w-1,im_w), np.linspace(0,im_h-1,im_h))
    cam_pts_x = np.multiply(pix_x-camera_intrinsics[0][2],depth_img/camera_intrinsics[0][0])
    cam_pts_y = np.multiply(pix_y-camera_intrinsics[1][2],depth_img/camera_intrinsics[1][1])
    cam_pts_z = depth_img.copy()
    cam_pts_x.shape = (im_h*im_w,1)
    cam_pts_y.shape = (im_h*im_w,1)
    cam_pts_z.shape = (im_h*im_w,1)

    # Reshape image into colors for 3D point cloud
    rgb_pts_r = color_img[:,:,0]
    rgb_pts_g = color_img[:,:,1]
    rgb_pts_b = color_img[:,:,2]
    rgb_pts_r.shape = (im_h*im_w,1)
    rgb_pts_g.shape = (im_h*im_w,1)
    rgb_pts_b.shape = (im_h*im_w,1)

    cam_pts = np.concatenate((cam_pts_x, cam_pts_y, cam_pts_z), axis=1)
    rgb_pts = np.concatenate((rgb_pts_r, rgb_pts_g, rgb_pts_b), axis=1)

    return cam_pts, rgb_pts


def get_heightmap(color_img, depth_img, cam_intrinsics, cam_pose, workspace_limits, heightmap_resolution):
    '''
    Get the heightmap with depth image
    '''
    # Compute heightmap size
    heightmap_size = np.round(((workspace_limits[1][1] - workspace_limits[1][0])/heightmap_resolution, (workspace_limits[0][1] - workspace_limits[0][0])/heightmap_resolution)).astype(int)

    # Get 3D point cloud from RGB-D images
    surface_pts, color_pts = get_pointcloud(color_img, depth_img, cam_intrinsics)

    # Transform 3D point cloud from camera coordinates to robot coordinates
    surface_pts = np.transpose(np.dot(cam_pose[0:3,0:3],np.transpose(surface_pts)) + np.tile(cam_pose[0:3,3:],(1,surface_pts.shape[0])))

    # Sort surface points by z value
    sort_z_ind = np.argsort(surface_pts[:,2])
    surface_pts = surface_pts[sort_z_ind]
    color_pts = color_pts[sort_z_ind]
    print(surface_pts)
    # Filter out surface points outside heightmap boundaries
    heightmap_valid_ind = np.logical_and(np.logical_and(np.logical_and(np.logical_and(surface_pts[:,0] >= workspace_limits[0][0], surface_pts[:,0] < workspace_limits[0][1]), surface_pts[:,1] >= workspace_limits[1][0]), surface_pts[:,1] < workspace_limits[1][1]), surface_pts[:,2] < workspace_limits[2][1])
    
    # surface_pts = surface_pts[heightmap_valid_ind]
    # color_pts = color_pts[heightmap_valid_ind]
    # Create orthographic top-down-view RGB-D heightmaps
    color_heightmap_r = np.zeros((heightmap_size[0], heightmap_size[1], 1), dtype=np.uint8)
    color_heightmap_g = np.zeros((heightmap_size[0], heightmap_size[1], 1), dtype=np.uint8)
    color_heightmap_b = np.zeros((heightmap_size[0], heightmap_size[1], 1), dtype=np.uint8)
    depth_heightmap = np.zeros(heightmap_size)
    heightmap_pix_x = np.floor((surface_pts[:,0] - workspace_limits[0][0])/heightmap_resolution).astype(int)
    heightmap_pix_y = np.floor((surface_pts[:,1] - workspace_limits[1][0])/heightmap_resolution).astype(int)
    color_heightmap_r[heightmap_pix_y,heightmap_pix_x] = color_pts[:,[0]]
    color_heightmap_g[heightmap_pix_y,heightmap_pix_x] = color_pts[:,[1]]
    color_heightmap_b[heightmap_pix_y,heightmap_pix_x] = color_pts[:,[2]]
    color_heightmap = np.concatenate((color_heightmap_r, color_heightmap_g, color_heightmap_b), axis=2)
    print(surface_pts)
    depth_heightmap[heightmap_pix_y,heightmap_pix_x] = surface_pts[:,2]
    z_bottom = workspace_limits[2][0]
    depth_heightmap = depth_heightmap - z_bottom
    depth_heightmap[depth_heightmap < 0] = 0
    depth_heightmap[depth_heightmap == -z_bottom] = np.nan

    return color_heightmap, depth_heightmap