import numpy as np
from sympy import false, true
import seaborn as sns
import matplotlib.pyplot as plt


class Frontier():
    def __init__(self):
        self.size = None
        self.min_distance = None
        self.cost = None
        self.initial = None

        self.middle = None
        self.points = []

        self.centroid = None
        self.force    = None
        self.direct   = None



class HeatMap():
    def __init__(self, workspace_limits, resolutions):
        self.heatmap = np.zeros((resolutions, resolutions))
        self.explore_complete = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.workspace_limits = workspace_limits
        self.resolutions = resolutions
        self.range_ = int(resolutions/50)

        self.sub_range0 = {}
        self.sub_range1 = {}
        self.sub_range2 = {}
        self.sub_range3 = {}
        self.sub_range4 = {}
        self.sub_range5 = {}
        self.sub_range6 = {}
        self.sub_range7 = {}
        self.sub_range8 = {}

    def MapToWorld(self, idx):
        world_x = (idx[0] * np.fabs(self.workspace_limits[0][0]-self.workspace_limits[0][1]))/self.resolutions + self.workspace_limits[0][0]
        world_y = (idx[1] * np.fabs(self.workspace_limits[1][0]-self.workspace_limits[1][1]))/self.resolutions + self.workspace_limits[1][0]
        return([world_x, world_y])

    def WorldToMap(self, idx):
        map_x = int((idx[0] - self.workspace_limits[0][0]) * self.resolutions/np.fabs(self.workspace_limits[0][0]-self.workspace_limits[0][1]))
        map_y = int((idx[1] - self.workspace_limits[1][0]) * self.resolutions/np.fabs(self.workspace_limits[1][0]-self.workspace_limits[1][1]))
        return([map_x, map_y])

    def update_explore_compelete(self, pos):

        if (pos[0]>=0 and pos[0] < int(self.resolutions/3)) and(pos[1]>=0 and pos[1] < int(self.resolutions/3)):
            state = str(pos[0])+','+str(pos[1])
            if state not in self.sub_range0:
                self.sub_range0[state] = 1
                self.explore_complete[0] = (self.explore_complete[0]*(self.resolutions*self.resolutions)/9 + 1)/((self.resolutions*self.resolutions)/9)
        elif (pos[0]>=0 and pos[0] < int(self.resolutions/3)) and(pos[1]>=int(self.resolutions/3) and pos[1] < int(self.resolutions*2/3)):
            state = str(pos[0])+','+str(pos[1])
            if state not in self.sub_range1:
                self.sub_range1[state] = 1
                self.explore_complete[1] = (self.explore_complete[1]*(self.resolutions*self.resolutions)/9 + 1)/((self.resolutions*self.resolutions)/9)
        elif (pos[0]>=0 and pos[0] < int(self.resolutions/3)) and(pos[1]>=int(self.resolutions*2/3) and pos[1] < int(self.resolutions)):
            state = str(pos[0])+','+str(pos[1])
            if state not in self.sub_range2:
                self.sub_range2[state] = 1
                self.explore_complete[2] = (self.explore_complete[2]*(self.resolutions*self.resolutions)/9 + 1)/((self.resolutions*self.resolutions)/9)
        elif (pos[0]>=int(self.resolutions/3) and pos[0] < int(self.resolutions*2/3)) and(pos[1]>=0 and pos[1] < int(self.resolutions/3)):
            state = str(pos[0])+','+str(pos[1])
            if state not in self.sub_range3:
                self.sub_range3[state] = 1
                self.explore_complete[3] = (self.explore_complete[3]*(self.resolutions*self.resolutions)/9 + 1)/((self.resolutions*self.resolutions)/9)
        elif (pos[0]>=int(self.resolutions/3) and pos[0] < int(self.resolutions*2/3)) and(pos[1]>=int(self.resolutions/3) and pos[1] < int(self.resolutions*2/3)):
            state = str(pos[0])+','+str(pos[1])
            if state not in self.sub_range4:
                self.sub_range4[state] = 1
                self.explore_complete[4] = (self.explore_complete[4]*(self.resolutions*self.resolutions)/9 + 1)/((self.resolutions*self.resolutions)/9)
        elif (pos[0]>=int(self.resolutions/3) and pos[0] < int(self.resolutions*2/3)) and(pos[1]>=int(self.resolutions*2/3) and pos[1] < int(self.resolutions)):
            state = str(pos[0])+','+str(pos[1])
            if state not in self.sub_range5:
                self.sub_range5[state] = 1
                self.explore_complete[5] = (self.explore_complete[5]*(self.resolutions*self.resolutions)/9 + 1)/((self.resolutions*self.resolutions)/9)
        elif (pos[0]>=int(self.resolutions*2/3) and pos[0] < int(self.resolutions)) and(pos[1]>=0 and pos[1] < int(self.resolutions/3)):
            state = str(pos[0])+','+str(pos[1])
            if state not in self.sub_range6:
                self.sub_range6[state] = 1
                self.explore_complete[6] = (self.explore_complete[6]*(self.resolutions*self.resolutions)/9 + 1)/((self.resolutions*self.resolutions)/9)
        elif (pos[0]>=int(self.resolutions*2/3) and pos[0] < int(self.resolutions)) and(pos[1]>=int(self.resolutions/3) and pos[1] < int(self.resolutions*2/3)):
            state = str(pos[0])+','+str(pos[1])
            if state not in self.sub_range7:
                self.sub_range7[state] = 1
                self.explore_complete[7] = (self.explore_complete[7]*(self.resolutions*self.resolutions)/9 + 1)/((self.resolutions*self.resolutions)/9)
        elif (pos[0]>=int(self.resolutions*2/3) and pos[0] < int(self.resolutions)) and(pos[1]>=int(self.resolutions*2/3) and pos[1] < int(self.resolutions)):
            state = str(pos[0])+','+str(pos[1])
            if state not in self.sub_range8:
                self.sub_range0[state] = 1
                self.explore_complete[8] = (self.explore_complete[8]*(self.resolutions*self.resolutions)/9 + 1)/((self.resolutions*self.resolutions)/9)

    def updatemap(self, x, y, value):
        if (x >= 0 and x < self.heatmap.shape[0]) and (y >=0 and y < self.heatmap.shape[1]):
            self.heatmap[x][y] =  value

    def updateFree(self, pos, angle):
        self.update_explore_compelete(pos)
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
        self.update_explore_compelete(pos)
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


class FrontierSearch():
    def __init__(self,workspace_limits, resolutions):
        self.map    = HeatMap(workspace_limits, resolutions)
        self.action_space = ['x+', 'x-', 'y+', 'y-']
        self.n_actions = len(self.action_space)
        self.points = []
        self.derive_points = []

    def buildNewFree(self, initial_cell, initial_angle):
        # initialize frontier structure
        frontier = Frontier()
        frontier.centroid = self.map.WorldToMap(initial_cell)
        frontier.direct   = initial_angle

        # build the initial frontier
        self.map.updateFree(frontier.centroid, frontier.direct)

    def buildNewFrontier(self, initial_cell, initial_force, initial_angle):

        # initialize frontier structure
        frontier = Frontier()
        frontier.centroid = self.map.WorldToMap(initial_cell)
        frontier.force    = initial_force
        frontier.direct   = initial_angle

        # build the initial frontier
        self.map.updateFrontier(frontier.centroid, frontier.direct)

        self.points.append(frontier.centroid)
        # print(self.points)

        # build 2 neighboor
        # for i in range(1):
        #     frontier_ = Frontier()
        #     frontier_.centroid = (
        #         frontier.centroid[0]-self.map.range_*(i+1)*2*np.cos(np.arctan2(frontier.force[0],frontier.force[1])-frontier.direct),
        #         frontier.centroid[1]+self.map.range_*(i+1)*2*np.sin(np.arctan2(frontier.force[0],frontier.force[1])-frontier.direct))
        #     self.map.updateFrontier(frontier_.centroid, frontier.direct)
        #     self.derive_points.append(frontier_.centroid)

        #     frontier_.centroid = (
        #         frontier.centroid[0]-self.map.range_*(i+1)*2*-np.cos(np.arctan2(frontier.force[0],frontier.force[1])-frontier.direct),
        #         frontier.centroid[1]+self.map.range_*(i+1)*2*-np.sin(np.arctan2(frontier.force[0],frontier.force[1])-frontier.direct))
        #     self.map.updateFrontier(frontier_.centroid, frontier.direct)
        #     self.derive_points.append(frontier_.centroid)

    def step(self, action, current_pos, unit):
        '''
        action: | x+ | x- | y+ | y- |
        '''
        act_pos = [current_pos[0],current_pos[1]]
        if action == 0: # x+
            if ((current_pos[0] + unit) >= self.map.workspace_limits[0][0] and (current_pos[0] + unit) <= self.map.workspace_limits[0][1]):
                act_pos[0] = current_pos[0] + unit
        elif action == 1: # x-
            if ((current_pos[0] - unit) >= self.map.workspace_limits[0][0] and (current_pos[0] - unit) <= self.map.workspace_limits[0][1]):
                act_pos[0] = current_pos[0] - unit
        elif action == 2: # x+
            if ((current_pos[1] + unit) >= self.map.workspace_limits[1][0] and (current_pos[1] + unit) <= self.map.workspace_limits[1][1]):
                act_pos[1] = current_pos[1] + unit
        elif action == 3: # x-
            if ((current_pos[1] - unit) >= self.map.workspace_limits[1][0] and (current_pos[1] - unit) <= self.map.workspace_limits[1][1]):
                act_pos[1] = current_pos[1] - unit
        return act_pos

    def grasp_point_angle(self):

        grasp_point = [0.0, 0.0]
        for i in range(len(self.points)):
            grasp_point[0] += self.points[i][0]
            grasp_point[1] += self.points[i][1]
        grasp_point[0] = grasp_point[0]/len(self.points)
        grasp_point[1] = grasp_point[1]/len(self.points)

        grasp_angle = 0
        for i in range(len(self.points)):
            k = (grasp_point[1]-self.points[i][1])/(grasp_point[0]-self.points[i][0])
            grasp_angle += k
        grasp_angle = grasp_angle/len(self.points)
        print(grasp_angle)
        grasp_angle = np.arctan((grasp_angle))

        grasp_point = self.map.MapToWorld(grasp_point)
        print(grasp_point)
        grasp_point[0] = grasp_point[0] - 0.04*np.sin(grasp_angle)
        grasp_point[1] = grasp_point[1] - 0.04*np.cos(grasp_angle)
        print(grasp_point)
        return(grasp_point, grasp_angle)