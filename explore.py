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
        self.workspace_limits = workspace_limits
        self.resolutions = resolutions
        self.range_ = int(resolutions/50)

    def MapToWorld(self, idx):
        world_x = (idx[0] * np.fabs(self.workspace_limits[0][0]-self.workspace_limits[0][1]))/self.resolutions + self.workspace_limits[0][0]
        world_y = (idx[1] * np.fabs(self.workspace_limits[1][0]-self.workspace_limits[1][1]))/self.resolutions + self.workspace_limits[1][0]
        return(world_x, world_y)


    def WorldToMap(self, idx):
        map_x = int((idx[0] - self.workspace_limits[0][0]) * self.resolutions/np.fabs(self.workspace_limits[0][0]-self.workspace_limits[0][1]))
        map_y = int((idx[1] - self.workspace_limits[1][0]) * self.resolutions/np.fabs(self.workspace_limits[1][0]-self.workspace_limits[1][1]))
        return(map_x, map_y)

    def updateFrontier(self, pos, angle):
        print(pos)

        for i in range(self.range_):
            for j in range(self.range_):
                #TODO: Add some if condition to judge in the map limits

                x = int(pos[0]+ (i * np.cos(angle) - j * np.sin(angle)))
                y = int(pos[1]+ (i * np.sin(angle) + j * np.cos(angle)))
                self.heatmap[x][y] = 255

                # x = int(pos[0]+ (i * np.cos(angle) - j * np.sin(angle)))
                # y = int(pos[1]+ (i * np.sin(angle) + j * np.cos(angle)))
                # self.heatmap[x][y] = 255

                x = int(pos[0]- (i * np.cos(angle) - j * np.sin(angle)))
                y = int(pos[1]- (i * np.sin(angle) + j * np.cos(angle)))
                self.heatmap[x][y] = 255

                # x = int(pos[0]- (i * np.cos(angle) - j * np.sin(angle)))
                # y = int(pos[1]- (i * np.sin(angle) + j * np.cos(angle)))
                # self.heatmap[x][y] = 255

        sns.set()
        ax = sns.heatmap(self.heatmap)
        plt.ion()
        plt.pause(1)
        plt.close()



class FrontierSearch():
    def __init__(self,workspace_limits, resolutions):
        self.map = HeatMap(workspace_limits, resolutions)

    def isNewFrontierCell(self, idx, frontier_flag):

        # check that cell is unknown and not already marked as frontier
        if(self.map[idx[0]][idx[1]] == 0 | frontier_flag[idx[0]][idx[1]]):
            return false

        # frontier cells should have at least one cell
        # in 4-connected neighbourhood that is free
        # TODO: Need add nhood4 function
        else:
            return true

    def buildNewFrontier(self, initial_cell, initial_force, initial_angle):

        # initialize frontier structure
        frontier = Frontier()
        frontier.centroid = self.map.WorldToMap(initial_cell)
        frontier.force    = initial_force
        frontier.direct   = initial_angle

        # build the initial frontier
        self.map.updateFrontier(frontier.centroid, frontier.direct)

        # build 4 neighboor
        for i in range(1):
            frontier_ = Frontier()
            frontier_.centroid = (
                frontier.centroid[0]-self.map.range_*(i+1)*2*np.cos(np.arctan2(frontier.force[0],frontier.force[1])-frontier.direct),
                frontier.centroid[1]+self.map.range_*(i+1)*2*np.sin(np.arctan2(frontier.force[0],frontier.force[1])-frontier.direct))
            self.map.updateFrontier(frontier_.centroid, frontier.direct)

            frontier_.centroid = (
                frontier.centroid[0]-self.map.range_*(i+1)*2*-np.cos(np.arctan2(frontier.force[0],frontier.force[1])-frontier.direct),
                frontier.centroid[1]+self.map.range_*(i+1)*2*-np.sin(np.arctan2(frontier.force[0],frontier.force[1])-frontier.direct))
            self.map.updateFrontier(frontier_.centroid, frontier.direct)

        # output.centroid = (0,0)
        # output.size = 1
        # output.min_distance = np.NAN

        # record initial contact point for frontier
        # output.initial = self.map.WorldToMap(initial_cell)

