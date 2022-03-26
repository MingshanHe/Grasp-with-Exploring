import time
import datetime
import os
import numpy as np
import cv2
import seaborn as sns
import matplotlib.pyplot as plt

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

        self.heatmap_image_directory = os.path.join(self.base_directory, 'image')
        if not os.path.exists(self.heatmap_image_directory):
            os.makedirs(self.heatmap_image_directory)


    def save_force_data(self, force_data):
        np.savetxt(os.path.join(self.force_sensor_data_directory, 'foce_data.csv'), force_data, delimiter=',')

    def save_heatmaps(self, heatmap):
        # heatmap = cv2.cvtColor(heatmap, cv2.COLOR_RGB2BGR)
        sns.set()
        ax = sns.heatmap(heatmap)
        plt.ion()
        plt.pause(3)
        plt.close()
        cv2.imwrite(os.path.join(self.heatmap_image_directory, 'heatmap.png'), heatmap)


class Filter():
    def __init__(self):
        """
        Filter Class: Filter data
        """
        self.OldData = None
        self.NewData = None
        self.Initial = False

    def LowPassFilter(self, data, filterParam=0.2):
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