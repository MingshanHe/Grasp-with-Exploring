import time
import datetime
import os
import numpy as np
import cv2
import torch 
# import h5py 

class Logger():

    def __init__(self, logging_directory):

        # Create directory to save data
        timestamp = time.time()
        timestamp_value = datetime.datetime.fromtimestamp(timestamp)

        self.base_directory = os.path.join(logging_directory, timestamp_value.strftime('%Y-%m-%d@%H-%M-%S'))
        print('Creating data logging session: %s' % (self.base_directory))

        self.force_sensor_data_directory = os.path.join(self.base_directory, 'data', 'force-sensor-data')
        if not os.path.exists(self.force_sensor_data_directory):
            os.makedirs(self.force_sensor_data_directory)


    def save_force_data(self, force_data):
        np.savetxt(os.path.join(self.force_sensor_data_directory, 'foce_data.csv'), force_data, delimiter=',')
