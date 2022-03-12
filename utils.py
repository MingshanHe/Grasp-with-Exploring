import time
import datetime
import os
import numpy as np

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


class Filter():
    def __init__(self):
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














