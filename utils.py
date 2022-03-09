import time
import datetime
import os
import numpy as np


class DataLogger():

    def __init__(self):

        # Create directory to save data
        timestamp = time.time()
        timestamp_value = datetime.datetime.fromtimestamp(timestamp)

        logging_directory = os.path.abspath('logs')

        self.base_directory = os.path.join(logging_directory, timestamp_value.strftime('%Y-%m-%d.%H:%M:%S'))
        self.force_data_directory = os.path.join(self.base_directory, 'data', 'force_data')

        if not os.path.exists(self.force_data_directory):
            os.makedirs(self.force_data_directory)

    def save_tcp_force(self, force):
        print(force)
        #TODO: Need Fix BUG
        # with open(os.path.join(self.force_data_directory, 'tcp-force.txt'), 'w', encoding='utf-8') as f:
        #     for i in force:
        #         f.write(i+',')
        #     f.write('\n')

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
            return_ = (1-filterParam)*self.OldData + self.NewData
            self.OldData = data
            # print(return_)
            return return_