'''
This file is used to process the raw data & generate features.

Reference: https://github.com/philipperemy/deep-learning-bitcoin
'''

import math
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
from constants import *

class Preprocessor:

    def __init__(self, logger):
        # suppose alway put the .csv data file in the 'data' fold in the same directory as this file
        self.dataset_path = "{}/data/{}.csv".format(PARENT_PATH, DATA_FILE)
        self.logger = logger

        self.preprocess()

    @property 
    def UTC_time(self):
        return self._UTC_time

    @property 
    def price(self):
        return self._price

    # the data imported has no holes & is average from 3 exchanges    
    def preprocess(self):
        self._data = pd.read_csv(self.dataset_path) # import .csv file as a dataframe
        message = 'Columns found in the dataset {}'.format(self._data.columns)
        self.logger.info(message)

        # bitcoin price averaged from 3 different exchanges
        self._price = self._data['Average_price'].values

        split_index = math.floor(0.4*len(self._price)) 

        self._price = self._price[split_index::2] # only use the per hour data

        timestamps = self._data['Timestamp'].values

        timestamps = timestamps[split_index::2]
        # conver the timstamps to UTC: Coordinated Universal Time (same as GMT: Greenwich Mean Time)
        self._UTC_time = []
        for timestamp in timestamps:
            utc_time = datetime.utcfromtimestamp(timestamp)
            self._UTC_time.append(utc_time.strftime("%Y-%m-%d %H:%M:%S.%f+00:00 (UTC)"))
        self._UTC_time = np.array(self._UTC_time)

        self._data = None # free memory          
    