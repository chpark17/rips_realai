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

class Processor:

    def __init__(self, logger):
        # suppose alway put the .csv data file in the 'data' fold in the same directory as this file
        self.dataset_path = "{}/data/{}.csv".format(PARENT_PATH, DATA_FILE)
        self.logger = logger
        self.history_length = HISTORY_LENGTH
        self.horizon = HORIZON
        self.total_num_data = 0
        self.train_start = 0

        self.preprocess()

    @property 
    def UTC_time(self):
        return self._UTC_time

    @property 
    def price(self):
        return self._price

    @property 
    def price_diff(self):
        return self._price_diff

    @property 
    def hight(self):
        return self._high
    
    @property 
    def low(self):
        return self._low

    @property 
    def reddit_doc(self):
        return self._reddit_doc

    @property 
    def MACD(self):
        return self._MACD
        
    # the data imported has no holes & is average from 3 exchanges    
    def preprocess(self):
        self._data = pd.read_csv(self.dataset_path) # import .csv file as a dataframe
        message = 'Columns found in the dataset {}'.format(self._data.columns)
        self.logger.info(message)
        # for MACD 26-12 (26 more data points)
        # self._data.drop(labels = range(len(self._data.index)-26, len(self._data.index)), axis = 0)
        self._data = self._data.dropna() # drop the rows containing NaN
        # bitcoin price averaged from 3 different exchanges
        self._price = self._data['Average_price'].values

        # size of the whole data set
        self.total_num_data = len(self._price)
        # use the last 60% data since they are more similar to the Bitcoin prices now
        self.train_start = math.floor(0.4 * self.total_num_data)
        message = 'Raw data size: {}, training data start at index: {}'.format(self.total_num_data, self.train_start)
        self.logger.info(message)

        pre_price = np.delete(self._price, len(self._price) - 1)
        self._price = np.delete(self._price, 0)
        # bitcoin price difference between current price and the price of last half an hour
        self._price_diff = np.subtract(self._price, pre_price)

        self._price = self._price[self.train_start - 1:]
        self._price_diff = self._price_diff[self.train_start - 1:]

        # the count of reddit documents that countain the word 'hack'
        self._reddit_doc = self._data['Reddit_count'].values
        self._reddit_doc = self._reddit_doc[self.train_start:]

        # MACD
        self._MACD = self._data['MACD'].values
        self._MACD = self._MACD[self.train_start:]

        # High
        self._high = self._data['Average_high'].values
        self._high = self._high[self.train_start:]

        # Low
        self._low = self._data['Average_low'].values
        self._low = self._low[self.train_start:]

        timestamps = self._data['Timestamp'].values
        # convert the timstamps to UTC: Coordinated Universal Time (same as GMT: Greenwich Mean Time)
        self._UTC_time = []
        for timestamp in timestamps:
            utc_time = datetime.utcfromtimestamp(timestamp)
            self._UTC_time.append(utc_time.strftime("%Y-%m-%d %H:%M:%S.%f+00:00 (UTC)"))
        self._UTC_time = np.array(self._UTC_time)
        self._UTC_time = self.UTC_time[self.train_start:]  

        self._data = None # free memory   
    
    def get_channels(self, current):
        return self._price_diff[current], self._high[current], self._low[current], self._reddit_doc[current], self._MACD[current]