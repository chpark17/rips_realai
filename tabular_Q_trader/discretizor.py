'''
This file is used to discretize the price data.

1 day(low:-1,same:0,high:1), 3 days(low:-1,same:0,high:1), 1 week(low:-1,same:0,high:1), 1 month(low:-1,same:0,high:1)

action(sell:-1,hold:0,buy:1)
'''

import numpy as np

class discretizor:

    def __init__(self, logger, processor):
        self.logger = logger

        # Timeframe Constants (hours)
        self.tf_day = 24
        self.tf_3day = 72
        self.tf_week = 168
        self.tf_month = 720

        self.price = processor.price
        self.start_index = self.tf_month
        self.num_data = len(self.price)

        # processed data
        self._day_rel = np.zeros((self.num_data,), dtype = int)
        self._3days_rel = np.zeros((self.num_data,), dtype = int)
        self._week_rel = np.zeros((self.num_data,), dtype = int)
        self._month_rel = np.zeros((self.num_data,), dtype = int)

        self.process()

    def find_relative_price(self, index, timeframe):
        # Find the price points for the past given timeframe
        tf_temp = []
        for i in range(index, index - timeframe, -1):
            tf_temp.append(self.price[i])
        tf_temp = np.array(tf_temp)

        # Find the thresholds according to the mean and std
        mean = np.mean(tf_temp, axis=0)
        std = np.std(tf_temp, axis=0)
        if self.price[index] < (mean - 0.425*std):
            return -1
        if self.price[index] > (mean + 0.425*std):
            return 1
        else:
            return 0

    def process(self):
        # Discretize the data
        for i in range(self.start_index, self.num_data):
            self._day_rel[i] = self.find_relative_price(i, self.tf_day)
            self._3days_rel[i] = self.find_relative_price(i, self.tf_3day)
            self._week_rel[i] = self.find_relative_price(i, self.tf_week)
            self._month_rel[i] = self.find_relative_price(i, self.tf_month)