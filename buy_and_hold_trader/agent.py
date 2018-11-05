'''
This file implements other bitcoin trading strategies to make comparison with the DQT and evalute its performance.
'''
import math
import numpy as np
from deep_Q_trader.utils import save_npy
from constants import *

# buy and hold strategy
class buy_and_hold:
    def __init__(self, logger, processor):
        self.logger = logger
        self.test_frac = TEST_FRAC

        self.processor = processor
        self.start_index = math.floor(len(self.processor.price)*(1 - self.test_frac))
        self.end_index = len(self.processor.price)
        self.current = self.start_index

        self.initial_price = self.processor.price[self.start_index]
        self.current_price = self.processor.price[self.current]

        self.save_dir = "{}/results/{}.npy".format(PARENT_PATH, 'BH_profit_rates')


    def trade(self): # action is the index of action_list
        '''
        return the current price (i.e. the state of current hour), terminal & profit rate
        '''
        profit_rates = []

        for i in range(self.start_index, self.end_index):
            self.current = i
            self.current_price = self.processor.price[self.current]
            profit_rate = 100*(self.current_price - self.initial_price)/self.initial_price

            message = ' BH step {} :=: price: {}, profit_rate:{}.'.format(i, self.current_price, profit_rate)
            self.logger.debug(message)

            profit_rates.append(profit_rate)

        profit_rates = np.array(profit_rates)
        save_npy(profit_rates, self.save_dir, self.logger)
        


