'''
This file keeps the portfolio of Bitcoin and USD.
'''

import math
import numpy as np
from copy import copy
from constants import *

class Portfolio:
    
    # initialize the portfolio variables
    def __init__(self, logger, initial_price, frac_bitcoin = 0.5, initial_total_in_USD = 10000, long_frac = 0.9, short_frac = 0.1):
        self.logger = logger
        self.frac_bitcoin = frac_bitcoin
        self.long_frac = long_frac
        self.short_frac = short_frac
        self.initial_total_in_USD = initial_total_in_USD
        self.current_total_in_USD = self.initial_total_in_USD
        self.previous_total_in_USD = self.initial_total_in_USD

        self.USD = (1-self.frac_bitcoin)*self.current_total_in_USD
        self.bitcoin = self.frac_bitcoin*self.current_total_in_USD/initial_price

    # reset portfolio
    def reset(self, initial_price, frac_bitcoin = 0.5, initial_total_in_USD = 10000):
        self.frac_bitcoin = frac_bitcoin
        self.initial_total_in_USD = initial_total_in_USD
        self.current_total_in_USD = self.initial_total_in_USD
        self.previous_total_in_USD = self.initial_total_in_USD

        self.USD = (1-self.frac_bitcoin)*self.current_total_in_USD
        self.bitcoin = self.frac_bitcoin*self.current_total_in_USD/initial_price
        message = "Rest portfolio"
        self.logger.debug(message)

    def printProtfolio(self):
        message = 'protfolio: {} USD & {} bitcoins, {} USD in total currently; {} USD in total previously.'\
        .format(self.USD, self.bitcoin, self.current_total_in_USD, self.previous_total_in_USD)
        self.logger.debug(message)
        
    # return current protfolio    
    def getProtfolio(self):
        return self.USD, self.bitcoin, self.current_total_in_USD

    def getCurrentProfit(self):
        return self.current_total_in_USD - self.initial_total_in_USD

    # apply action to the portfolio
    def apply_action(self, current_price, action):
        # total asset previously in USD
        self.previous_total_in_USD = self.current_total_in_USD
        # total asset currently in USD
        self.current_total_in_USD = self.USD + self.bitcoin*current_price

        self.frac_bitcoin = self.bitcoin * current_price / self.current_total_in_USD

        if action is LONG:
            if self.frac_bitcoin < self.long_frac:
                self.bitcoin = self.current_total_in_USD * self.long_frac / current_price
                self.USD = self.current_total_in_USD * (1 - self.long_frac)
                self.frac_bitcoin = copy(self.long_frac)
        if action is SHORT:
            if self.frac_bitcoin > self.short_frac:
                self.bitcoin = self.current_total_in_USD * self.short_frac / current_price
                self.USD = self.current_total_in_USD * (1 - self.short_frac)
                self.frac_bitcoin = copy(self.short_frac)

        message = 'long_frac = {}, short_frac = {}'.format(self.long_frac, self.short_frac)
        self.logger.debug(message)

    

