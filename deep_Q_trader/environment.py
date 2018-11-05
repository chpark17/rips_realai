'''
This file is the Exchange Simulator for Bitcoin based upon per minute historical prices.
'''

import random
import numpy as np
import math

from deep_Q_trader.portfolio import Portfolio
from constants import *

class train_environment:

    def __init__(self, logger, processor):
        self.logger = logger
        self.episode_number = 0
        self.history_length = HISTORY_LENGTH
        self.horizon = HORIZON
        self.test_frac = TEST_FRAC

        self.processor = processor
        self.start_index = self.history_length
        self.end_index = math.floor(len(self.processor.price)*(1 - self.test_frac)) - self.horizon
        # percentatge of the part of portfolio that is in the bitcoin market
        self.action_list = [LONG, NEUTRAL, SHORT]
        self.action_name = ['LONG', 'NEUTRAL', 'SHORT']
        self.portfolio = Portfolio(logger,self.processor.price[0])
        
    def new_random_episode(self, history, replay_memory):
        '''
        select a new random episode from the unused data and return trade_rem
        each episode is used for #(horizon) trainings
        '''
        self.timesteps = 0
        message_list = []
        self.episode_number = self.episode_number + 1
        message_list.append("Starting a new episode numbered {}".format(self.episode_number))

        self.current = random.randint(self.start_index, self.end_index) # start & end inclusive
        message_list.append("Starting index selected for episode number {} is {}".format(
            self.episode_number, self.current))

        for i in range(self.current - self.history_length+1, self.current+1):
            # self.portfolio.get_channels()
            screen = np.array(self.processor.get_channels(i))
            history.add(screen)

        map(self.logger.debug, message_list)
        # trade_rem is a float between 0.0 and 1.0
        # trede_rem = 1 means this episode is unused
        return 1.0 

    def act(self, action): # action is the index of action_list
        '''
        return the current screen (i.e. the state of current hour) & 
        its corresponding reward, terminal, trade_rem
        '''
        self.portfolio.apply_action(self.processor.price[self.current], self.action_list[action])
        self.portfolio.printProtfolio()
        
        screen = np.array(self.processor.get_channels(self.current))
        # define the immd reward as the profit made compared to the previous step
        reward = -(self.processor.price[self.current] / self.processor.price[self.current-1] - 1)*self.action_list[action]
        profit_rate = 100*self.portfolio.getCurrentProfit()/self.portfolio.initial_total_in_USD

        message = "Index {}:==: Action: {} ; Reward: {} ; Profit rate: {} %".format(
            self.current, self.action_name[action], reward, profit_rate
        )
        self.logger.debug(message)

        self.timesteps = self.timesteps + 1
        if self.timesteps != self.horizon:
            self.current = self.current + 1
            terminal = False
            trade_rem = ((1.0/self.horizon) * (self.horizon - self.timesteps)) 
        else: # terminal if timesteps == horizon
            self.portfolio.reset(initial_price = self.processor.price[self.current])
            terminal = True
            trade_rem = 0.0
            
        print('screen:{}, reward:{}, terminal:{}, trade_rem:{}.'.format(screen, reward, terminal, trade_rem))
        return screen, reward, terminal, trade_rem

class test_environment:

    def __init__(self, logger, processor):
        self.logger = logger
        self.history_length = HISTORY_LENGTH
        self.test_frac = TEST_FRAC

        self.processor = processor
        self.start_index = math.floor(len(self.processor.price)*(1 - self.test_frac))
        self.end_index = len(self.processor.price) - 2
        # percentatge of the part of portfolio that is in the bitcoin market
        self.action_list = [LONG, NEUTRAL, SHORT]
        self.action_name = ['LONG', 'NEUTRAL', 'SHORT']
        self.portfolio = Portfolio(logger,self.processor.price[self.start_index])

        self.current = self.start_index
        
    def new_data_point(self, history, replay_memory):
        '''
        select a new random episode from the unused data and return trade_rem
        each episode is used for #(horizon) trainings
        '''
        self.current += 1

        message = "Grab a new data point at index {}".format(self.current)
        self.logger.debug(message)
        print("Grab a new data point at index {}".format(self.current))

        for i in range(self.current - self.history_length+1, self.current+1): # include the start & exclude the end
            screen = np.array(self.processor.get_channels(i))
            history.add(screen)

        # trade_rem is a float between 0.0 and 1.0
        # trede_rem = 1 means this episode is unused
        # the length of future data is infinite, so return 1
        return 1.0 

    def act(self, action): # action is the index of action_list
        '''
        return the current screen (i.e. the state of current hour) & 
        its corresponding reward, terminal, trade_rem, profit_rate
        '''        
        self.portfolio.apply_action(self.processor.price[self.current], self.action_list[action])
        self.portfolio.printProtfolio()

        screen = np.array(self.processor.get_channels(self.current))
        # define the immd reward as the profit made compared to the previous step
        reward = -(self.processor.price[self.current] / self.processor.price[self.current-1] - 1)*self.action_list[action]
        profit_rate = 100 * self.portfolio.getCurrentProfit() / self.portfolio.initial_total_in_USD

        message = "Index {}:==: Action: {} ; Reward: {} ; Profit rate: {} %".format(
            self.current, self.action_name[action], reward, profit_rate
        )
        self.logger.debug(message)

        if self.current < self.end_index:
            terminal = False
        else: # terminal if current == the last index in the testing set
            terminal = True
            self.logger.info("Testing ends.")

        trade_rem = 1.0
            
        print('screen:{}, reward:{}, terminal:{}, trade_rem:{}.'.format(screen, reward, terminal, trade_rem))
        return screen, reward, terminal, trade_rem, profit_rate
