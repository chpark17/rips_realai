'''
This file keep the tabular Q learning class.

'''
import numpy as np
import math
from random import randint
from deep_Q_trader.utils import save_npy
from constants import *

class tabular_Q:
    def __init__(self, logger, preprocessor, discretizor, gamma = DISCOUNT_FACTOR, alpha = LEARNING_RATE, test_frac = TEST_FRAC):
        self.logger = logger
        self.preprocessor = preprocessor
        self.discretizor = discretizor

        self.discountFactor = gamma
        self.learning_rate = alpha
        self.Q_table = np.zeros(243).reshape(3,3,3,3,3)  # initialize all entries of Q table to zeros

        self.test_frac = test_frac # fraction of test data to the whole data set
        self.train_start = max([HISTORY_LENGTH, self.discretizor.tf_month])
        self.test_start = math.floor(len(self.discretizor.price)*(1 - self.test_frac))
        self.train_end = self.test_start - 1
        self.test_end = len(self.discretizor.price) - 1
        
        self.test_outputs = []
        self.save_dir = "{}/results/{}.npy".format(PARENT_PATH, 'TBQ_profit_rates')

    def get_Q_table(self):
        return self.Q_table

    # define the immediate reward as the increment of the USD the trader has
    # 'price' is the current BTC price, 'áction' can only take -1(sell), 0(hold), 1(buy)
    def immedReward(self, pre_price, price, action):
        # reward = -(price - pre_price)*action
        reward = -(price/pre_price - 1)*action
        #if action == -1:
        #    reward = price/pre_n_price
        #elif action == 0:
        #    reward = pre_price/pre_n_price
        #else:
        #    reward = (2 - price/pre_price)*(pre_price/pre_n_price)
        return reward
  
    # data in the form of pandas.dataFrame
    # greedy policy: in 90% of time, choose the maximum as the Q value of next state; in the rest 10% of time, choose randomly
    # return a list of errors in each iteration
    def train(self):
        refer_Q_table = []

        message = 'training data from {} to {}.'.format(
            self.preprocessor.UTC_time[self.train_start],
            self.preprocessor.UTC_time[self.train_end]
            )
        self.logger.info(message)

        for i in range(self.train_start, self.train_end): 
            refer_Q_table = np.copy(self.Q_table)
  
            state_0 = refer_Q_table[self.discretizor._day_rel[i] + 1][self.discretizor._3days_rel[i] + 1][self.discretizor._week_rel[i] + 1][self.discretizor._month_rel[i] + 1]
            state_1 = self.Q_table[self.discretizor._day_rel[i] + 1][self.discretizor._3days_rel[i] + 1][self.discretizor._week_rel[i] + 1][self.discretizor._month_rel[i] + 1]
            state_2 = refer_Q_table[self.discretizor._day_rel[i + 1] + 1][self.discretizor._3days_rel[i + 1] + 1][self.discretizor._week_rel[i + 1] + 1][self.discretizor._month_rel[i + 1] + 1]

            pre_n_price = self.discretizor.price[i - 5] 
            pre_price = self.discretizor.price[i - 1]
            price = self.discretizor.price[i]
       
            # exploit 
            n = randint(0,10)
            if n == 0:
                q_next_state = state_2[randint(0,2)]
            else:
                q_next_state = max(state_2)
      
            # update Q(s,a) of the i-th data
            state_1[0] = (1-self.learning_rate)*state_1[0] + self.learning_rate*(self.immedReward(pre_price, price, -1) + self.discountFactor*q_next_state)
            state_1[1] = (1-self.learning_rate)*state_1[1] + self.learning_rate*(self.immedReward(pre_price, price, 0) + self.discountFactor*q_next_state)
            state_1[2] = (1-self.learning_rate)*state_1[2] + self.learning_rate*(self.immedReward(pre_price, price, 1) + self.discountFactor*q_next_state)
  
            if i % (10*SCALE) == 0:      
                a = max([(v,i) for i,v in enumerate(state_1)])
                action = a[1]

                error = abs(max(state_0)-max(state_1))

                message = 'Training step {} :=: price: {}, action: {}, error: {}.'.format(i, price, action, error)
                self.logger.debug(message)
    
  # train the model again after feeded by each new data point
  # return a list of actions
    def test(self):
        refer_Q_table = []
        
        message = 'testing data from {} to {}.'.format(
            self.preprocessor.UTC_time[self.test_start],
            self.preprocessor.UTC_time[self.test_end]
            )
        self.logger.info(message)

        for i in range(self.test_start, self.test_end):
            refer_Q_table = np.copy(self.Q_table)
  
            state_1 = self.Q_table[self.discretizor._day_rel[i - 1] + 1][self.discretizor._3days_rel[i - 1] + 1][self.discretizor._week_rel[i - 1] + 1][self.discretizor._month_rel[i - 1] + 1]
            state_2 = refer_Q_table[self.discretizor._day_rel[i] + 1][self.discretizor._3days_rel[i] + 1][self.discretizor._week_rel[i] + 1][self.discretizor._month_rel[i] + 1]
  
            pre_n_price = self.discretizor.price[i - 6] 
            pre_price = self.discretizor.price[i - 2]
            price = self.discretizor.price[i - 1]
      
            # exploit 
            n = randint(0,10)
            if n == 0:
                q_next_state = state_2[randint(0,2)]
            else:
                q_next_state = max(state_2)
      
            # update Q(s,a) of the (i-1)-th data
            state_1[0] = (1-self.learning_rate)*state_1[0] + self.learning_rate*(self.immedReward(pre_price, price, -1) + self.discountFactor*q_next_state)
            state_1[1] = (1-self.learning_rate)*state_1[1] + self.learning_rate*(self.immedReward(pre_price, price, 0) + self.discountFactor*q_next_state)
            state_1[2] = (1-self.learning_rate)*state_1[2] + self.learning_rate*(self.immedReward(pre_price, price, 1) + self.discountFactor*q_next_state)
  
            # choose action a corresponding to the largest Q(s,a) value
            m = max([(v,i) for i,v in enumerate(state_1)])
            m = m[1]
      
            self.test_outputs.append([i, m-1])

    def trade(self, initial_asset_USD = 10000.0, initial_asset_BTC = 0.0, split_ratio = 0.9):
        # strategy of using TBQ:
        # suppose have 1000 USD and 0 bitcoin initially
        # split the total asset according to the split_ratio, each time only put part of the asset into the BTC market
        account = {'USD':initial_asset_USD, 'BTC':initial_asset_BTC}
        initial_total = initial_asset_USD + initial_asset_BTC
        profit_rates = []

        i = 0
        for a in self.test_outputs:
            i = i+1
            data_index = a[0]
            action = a[1]
            price = self.discretizor.price[data_index]
            total_asset = account['USD'] + account['BTC']*price

            if action == -1: # if sell
                if account['USD']/total_asset < split_ratio:
                    account['BTC'] = total_asset*(1-split_ratio)/price
                    account['USD'] = total_asset*split_ratio
      
            if action == 1: # if buy
                if account['BTC']*price/total_asset < split_ratio:
                    account['USD'] = total_asset*(1-split_ratio)
                    account['BTC'] = total_asset*split_ratio/price
      
            profit_rate = ((total_asset - initial_total)/initial_total)*100
            profit_rates.append(profit_rate)
    
            message = 'Testing step {} :=: USD holding: {}, BTC holding: {}, price: {}, action: {}, profit rate: {}.'\
                        .format(i, account['USD'], account['BTC'], price, action, profit_rate)
            self.logger.debug(message)
        
        profit_rates = np.array(profit_rates)
        save_npy(profit_rates, self.save_dir, self.logger)
