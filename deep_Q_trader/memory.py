'''
This file keeps the memory of training.
'''

import random
import numpy as np
from deep_Q_trader.utils import save_npy, load_npy
from constants import *

class History:
    '''Experiance buffer of the behaniour policy of the agent'''

    def __init__(self, logger):
        self.logger = logger

        batch_size, history_length, self.num_channels = \
            BATCH_SIZE, HISTORY_LENGTH, NUM_CHANNELS

        self.dims = (self.num_channels,)
        self._history = np.zeros(
            [history_length, self.num_channels], dtype=np.float32)

    def add(self, screen):
        if screen.shape != self.dims:
            self.logger.error('INVALID TIMESTEP')
            
        self._history[:-1] = self._history[1:] # pop the first one
        self._history[-1] = screen # append at the end

    @property
    def history(self):
        return self._history



class ReplayMemory:
    '''Memory buffer for experiance replay'''

    def __init__(self, logger):
        self.logger = logger

        self._model_dir = "{}/logs/{}".format(PARENT_PATH, 'repley_memory')
        if not os.path.exists(self._model_dir):
            os.makedirs(self._model_dir)

        self.batch_size = BATCH_SIZE
        self.history_length = HISTORY_LENGTH
        self.memory_size = MEMORY_SIZE
        self.horizon = HORIZON
        self.num_channels = NUM_CHANNELS
        self.dims = (self.num_channels,)

        self.actions = np.empty(self.memory_size, dtype = np.uint8) # data type: 8-bit unsigned int
        self.rewards = np.empty(self.memory_size, dtype = np.float32)
        self.screens = np.empty((self.memory_size, NUM_CHANNELS), dtype = np.float32)
        self.terminals = np.empty(self.memory_size, dtype = np.bool) # flag of the terminal states
        self.trades_rem = np.empty(self.memory_size, dtype = np.float32) # (?)
        
        # pre-allocate prestates and poststates for minibatch
        self.prestates = (np.empty((self.batch_size, self.history_length, self.num_channels), 
                                        dtype = np.float32),\
                                        np.empty(self.batch_size, dtype=np.float32))
        self.poststates = (np.empty((self.batch_size, self.history_length, self.num_channels), 
                                        dtype = np.float32),\
                                        np.empty(self.batch_size, dtype=np.float32))
        
        self.count = 0 # number of data points before the most recent data point in memory
        self.current = 0 # index of the current data point
    
    def rest(self):
        self.actions = np.empty(self.memory_size, dtype = np.uint8) # data type: 8-bit unsigned int
        self.rewards = np.empty(self.memory_size, dtype = np.float32)
        self.screens = np.empty((self.memory_size, NUM_CHANNELS), dtype = np.float32)
        self.terminals = np.empty(self.memory_size, dtype = np.bool) # flag of the terminal states
        self.trades_rem = np.empty(self.memory_size, dtype = np.float32) # (?)
        
        # pre-allocate prestates and poststates for minibatch
        self.prestates = (np.empty((self.batch_size, self.history_length, self.num_channels), 
                                        dtype = np.float32),\
                                        np.empty(self.batch_size, dtype=np.float32))
        self.poststates = (np.empty((self.batch_size, self.history_length, self.num_channels), 
                                        dtype = np.float32),\
                                        np.empty(self.batch_size, dtype=np.float32))
        
        self.count = 0 # number of data points before the most recent data point in memory
        self.current = 0 # index of the current data point

        self.logger.info('Replay memory reset done.')

    def add(self, screen, reward, action, terminal, trade_rem):
        '''
        add a data point to the memory at the current index
        '''
        if screen.shape != self.dims:
            self.logger.error('INVALID_TIMESTEP')
            
        else:
            self.actions[self.current] = action
            self.rewards[self.current] = reward
            self.screens[self.current, ...] = screen
            self.terminals[self.current] = terminal
            self.trades_rem[self.current] = trade_rem
            self.count = max(self.count, self.current + 1)
            self.current = (self.current + 1) % self.memory_size

    def getState(self, index):
        '''
        return the previous history_length data points & number of data points remains (?) at the input index
        '''

        if self.count == 0:
            self.logger.error('REPLAY_MEMORY_ZERO')
            
        else:
            index = index % self.count
            if index >= self.history_length - 1:
                return self.screens[(index - (self.history_length - 1)):(index + 1), ...], \
                        self.trades_rem[index]
                        
            else:
                indexes = [(index - i) % self.count for i in reversed(range(self.history_length))]
                return self.screens[indexes, ...], self.trades_rem[index]

    def save(self):
        '''
        save memory
        '''
        message = "Saving replay memory to {}".format(self._model_dir)
        self.logger.info(message)
        for idx, (name, array) in enumerate(
            zip(['actions', 'rewards', 'screens', 'terminals', 'trades_rem', 'prestates', 'poststates'],
                [self.actions, self.rewards, self.screens, self.terminals, self.trades_rem, self.prestates, self.poststates])):
            for a in array:
                save_npy(a, os.path.join(self._model_dir, name), self.logger)

        message = "Replay memory successfully saved to {}".format(self._model_dir)
        self.logger.info(message)

    def load(self):
        '''
        load memory
        '''
        message = "Loading replay memory from {}".format(self._model_dir)
        self.logger.info(message)

        for idx, (name, array) in enumerate(
            zip(['actions', 'rewards', 'screens', 'terminals', 'trades_rem', 'prestates', 'poststates'],
                [self.actions, self.rewards, self.screens, self.terminals, self.trades_rem, self.prestates, self.poststates])):
            path = os.path.join(self._model_dir, name)
            if os.path.exists(path):
                array = load_npy(path, self.logger)
                message = "Replay memory {} successfully loaded from {}".format(name, path)
                self.logger.info(message)
            else:
                message = "Replay memory {} cannot be loaded from {}".format(name, path)
                self.logger.info(message)

        

    @property
    def model_dir(self):
        return self._model_dir
        
    @property
    def sample(self):
        '''
        sample a batch of data from the memory (data used before)
        '''
        if self.count <= self.history_length:
            self.logger.error('REPLAY_MEMORY_INSUFFICIENT')
        
        else:
            indexes = []
            while len(indexes) < self.batch_size:
                # find a random index 
                while True:
                    # sample one index (ignore states wraping over) 
                    index = random.randint(self.history_length, self.count - 1)
                    #print('get a random index from range: [{},{}].'.format(self.history_length, self.count - 1))
                    # if wraps over current pointer, then get new one
                    if index >= self.current and index - self.history_length < self.current:
                        print('Sample wraps over current pointer. The current index is {}.'.format(self.current))
                        continue
                    # otherwise use this index
                    break
                print('get a random index {} from memory.'.format(index))

                # having index first is fastest in C-order matrices
                self.prestates[0][len(indexes), ...], self.prestates[1][len(indexes)] = self.getState(index - 1)
                self.poststates[0][len(indexes), ...], self.poststates[1][len(indexes)] = self.getState(index)
                print('get the current state and the previuos one.')
                indexes.append(index)

            actions = self.actions[indexes]
            rewards = self.rewards[indexes]
            terminals = self.terminals[indexes]

            return self.prestates, actions, rewards, self.poststates, terminals
