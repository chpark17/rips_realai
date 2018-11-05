'''
This file keeps some constants used through out the program.
'''
import os

# dataset

# __file__: point to the filename of the current module
# os.path.abspath(): turn that into an absolute path
# os.path.dirname(): remove the last segment of the path
PARENT_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = 'MACD_48-12'

BATCH_SIZE = 50
HISTORY_LENGTH = 180
HORIZON = 24
TEST_FRAC = 0.3
MEMORY_SIZE = 100000
#price difference between this and the last hour, hig and low in the last half an hour, count of docs contain word 'hack' in reddit per hour, MACD
NUM_CHANNELS = 5 
SPLIT_SIZE = 9
WINDOW_SIZE = 20

# actions
NUM_ACTIONS = 3
LONG = 1
NEUTRAL = 0
SHORT = -1

# dropout
CONV_KEEP_PROB = 0.9
DENSE_KEEP_PROB = 0.5
GRU_KEEP_PROB = 0.5

# convolution
FILTER_SIZES = [10, 20]
KERNEL_SIZES = [5, 3]
PADDING = 'SAME'

# gru
GRU_CELL_SIZE = 128
GRU_NUM_CELLS = 2

# dense
DENSE_LAYER_SIZES = [128, 64, 32]

# Q learning
DISCOUNT_FACTOR = 0.8
SCALE = 100 # deep Q
LEARNING_RATE = 0.7 # tabular




