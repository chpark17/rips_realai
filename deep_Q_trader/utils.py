'''
Utilities.
'''

import sys
import logging
import matplotlib.pyplot as plt
import numpy as np
from constants import *

def get_logger():

    formatter = logging.Formatter(logging.BASIC_FORMAT)   
    info_handler = logging.FileHandler( "{}/logs/{}.log".format(PARENT_PATH, 'run'))
    info_handler.setLevel(logging.DEBUG)
    info_handler.setFormatter(formatter)

    out_handler = logging.StreamHandler(sys.stdout)
    out_handler.setLevel(logging.INFO)
    out_handler.setFormatter(formatter)

    logger = logging.getLogger(name = 'deep_Q_trader')
    logger.setLevel(logging.DEBUG)
    logger.addHandler(info_handler)
    logger.addHandler(out_handler)
    
    return logger

def save_npy(obj, path, logger):
    np.save(path, obj)
    message = "  [*] saved at {}".format(path)
    logger.info(message)

def load_npy(path, logger):
    obj = np.load(path)
    message = "  [*] loaded from {}".format(path)
    logger.info(message)
    return obj

def plot_profit_rates_DQT_TBQ_BH(DQT, TBQ, BH, file_name):
    
    plt.plot(DQT, 'r-')
    plt.plot(TBQ, 'b-')
    plt.plot(BH, 'g-')
    plt.xlabel('time (h)')
    plt.ylabel('profit rate (%)')
    plt.title('profit rate of DQT, TBQ and BH')

    path = "{}/results/{}.png".format(PARENT_PATH, file_name)
    plt.savefig(path)

def plot_profit_rates_TBQ_BH(TBQ, BH, file_name):
    
    plt.plot(TBQ, 'b-')
    plt.plot(BH, 'g-')
    plt.xlabel('time (h)')
    plt.ylabel('profit rate (%)')
    plt.title('profit rate of TBQ and BH')

    path = "{}/results/{}.png".format(PARENT_PATH, file_name)
    plt.savefig(path)

def plot_profit_rates_DQT_BH(DQT, BH, file_name):
    
    plt.plot(DQT, 'r-')
    plt.plot(BH, 'g-')
    plt.xlabel('time (h)')
    plt.ylabel('profit rate (%)')
    plt.title('profit rate of DQT and BH')

    path = "{}/results/{}.png".format(PARENT_PATH, file_name)
    plt.savefig(path)

def printResult(logger, result):
    maximum = max(result)
    minimum = min(result)
    std = np.std(result)
    message = "max: {}, min: {}, std: {}.".format(maximum, minimum, std)
    logger.info(message)