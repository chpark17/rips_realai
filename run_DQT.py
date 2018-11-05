'''
Only run the deep Q trader and compare it with the buy and hold agent.
'''

import time
from os.path import join

import tensorflow as tf
import numpy as np

from deep_Q_trader.agent import Agent
from deep_Q_trader.environment import train_environment, test_environment
from deep_Q_trader.utils import get_logger, plot_profit_rates_DQT_BH, plot_profit_rates_TBQ_BH, plot_profit_rates_DQT_TBQ_BH
from deep_Q_trader.processor import Processor
from buy_and_hold_trader.agent import buy_and_hold

def main():
    logger = get_logger()
    processor = Processor(logger)

    with tf.Session() as sess:
        
        train_env = train_environment(logger, processor)
        test_env = test_environment(logger, processor)
        agent = Agent(sess, logger, train_env, test_env)
        logger.info("***DQT Training starts.***")
        agent.train()
        logger.info("***DQT Training ends. Testing starts.***")
        DQT_profit_rates = agent.test()
        logger.info("***DQT Testing ends.***")
        agent.summary_writer.close()

    BH_agent = buy_and_hold(logger, processor)
    logger.info("***BH starts.***")
    BH_profit_rates = BH_agent.trade()
    logger.info("***BH ends.***")

    plot_profit_rates_DQT_BH(DQT_profit_rates, BH_profit_rates, 'DQT_BH_profit_rates_per_half_hour')
    logger.info("***Results saved.***")

if __name__ == "__main__":
    main()
