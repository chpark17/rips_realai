'''
RIPS-HK 2018, RealAI project.
Use deep Q network to implement a bitcoin trader.
'''

import time

import tensorflow as tf
import numpy as np

from deep_Q_trader.agent import Agent
from deep_Q_trader.environment import train_environment, test_environment
from deep_Q_trader.utils import get_logger, load_npy, plot_profit_rates_DQT_BH, plot_profit_rates_TBQ_BH, plot_profit_rates_DQT_TBQ_BH, printResult
from deep_Q_trader.processor import Processor
from buy_and_hold_trader.agent import buy_and_hold
from tabular_Q_trader.discretizor import discretizor
from tabular_Q_trader.agent import tabular_Q

def main():
    logger = get_logger()
    processor = Processor(logger)
    dis = discretizor(logger, processor)

    with tf.Session() as sess:
        
        train_env = train_environment(logger, processor)
        test_env = test_environment(logger, processor)
        agent = Agent(sess, logger, train_env, test_env)
        #logger.info("***DQT Training starts.***")
        #agent.train()
        logger.info("***DQT Training ends. Testing starts.***")
        agent.test()
        logger.info("***DQT Testing ends.***")
        DQT_profit_rates = load_npy(agent.save_dir, logger)
        
        agent.summary_writer.close()
    
    TBQ_agent = tabular_Q(logger, processor, dis)
    logger.info("***TBQ Training starts.***")
    TBQ_agent.train()
    logger.info("***TBQ Training ends. Testing starts.***")
    TBQ_agent.test()
    TBQ_agent.trade()
    logger.info("***TBQ Testing ends.***")
    TBQ_profit_rates = load_npy(TBQ_agent.save_dir, logger)
    

    BH_agent = buy_and_hold(logger, processor)
    logger.info("***BH starts.***")
    BH_agent.trade()
    logger.info("***BH ends.***")
    BH_profit_rates = load_npy(BH_agent.save_dir, logger)
    
    logger.info("***Testing results***")
    logger.info("TBQ testing results:")
    printResult(logger, TBQ_profit_rates)

    logger.info("DQT testing results:")
    printResult(logger, DQT_profit_rates)

    logger.info("BH results:")
    printResult(logger, BH_profit_rates)

    plot_profit_rates_DQT_BH(DQT_profit_rates, BH_profit_rates, 'DQT_BH_profit_rates_per_half_hour_26_12')
    plot_profit_rates_DQT_TBQ_BH(DQT_profit_rates, TBQ_profit_rates, BH_profit_rates, 'DQT_TBQ_BH_profit_rates_per_half_hour_26_12')
    logger.info("***Results saved.***") 

if __name__ == "__main__":
    main()
