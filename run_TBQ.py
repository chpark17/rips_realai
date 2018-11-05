'''
Only run the tabular Q trader and compare it with the buy and hold agent.
'''

from deep_Q_trader.utils import get_logger, load_npy, plot_profit_rates_TBQ_BH, printResult
from tabular_Q_trader.processor import Preprocessor
from buy_and_hold_trader.agent import buy_and_hold
from tabular_Q_trader.discretizor import discretizor
from tabular_Q_trader.agent import tabular_Q

logger = get_logger()
preprocessor = Preprocessor(logger)
dis = discretizor(logger, preprocessor)

TBQ_agent = tabular_Q(logger, preprocessor, dis)
logger.info("***TBQ Training starts.***")
TBQ_agent.train()
logger.info("***TBQ Training ends. Testing starts.***")
TBQ_agent.test()
TBQ_agent.trade()
logger.info("***TBQ Testing ends.***")


BH_agent = buy_and_hold(logger, preprocessor)
logger.info("***BH starts.***")
BH_profit_rates = BH_agent.trade()
logger.info("***BH ends.***")

TBQ_profit_rates = load_npy(TBQ_agent.save_dir, logger)
BH_profit_rates = load_npy(BH_agent.save_dir, logger)

logger.info("BH results:")
printResult(logger, BH_profit_rates)

logger.info("TBQ testing results:")
printResult(logger, TBQ_profit_rates)

plot_profit_rates_TBQ_BH(TBQ_profit_rates, BH_profit_rates, 'TBQ_BH_profit_rates_per_half_hour')
logger.info("***Results saved.***")