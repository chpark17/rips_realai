#### This is the RIPS-HK 2018 Real-AI project ####

We implemented a Bitcoin trader using deep Q-learning, with a Q-network containing convlution layers and GRU layers.

We provide two modules: 
    buy_and_hold_trader
    deep_Q_trader
Hyperparameters can be set in constant.py

To run the whole project, simply use 
    python main.py
To run the deep Q trader only, use
    python run_DQT.py
To run the tabular Q trader, use 
    python run_TBQ.py

And the results are saved under ./result, run log and tensorboard file are saved under ./logs.

More details of this project are in RIPS_HK_2018_Final_Report.pdf
