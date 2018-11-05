'''
This file impelments the deep Q trading agent
'''

import sys
import time
import random
from tqdm import tqdm
import tensorflow as tf
import numpy as np

from deep_Q_trader.deepSense import DeepSense, DeepSenseParams, DropoutKeepProbs
from deep_Q_trader.environment import train_environment, test_environment
from deep_Q_trader.memory import History, ReplayMemory
from deep_Q_trader.utils import save_npy
from constants import *

# helper function
def clipped_error(x):
    # Huber loss
    try:
        return tf.where(tf.abs(x) < 1.0, 0.5 * tf.square(x), tf.abs(x) - 0.5)
    except:
        return tf.where(tf.abs(x) < 1.0, 0.5 * tf.square(x), tf.abs(x) - 0.5)

class Agent(object):
    '''
    Deep Trading Agent based on Deep Q Learning
    containing all the parameters for reinforcement learning
    '''
    '''TODO: 
        1. add `play` function to run tests in the simulated environment
    '''

    def __init__(self, sess, logger, train_env, test_env):
       
        self.logger = logger
        self._checkpoint_dir = "{}/logs/saved_models/{}".format(PARENT_PATH, 'checkpoints/')
        if not os.path.exists(self._checkpoint_dir):
            os.makedirs(self._checkpoint_dir)

        self.sess = sess
        self.logger = logger
        params = DeepSenseParams()

        self.train_env = train_env
        self.test_env = test_env
        self.history = History(logger)
        self.replay_memory = ReplayMemory(logger)
        # self.replay_memory.load()

        self.gamma = DISCOUNT_FACTOR

        self.max_step = 500 * SCALE
        self.target_q_update_step = 1 * SCALE
        self.learning_rate = 0.0025
        self.learning_rate_minimum = 0.0025
        self.learning_rate_decay = 0.96
        self.learning_rate_decay_step = 5 * SCALE

        self.ep_end = 0.1
        self.ep_start = 1.
        self.ep_end_t = 0.1*self.max_step

        self.train_frequency = 4
        self.learn_start = 5. * SCALE

        self.min_delta = -1
        self.max_delta = 1

        self.test_step = 50 * SCALE
        self.env_name = "btc_sim"
        self._saver = None

        self.save_dir = "{}/results/{}.npy".format(PARENT_PATH, 'DQT_profit_rates')

        with tf.variable_scope('steps'):
            self.step_op = tf.Variable(initial_value = 0, trainable=False, name='step')
            self.step_input = tf.placeholder('int32', None, name='step_input')
            # assign step_input to step_op
            self.step_assign_op = self.step_op.assign(self.step_input) 

        self.build_dqn(params)

    @property
    def checkpoint_dir(self):
        return self._checkpoint_dir

    @property
    def saver(self):
        if self._saver == None:
            self._saver = tf.train.Saver(max_to_keep=10)
        return self._saver

    def save_model(self, step=None):
        message = "Saving checkpoint to {}".format(self.checkpoint_dir)
        self.logger.info(message)
        self.saver.save(self.sess, self.checkpoint_dir, global_step=step)

    def load_model(self):
        message = "Loading checkpoint from {}".format(self.checkpoint_dir)
        self.logger.info(message)

        ckpt = tf.train.get_checkpoint_state(self.checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            fname = os.path.join(self.checkpoint_dir, ckpt_name)
            self.saver.restore(self.sess, fname)
            message = "Checkpoint successfully loaded from {}".format(fname)
            self.logger.info(message)
            return True
        else:
            message = "Checkpoint could not be loaded from {}".format(self.checkpoint_dir)
            self.logger.info(message)
            return False

    @property
    def summary_writer(self):
        return self._summary_writer

    
    def build_dqn(self, params):
        # build the Q network
        with tf.variable_scope('prediction'):
            self.s_t = tf.placeholder(
                dtype=tf.float32,
                shape=[None, self.replay_memory.history_length,  # None is going to be the batch_size
                            self.replay_memory.num_channels],
                name='historical_prices'
            )
            self.trade_rem_t = tf.placeholder(
                dtype=tf.float32,
                shape=[None,], # # None is going to be the batch_size
                name='trades_rem'
            )
            
            with tf.variable_scope('dropout_keep_probs'):
                self.q_conv_keep_prob = tf.placeholder(tf.float32)
                self.q_dense_keep_prob = tf.placeholder(tf.float32)
                self.q_gru_keep_prob = tf.placeholder(tf.float32)

        params.dropoutkeepprobs = DropoutKeepProbs(
                    conv_keep_prob = self.q_conv_keep_prob,
                    dense_keep_prob = self.q_dense_keep_prob,
                    gru_keep_prob = self.q_gru_keep_prob
                )

        self.q = DeepSense(params, self.logger, self.sess, name='q_network')
        self.q.build_model((self.s_t, self.trade_rem_t))

        # build the target Q network
        with tf.variable_scope('target'):
            self.t_s_t = tf.placeholder(
                dtype=tf.float32,
                shape=[None, self.replay_memory.history_length, # None is going to be the batch_size
                            self.replay_memory.num_channels],
                name='historical_prices'
            )
            self.t_trade_rem_t = tf.placeholder(
                dtype=tf.float32,
                shape=[None,], # None is going to be the batch_size
                name='trades_rem'
            )

        params.dropoutkeepprobs = DropoutKeepProbs()
        self.t_q = DeepSense(params, self.logger, self.sess, name='t_q_network')
        self.t_q.build_model((self.t_s_t, self.t_trade_rem_t))

        with tf.variable_scope('update_target_network'):
            self.q_weights_placeholders = {}
            self.t_weights_assign_ops = {}

            for name in self.q.weights.keys():
                self.q_weights_placeholders[name] = tf.placeholder(
                            tf.float32,
                            self.q.weights[name].get_shape().as_list()
                        )
            for name in self.q.weights.keys():
                self.t_weights_assign_ops[name] = self.t_q.weights[name].assign(
                    self.q_weights_placeholders[name]
                )

        with tf.variable_scope('training'):
            self.target_q = tf.placeholder(tf.float32, [None], name='target_q')
            self.action = tf.placeholder(tf.int64, [None], name='action')
            
            action_one_hot = tf.one_hot(self.action, NUM_ACTIONS, 
                                            1.0, 0.0, name='action_one_hot')
            q_acted = tf.reduce_sum(self.q.values * action_one_hot, 
                                        reduction_indices=1, name='q_acted')
                                        
            with tf.variable_scope('loss'):
                self.delta = self.target_q - q_acted

                self.global_step = tf.Variable(0, trainable=False)

                self.loss = tf.reduce_mean(clipped_error(self.delta), name='loss')

            with tf.variable_scope('optimizer'):
                self.learning_rate_step = tf.placeholder(tf.int64, None, name='learning_rate_step')
                self.learning_rate_op = tf.maximum(self.learning_rate_minimum,
                    tf.train.exponential_decay(
                        self.learning_rate,
                        self.learning_rate_step,
                        self.learning_rate_decay_step,
                        self.learning_rate_decay,
                        staircase=True))

                self.optimizer = tf.train.RMSPropOptimizer(
                    self.learning_rate_op, momentum=0.95, epsilon=0.01).minimize(self.loss)

        with tf.variable_scope('summary'):
            scalar_summary_tags = ['average.reward', 'average.loss', 'average.q', \
                'episode.max reward', 'episode.min reward', 'episode.avg reward', \
                'episode.num of episodes', 'training.learning_rate']            

            self.summary_placeholders = {}
            self.summary_ops = {}

            for tag in scalar_summary_tags:
                self.summary_placeholders[tag] = \
                    tf.placeholder('float32', None, name=tag.replace(' ', '_'))
                self.summary_ops[tag] = \
                    tf.summary.scalar(
                        name="{}-{}".format(self.env_name, tag.replace(' ', '_')),
                        tensor=self.summary_placeholders[tag]
                    )

            histogram_summary_tags = ['episode.rewards', 'episode.actions']
            for tag in histogram_summary_tags:
                self.summary_placeholders[tag] = \
                    tf.placeholder('float32', None, name=tag)
                self.summary_ops[tag] = \
                    tf.summary.histogram(
                        tag,
                        self.summary_placeholders[tag]
                    )

        self.sess.run(tf.local_variables_initializer())
        self.sess.run(tf.global_variables_initializer())
        # changed, not sure if it is right
        self._saver = tf.train.Saver(self.q.weights, max_to_keep=30)
        
        self.load_model()
        self.update_target_network()

        self._summary_writer = tf.summary.FileWriter('logs/tensorboard/')
        self._summary_writer.add_graph(self.sess.graph)



    def predict(self, state, test_ep = None): # test_ep = trade_rem ?
        s_t = state[0]
        trade_rem_t = state[1]

        # reduce the exploit rate (epsilon) when step increases
        ep = test_ep or (self.ep_end + 
            max(0., (self.ep_start - self.ep_end)
            * (self.ep_end_t - max(0., self.step - self.learn_start)) / self.ep_end_t))
        
        self.logger.debug('Step {}: epsilon is {}.'.format(self.step, ep))

        # random.random() return a float between 0.0 and 1.0
        if random.random() < ep: # exploit
            action = random.randrange(NUM_ACTIONS)
        else: # explore
            action = self.sess.run(
                fetches=self.q.action,
                feed_dict={
                    self.q.phase: 0,  
                    self.s_t: [s_t], 
                    self.trade_rem_t: [trade_rem_t],
                    self.q_conv_keep_prob: 1.0,
                    self.q_dense_keep_prob: 1.0,
                    self.q_gru_keep_prob: 1.0
                }
            )[0]

        return action
        

    def observe(self, screen, reward, action, terminal, trade_rem):
        #clip reward in the range min to max
        #reward = max(self.min_reward, min(self.max_reward, reward)) 
        
        self.history.add(screen)
        self.replay_memory.add(screen, reward, action, terminal, trade_rem)
        print('step {}: memeory added: {}, {}, {}, {}, {}'.format(self.step, screen, reward, action, terminal, trade_rem))

        if self.step > self.learn_start:
            if self.step % self.train_frequency == 0:
                print('step {}: do memory replay.'.format(self.step))
                self.q_learning_mini_batch()

            if self.step % self.target_q_update_step == self.target_q_update_step - 1:
                print('step {}: update target networks.'.format(self.step))
                self.update_target_network()

    def q_learning_mini_batch(self):
        '''
        doing memory replay & training
        '''
        if self.replay_memory.count >= self.replay_memory.history_length:
            print('step {}: sample from memory.'.format(self.step))
            state_t, action, reward, state_t_plus_1, terminal = self.replay_memory.sample
            s_t, trade_rem_t = state_t[0], state_t[1]
            s_t_plus_1, trade_rem_t_plus_1 = state_t_plus_1[0], state_t_plus_1[1]
            print('step {}: get q value of state t+1.'.format(self.step))
            q_t_plus_1 = self.sess.run(
                fetches=self.t_q.values,
                feed_dict={
                    self.t_q.phase: 0, 
                    self.t_s_t: s_t_plus_1, 
                    self.t_trade_rem_t: trade_rem_t_plus_1
                }
            )

            max_q_t_plus_1 = np.max(q_t_plus_1, axis=1)

            terminal = np.array(terminal) + 0.
            target_q = reward + self.gamma * (1 - terminal) * max_q_t_plus_1
            print('step {}: train the q network.'.format(self.step))
            _, q_t, loss, avg_q_summary = self.sess.run([self.optimizer, self.q.values, self.loss, self.q.avg_q_summary], {
                self.q.phase: 1,
                self.target_q: target_q,
                self.action: action,
                self.s_t: s_t,
                self.trade_rem_t: trade_rem_t,
                self.q_conv_keep_prob: CONV_KEEP_PROB,
                self.q_dense_keep_prob: DENSE_KEEP_PROB,
                self.q_gru_keep_prob: GRU_KEEP_PROB,
                self.learning_rate_step: self.step
            })

            self.summary_writer.add_summary(avg_q_summary, self.step)
            self.total_loss += loss
            self.total_q += q_t.mean()
            self.update_count += 1

    
    def update_target_network(self):
        for name in self.q.weights.keys():
            self.sess.run(
                fetches=self.t_weights_assign_ops[name],
                feed_dict=
                {self.q_weights_placeholders[name]: self.sess.run(
                    fetches=self.q.weights[name]
                )}
            )
    
    def inject_summary(self, tag_dict, step):
        summary_str_lists = self.sess.run([self.summary_ops[tag] for tag in tag_dict.keys()], {
            self.summary_placeholders[tag]: value for tag, value in tag_dict.items()
        })
        for summary_str in summary_str_lists:
            self.summary_writer.add_summary(summary_str, self.step)

    def train(self):
        start_step = self.sess.run(self.step_op)
        print('start_step: {}'.format(start_step))

        message = 'training data from {} to {}.'.format(
            self.train_env.processor.UTC_time[self.train_env.start_index],
            self.train_env.processor.UTC_time[self.train_env.end_index]
            )
        self.logger.info(message)

        num_episodes, self.update_count, ep_reward = 0, 0, 0.
        total_reward, self.total_loss, self.total_q = 0., 0., 0.
        max_avg_ep_reward = 0
        ep_rewards, actions = [], []
        
        # selection a new episode from the unused data
        trade_rem = self.train_env.new_random_episode(self.history, self.replay_memory)

        for self.step in range(start_step, self.max_step):
            # self.step acts like an integer in the range [start_step, max_step]
            # ncols: The width of the entire output message
            # initial: The initial counter value. Useful when restarting a progress bar [default: 0].
            if self.step == self.learn_start:
                # reset everything
                num_episodes, self.update_count, ep_reward = 0, 0, 0.
                total_reward, self.total_loss, self.total_q = 0., 0., 0.
                ep_rewards, actions = [], []


            # 1. predict
            action = self.predict((self.history.history, trade_rem))
            print('step {}: predict done.'.format(self.step))
            # 2. act
            screen, reward, terminal, trade_rem = self.train_env.act(action)
            print('step {}: act done.'.format(self.step))
            # 3. observe
            self.observe(screen, reward, action, terminal, trade_rem)
            print('step {}: observe done.'.format(self.step))

            if terminal:
                self.train_env.new_random_episode(self.history, self.replay_memory)
                num_episodes += 1
                ep_rewards.append(ep_reward)
                ep_reward = 0.

            else:
                ep_reward += reward

            actions.append(action)
            total_reward += reward
            
            if self.step >= self.learn_start:
                if self.step % self.test_step == self.test_step - 1: # write the training result to log
                    avg_reward = total_reward / self.test_step
                    avg_loss = self.total_loss / self.update_count
                    avg_q = self.total_q / self.update_count

                    try:
                        max_ep_reward = np.max(ep_rewards)
                        min_ep_reward = np.min(ep_rewards)
                        avg_ep_reward = np.mean(ep_rewards)
                    except:
                        max_ep_reward, min_ep_reward, avg_ep_reward = 0, 0, 0

                    message = 'avg_r: %.4f, avg_l: %.6f, avg_q: %3.6f, avg_ep_r: %.4f, max_ep_r: %.4f, min_ep_r: %.4f, # episodes: %d' \
                        % (avg_reward, avg_loss, avg_q, avg_ep_reward, max_ep_reward, min_ep_reward, num_episodes)
                    self.logger.info(message)

                    self.sess.run(
                        fetches=self.step_assign_op,
                        feed_dict={self.step_input: self.step + 1}
                    )
                    self.save_model(self.step + 1)

                    # self.replay_memory.save()

                    max_avg_ep_reward = max(max_avg_ep_reward, avg_ep_reward)

                    if self.step > 180:
                        self.inject_summary({
                            'average.reward': avg_reward,
                            'average.loss': avg_loss,
                            'average.q': avg_q,
                            'episode.max reward': max_ep_reward,
                            'episode.min reward': min_ep_reward,
                            'episode.avg reward': avg_ep_reward,
                            'episode.num of episodes': num_episodes,
                            'episode.rewards': ep_rewards,
                            'episode.actions': actions,
                            'training.learning_rate': self.sess.run(
                                fetches=self.learning_rate_op,
                                feed_dict={self.learning_rate_step: self.step}
                            )
                        }, self.step)
                    # reset
                    num_episodes = 0
                    total_reward = 0.
                    self.total_loss = 0.
                    self.total_q = 0.
                    self.update_count = 0
                    ep_reward = 0.
                    ep_rewards = []
                    actions = []
        
    
    def test(self):
        start_step = self.test_env.start_index
        end_step = self.test_env.end_index
        print('start_step: {}'.format(start_step))

        message = 'testinging data from {} to {}.'.format(
            self.test_env.processor.UTC_time[self.test_env.start_index],
            self.test_env.processor.UTC_time[self.test_env.end_index]
            )
        self.logger.info(message)

        num_data, self.update_count = 0, 0
        total_reward, self.total_loss, self.total_q = 0., 0., 0.

        actions = []
        profit_rates = []
        
        # selection a new episode from the unused data
        trade_rem = self.test_env.new_data_point(self.history, self.replay_memory)

        for self.step in range(start_step + 1, end_step + 1):
            # self.step acts like an integer in the range [start_step, max_step]
            # ncols: The width of the entire output message
            # initial: The initial counter value. Useful when restarting a progress bar [default: 0].

            # 1. predict
            action = self.predict((self.history.history, trade_rem), test_ep = 0.0)
            print('step {}: predict done.'.format(self.step))
            # 2. act
            screen, reward, terminal, trade_rem, profit_rate = self.test_env.act(action)
            print('step {}: act done.'.format(self.step))
            
            profit_rates.append(profit_rate)

            if terminal == True:
                print(message)
                profit_rates = np.array(profit_rates)
                save_npy(profit_rates, self.save_dir, self.logger)
            
            # 3. observe
            self.observe(screen, reward, action, terminal, trade_rem)
            print('step {}: observe done.'.format(self.step))

            actions.append(action)
            total_reward += reward
            
            if self.step % self.test_step == self.test_step - 1:
                avg_reward = total_reward / self.test_step
                avg_loss = self.total_loss / self.update_count
                avg_q = self.total_q / self.update_count

                message = 'avg_r: %.4f, avg_l: %.6f, avg_q: %3.6f, # data: %d' \
                    % (avg_reward, avg_loss, avg_q, num_data)
                self.logger.info(message)

                if self.step > 180:
                    self.inject_summary({
                            'average.reward': avg_reward,
                            'average.loss': avg_loss,
                            'average.q': avg_q,
                            'episode.max reward': 0.0,
                            'episode.min reward': 0.0,
                            'episode.avg reward': 0.0,
                            'episode.num of episodes': num_data,
                            'episode.rewards': 0.0,
                            'episode.actions': actions,
                            'training.learning_rate': self.sess.run(
                                fetches=self.learning_rate_op,
                                feed_dict={self.learning_rate_step: self.step}
                            )
                        }, self.step)

                num_data = 0
                total_reward = 0.
                self.total_loss = 0.
                self.total_q = 0.
                self.update_count = 0
                actions = []

            self.test_env.new_data_point(self.history, self.replay_memory)