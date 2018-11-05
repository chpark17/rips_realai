'''
This file build the DeepSense network (unified deep learning network for timeseries data).
more info: https://arxiv.org/abs/1611.01942
'''

from os.path import join
import tensorflow as tf
from constants import *


class DropoutKeepProbs:
    '''Defines the keep probabilities for different dropout layers'''

    def __init__(self, conv_keep_prob=1.0, dense_keep_prob=1.0, gru_keep_prob=1.0):
        self.conv_keep_prob = conv_keep_prob
        self.dense_keep_prob = dense_keep_prob
        self.gru_keep_prob = gru_keep_prob

class DeepSenseParams:
    '''Defines the parameters for the DeepSense Q Network Architecture'''

    def __init__(self, dropoutkeeprobs = None):
        #Timeseries Parameters
        self.num_actions = NUM_ACTIONS
        self.num_channels = NUM_CHANNELS
        self.split_size = SPLIT_SIZE
        self.window_size = WINDOW_SIZE

        #Dropout Layer Parameters
        self._dropoutkeeprobs = dropoutkeeprobs

        #Convolutional Layer Parameters
        self.filter_sizes = FILTER_SIZES # a list with length 2
        self.kernel_sizes = KERNEL_SIZES # a list with length 2
        self.padding = PADDING

        #GRU Parameters
        self.gru_cell_size = GRU_CELL_SIZE
        self.gru_num_cells = GRU_NUM_CELLS

        #FullyConnected Network Parameters
        self.dense_layer_sizes = DENSE_LAYER_SIZES

    @property
    def dropoutkeepprobs(self):
        return self._dropoutkeeprobs

    @dropoutkeepprobs.setter
    def dropoutkeepprobs(self, value):
        self._dropoutkeeprobs = value



class DeepSense:
    '''DeepSense Architecture for Q function approximation over Timeseries'''

    def __init__(self, deepsenseparams, logger, sess, name='DeepSense'):
        self.params = deepsenseparams
        self.logger = logger
        self.sess = sess
        self.__name__ = name

        self._weights = None

    @property
    def action(self):
        return self._action

    @property
    def avg_q_summary(self):
        return self._avg_q_summary

    @property
    def params(self):
        return self._params
    
    @params.setter
    def params(self, value):
        self._params = value

    @property
    def name(self):
        return self.__name__

    @property
    def values(self):
        return self._values

    @property
    def weights(self):
        if self._weights is None:
            self._weights = {}
            variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 
                                            scope=self.__name__)
            for variable in variables:
                name = "/".join(variable.name.split('/')[1:])
                self._weights[name] = variable
        return self._weights

    def conv2d_layer(self, inputs, filter_size, kernel_size, padding, name, reuse, activation=None):
        return tf.layers.conv2d(
                        inputs=inputs, 
                        filters=filter_size, # an int, number of filters. i.e. the size of outputs
                        kernel_size=[1, kernel_size], # filter (window) size with the first dimention fixed to 1
                        strides=(1, 1), # move the filter by 1 each time
                        padding=padding, # a string, "valid" or "same" (case-insensitive)
                        activation=activation, # activation function
                        name=name, # a string
                        reuse=reuse # a boolean, whether to reuse the weights of a previous layer by the same name
                    )

    def dense_layer(self, inputs, num_units, name, reuse, activation=None):
        output = tf.layers.dense(
                        inputs=inputs,
                        units=num_units, # an int or long, the size of outputs
                        activation=activation, 
                        name=name,
                        reuse=reuse
                    )
        return output

    # ignore units during the training phase of certain set of neurons which is chosen at random
    # At each training stage, individual nodes are either dropped out of the net with probability 1-p or kept with probability p
    # to prevent over-fitting
    def dropout_layer(self, inputs, keep_prob, name, is_conv=False):
        if is_conv:
            channels = tf.shape(inputs)[-1] # the last enrty of the input shape is the number of channels
            return tf.nn.dropout(
                            x = inputs, # a floating point tensor
                            keep_prob=keep_prob, # a scalar Tensor with the same type as x. The probability that each element is kept.
                            name=name,
                            noise_shape=[
                                self.batch_size, 1, 1, channels
                            ] # A 1-D Tensor of type int32, representing the shape for randomly generated keep/drop flags.
                        )
        else:
            return tf.nn.dropout(
                        inputs,
                        keep_prob=keep_prob,
                        name=name
                    )        

    def build_model(self, state, reuse = False): # reuse's default value changed from false to ture (!) 
        inputs = state[0] # size of state[0]: [batch_size, history_length, num_channels]
        trade_rem = state[1] # shape of state[1]: [batch_size,]

        with tf.variable_scope(self.__name__, reuse=reuse):
            with tf.name_scope('phase'): # create a new variable named 'phase' in the computational graph
                self.phase = tf.placeholder(dtype=tf.bool) # fed with boolean inputs (in any shape) and output a tensor 

            with tf.variable_scope('input_params', reuse=reuse): # create a new variable named 'input_params' in the computational graph
                self.batch_size = tf.shape(inputs)[0] # the length of the inputs, not fixed
            
            # step 0: reshape the inputs
            inputs = tf.reshape(inputs, 
                        shape=[self.batch_size, 
                                self.params.split_size, 
                                self.params.window_size, # history_length = split_size*window_size
                                self.params.num_channels]) # shape of a single input: [split_size,window_size,num_channels]

            with tf.variable_scope('conv_layers', reuse=reuse): # create a new variable named 'conv_layers' in the computational graph
                window_size = self.params.window_size 
                num_convs = len(self.params.filter_sizes)
                for i in range(0, num_convs): # create num_convs convolutional layers in the computational graph
                    with tf.variable_scope('conv_layer_{}'.format(i + 1), reuse=reuse):
                        window_size = window_size - self.params.kernel_sizes[i] + 1
                        # feed the inputs to a convolutional layer
                        inputs = self.conv2d_layer(inputs, self.params.filter_sizes[i], 
                                                    self.params.kernel_sizes[i], 
                                                    self.params.padding,
                                                    'conv_{}'.format(i + 1), 
                                                    reuse,
                                                    activation=tf.nn.relu)
                        # feed the output of the previous convolutional layer to a dropout layer                       
                        inputs = self.dropout_layer(inputs,
                                                    self.params.dropoutkeepprobs.conv_keep_prob, 
                                                    'dropout_conv_{}'.format(i + 1),
                                                    is_conv=True)
                                                                           
            if self.params.padding == 'VALID':
                inputs = tf.reshape(inputs, 
                                    shape=[
                                            self.batch_size, 
                                            self.params.split_size, 
                                            window_size * self.params.filter_sizes[-1]
                                        ]
                            )
            else:
                inputs = tf.reshape(inputs, 
                                    shape=[
                                            self.batch_size, 
                                            self.params.split_size, 
                                            self.params.window_size * self.params.filter_sizes[-1]
                                        ]
                            )

            # GRUs: Gated recurrent units, do not have an output gate
            # similar to LSTM: long short-term memory, have an output gate
            gru_cells = [] # create a list of gru cells, with a length of (gru_num_cells-1)
            for i in range(0, self.params.gru_num_cells): 
                cell = tf.nn.rnn_cell.GRUCell( # do not have an output gate
                    num_units=self.params.gru_cell_size, # size of output of one cell
                    reuse=reuse # whether to reuse variables in an existing scope
                )        
                # details: https://arxiv.org/abs/1512.05287
                # Create a cell with added input, state, and output dropout
                # State dropout is performed on the outgoing states of the cell
                cell = tf.nn.rnn_cell.DropoutWrapper( 
                    cell, 
                    output_keep_prob=self.params.dropoutkeepprobs.gru_keep_prob, 
                    variational_recurrent=True, # the same dropout mask is applied at every step (input, state, and output)
                    dtype=tf.float32 # The dtype of the input, state, and output tensors. Required and used iff variational_recurrent = True
                )
                gru_cells.append(cell)

            # Create a RNN cell composed sequentially of a number of RNNCells
            # accepted and returned states are n-tuples, where n = len(cells) = gru_num_cells
            multicell = tf.nn.rnn_cell.MultiRNNCell(cells = gru_cells)

            # create a RNN layer named 'dynamic_unrolling' in the computational graph
            with tf.name_scope('dynamic_unrolling'): 
                # Creates a recurrent neural network specified by 'cell'
                output, final_state = tf.nn.dynamic_rnn( 
                    cell=multicell,
                    inputs=inputs,
                    dtype=tf.float32
                )
                # shape of the output: [batch_size, gru_cell_size, gru_num_cells]

            # the outputs of the last RNN state in the order of the gru cells
            output = tf.unstack(output, axis=1)[-1] 
            # only take the last output tensor
            # shape of the output tensor: [batch_size, gru_num_cells]

            ''' 
            Append the information regarding the number of trades left in the episode
            '''
            # trade_rem = state[1]: trades remaining in the episode
            trade_rem = tf.expand_dims(trade_rem, axis=1) 
            # shape of trade_rem: [batch_szie, 1] ?
            output = tf.concat([output, trade_rem], axis=1)
            # shape of the output: [batch_size, gru_num_cells  + 1] ?

            # create a new set of layers named 'fully_connected' in the computational graph 
            with tf.variable_scope('fully_connected', reuse=reuse): 
                num_dense_layers = len(self.params.dense_layer_sizes)
                for i in range(0, num_dense_layers): 
                    with tf.variable_scope('dense_layer_{}'.format(i + 1), reuse=reuse):
                        # create a new layer with 1 dense layer and 1 dropout layer
                        output = self.dense_layer(output, self.params.dense_layer_sizes[i], 
                                                    'dense_{}'.format(i + 1), reuse, activation=tf.nn.relu)                    
                        output = self.dropout_layer(output,
                                                    self.params.dropoutkeepprobs.dense_keep_prob,
                                                    'dropout_dense_{}'.format(i + 1))
            # the output layer           
            self._values = self.dense_layer(output, self.params.num_actions, 'q_values', reuse)
            # shape of the output: [batch_size, num_actions]
            
            with tf.name_scope('avg_q_summary'):
                # compute the average among the Q values of each data
                avg_q = tf.reduce_mean(self._values, axis=0)
                self._avg_q_summary = []
                for idx in range(self.params.num_actions):
                    self._avg_q_summary.append(tf.summary.histogram('q/{}'.format(idx), avg_q[idx]))
                self._avg_q_summary = tf.summary.merge(self._avg_q_summary, name='avg_q_summary')
            
            # Returns the index with the largest value across dimensions of a tensor
            self._action = tf.argmax(self._values, dimension=1, name='action')
            # a 1d list with length batch_size
