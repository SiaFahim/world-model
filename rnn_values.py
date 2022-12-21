# MDN-RNN with all the valuse
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # changing the verbosity to emmit some warnings
# importing the libraries
import numpy as np
import tensorflow as tf
from tensorflow import keras
from collections import namedtuple

# Setting the Hyperparameters
HyperParams = namedtuple('HyperParams', ['num_steps',
                                         'max_seq_len',
                                         'input_seq_width',
                                         'output_seq_width',
                                         'rnn_size',
                                         'batch_size',
                                         'grad_clip',
                                         'num_mixture',
                                         'learning_rate',
                                         'decay_rate',
                                         'min_learning_rate',
                                         'use_layer_norm',
                                         'use_recurrent_dropout',
                                         'recurrent_dropout_prob',
                                         'use_input_dropout',
                                         'input_dropout_prob',
                                         'use_output_dropout',
                                         'output_dropout_prob',
                                         'is_training',
                                         ])

# Making a function the returns all the values of the default hyperparameters
def get_default_hps():
    return HyperParams(num_steps=2000,
                       max_seq_len=1000,
                       input_seq_width=35,
                       output_seq_width=32,
                       rnn_size=256,
                       batch_size=100,
                       grad_clip= 1.0,
                       num_mixture=5,
                       learning_rate=0.001,
                       decay_rate=1.0,
                       min_learning_rate=0.00001,
                       use_layer_norm=0,
                       use_recurrent_dropout=0,
                       recurrent_dropout_prob=0.90,
                       use_input_dropout=0,
                       input_dropout_prob=0.90,
                       use_output_dropout=0,
                       output_dropout_prob=0.90,
                       is_training=1,
                       )

# Getting these default hyperparameters
hps = get_default_hps()

# Building the RNN
KMIX = hps.num_mixture
INWIDTH = hps.input_seq_width
OUTWIDTH = hps.output_seq_width
LENGTH = hps.max_seq_len
if hps.is_training:
    global_step = tf.Variable(0, name='global_step', trainable=False)
cell_fn = tf.contrib.rnn.LayerNormBasicLSTMCell
use_recurrent_dropout = False if hps.use_recurrent_dropout == 0 else True
use_input_dropout = False if hps.use_input_dropout == 0 else True
use_output_dropout = False if hps.use_output_dropout == 0 else True
use_layer_norm = False if hps.use_layer_norm == 0 else True
if use_recurrent_dropout:
    cell = cell_fn(hps.rnn_size, layer_norm=use_layer_norm, dropout_keep_prob=hps.recurrent_dropout_prob)
else:
    cell = cell_fn(hps.rnn_size, layer_norm=use_layer_norm)
if use_input_dropout:
    cell = tf.nn.rnn_cell.DropoutWrapper(cell, input_keep_prob=hps.input_dropout_prob)
if use_output_dropout:
    cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=hps.output_dropout_prob)
sequence_lengths = LENGTH
input_x = tf.placeholder(tf.float32, shape=[hps.batch_size, hps.max_seq_len, INWIDTH], name='input_x')
output_x = tf.placeholder(tf.float32, shape=[hps.batch_size, hps.max_seq_len, OUTWIDTH], name='output_x')
actual_input = input_x
initial_state = cell.zero_state(batch_size=hps.batch_size, dtype=tf.float32)
NOUT = OUTWIDTH * KMIX * 3
with tf.variable_scope('RNN'):
    output_w = tf.get_variable("output_w", [hps.rnn_size, NOUT]) # Output weights
    output_b = tf.get_variable("output_b", [NOUT]) # Output biases
output, last_state = tf.nn.dynamic_rnn(cell=cell,
                                        inputs=actual_input,
                                        initial_state=initial_state,
                                        dtype=tf.float32,
                                        swap_memory=True,
                                        scope='RNN')
output = tf.reshape(output, [-1, hps.rnn_size])
output = tf.nn.xw_plus_b(output, output_w, output_b)
output = tf.reshape(output, [hps.batch_size, -1, KMIX * 3])
print(output)