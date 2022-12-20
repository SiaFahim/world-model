# Building the RNN model

# Importing the libraries

import numpy as np
import tensorflow as tf

# Building the MDN-RNN model within a class

class MDNRNN (object):
    # Initializing all the parameters and variables of the MDNRNN class
    def __init__(self, hps, reuse=False, gpu_mode=False):
        self.hps = hps
        with tf.variable_scope('mdn_rnn', reuse=reuse):
            if not gpu_mode:
                with tf.device('/cpu:0'):
                    tf.logging.info('Model using CPU.')
                    self.g = tf.Graph()
                    with self.g.as_default():
                        self._build_model(hps)
            else:
                tf.logging.info('Model using GPU.')
                self.g = tf.Graph()
                with self.g.as_default():
                    self._build_model(hps)
        self._init_session()

    # Making a method that creates the MDN-RNN model architecture itself
    def build_model(self, hps):
        # Building the RNN part of the MDN-RNN model
        self.num_mixture = hps.num_mixture
        KMIX = self.num_mixture
        INWIDTH = hps.input_seq_width
        OUTWIDTH = hps.output_seq_width
        LENGTH = self.hps.max_seq_len
        if hps.is_training:
            self.global_step = tf.Variable(0, name='global_step', trainable=False)
        cell_fn = tf.contrib.rnn.LayerNormBasicLSTMCell
        use_recurrent_dropout = False if self.hps.use_recurrent_dropout == 0 else True
        use_input_dropout = False if self.hps.use_input_dropout == 0 else True
        use_output_dropout = False if self.hps.use_output_dropout == 0 else True
        use_layer_norm = False if self.hps.use_layer_norm == 0 else True
        if use_recurrent_dropout:
            cell = cell_fn(hps.rnn_size, layer_norm=use_layer_norm, dropout_keep_prob=self.hps.recurrent_dropout_prob)
        else:
            cell = cell_fn(hps.rnn_size, layer_norm=use_layer_norm)
        if use_input_dropout:
            cell = tf.nn.rnn_cell.DropoutWrapper(cell, input_keep_prob=self.hps.input_dropout_prob)
        if use_output_dropout:
            cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=self.hps.output_dropout_prob)
        self.cell = cell
        self.sequence_lengths = LENGTH
        self.input_x = tf.placeholder(tf.float32, shape=[self.hps.batch_size, self.hps.max_seq_len, INWIDTH], name='input_x')
        self.output_x = tf.placeholder(tf.float32, shape=[self.hps.batch_size, self.hps.max_seq_len, OUTWIDTH], name='output_x')
        actual_input = self.input_x
        self.initial_state = cell.zero_state(batch_size=self.hps.batch_size, dtype=tf.float32)
        NOUT = OUTWIDTH * KMIX * 3
        with tf.variable_scope('RNN'):
            output_w = tf.get_variable("output_w", [self.hps.rnn_size, NOUT]) # Output weights
            output_b = tf.get_variable("output_b", [NOUT]) # Output biases
        output, last_state = tf.nn.dynamic_rnn(cell=cell,
                                                inputs=actual_input_x,
                                                initial_state=self.initial_state,
                                                dtype=tf.float32,
                                                swap_memory=True,
                                                scope='RNN')
        