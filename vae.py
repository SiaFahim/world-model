# building the VAE model

# importing the libraries 

import numpy as np
import tensorfloe as tf

# building the VAE model withibn a class

class ConvVAE(object):
    # Initializing all the parameters and variables of the VonvVAE class
    def __init__(self, z_size=32, batch_size=1, learning_rate=0.0001, kl_tolerance=0.5, is_training=False, reuse=False, gpu_mode=False):
        self.z_size = z_size
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.kl_tolerance = kl_tolerance
        self.is_training = is_training
        self.reuse = reuse
        with tf.variable_scope('conv_vae', reuse=self.reuse):
            if not gpu_mode:
                with tf.device('/cpu:0'):
                    tf.logging.info('Model using CPU.')
                    self._build_graph()
            else:
                tf.logging.info('Model using GPU.')
                self._build_graph()
        self._init_session() 

    # Making a method that creates the VAE model architecture itself
    def _build_graph(self):
        self.g = tf.Graph()
        with self.g.as_default():
            self.x = tf.placeholder(tf.float32, [None, 64, 64, 3], name='x')
            
            # Building the encoder part of VAE
            h = tf.layers.conv2d(self.x, 32, 4, strides=2, activation=tf.nn.relu, name='enc_conv1')
            h = tf.layers.conv2d(h, 64, 4, strides=2, activation=tf.nn.relu, name='enc_conv2')
            h = tf.layers.conv2d(h, 128, 4, strides=2, activation=tf.nn.relu, name='enc_conv3')
            h = tf.layers.conv2d(h, 256, 4, strides=2, activation=tf.nn.relu, name='enc_conv4')
            h = tf.reshape(h, [-11*2*256])

            # Building the V part of the VAE
            self.mu = tf.layers.dense(h, self.z_size, name='enc_fc_mu')
            self.logvar = tf.layers.dense(h, self.z_size, name='enc_fc_log_var')
            self.sigma = tf.exp(self.logvar / 2.0)
            self.epsilon = tf.random_normal([self.batch_size, self.z_size])
            self.z = self.mu + self.sigma * self.epsilon

            # Building the decoder part of VAE
            h = tf.layers.dense(self.z, 1024, name='dec_fc')
            h = tf.reshape(h, [-1, 1, 1, 1024])
            h = tf.layers.conv2d_transpose(h, 128, 5, strides=2, activation=tf.nn.relu, name='dec_deconv1')
            h = tf.layers.conv2d_transpose(h, 64, 5, strides=2, activation=tf.nn.relu, name='dec_deconv2')
            h = tf.layers.conv2d_transpose(h, 32, 6, strides=2, activation=tf.nn.relu, name='dec_deconv3')
            self.y = tf.layers.conv2d_transpose(h, 3, 6, strides=2, activation=tf.nn.sigmoid, name='dec_deconv4')

            
