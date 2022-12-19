# importing the libraries 

import numpy as np
import tensorfloe as tf

# building the VAE model within a class

class ConvVAE(object):
    # Initializing all the parameters and variables of the ConvVAE class
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

            # Implementing the training operations
            if self.is_training:
                self.global_step = tf.Variable(0,, name='global_step', trainable=False)

                # Defining the loss function
                self.r_loss = tf.reduced_sum(tf.square(self.x - self.y), reduction_indices= [1,2,3]) # calculating the squered error between the input and the output on all the dimentions
                self.r_loss = tf.reduce_mean(self.r_loss) # calculating the mean of the squared error
                self.kl_loss = -0.5 * tf.reduce_sum((1 + self.logvar - tf.square(self.mu) - tf.exp(self.logvar)), reduction_indices = 1) # calculating the KL divergence
                self.kl_loss = tf.maximum(self.kl_loss, self.kl_tolerance * self.z_size) # calculating the KL divergence
                self.kl_loss = tf.reduce_mean(self.kl_loss) # calculating the mean of the KL divergence
                self.loss = self.r_loss + self.kl_loss # calculating the total loss
                self.lr = tf.Variable(self.learning_rate, trainable=False) # defining the learning rate
                self.optimizer = tf.train.AdamOptimizer(self.lr) # defining the optimizer
                grads = self.optimizer.compute_gradients(self.loss) # calculating the gradients
                self.train_op = self.optimizer.apply_gradients(grads, global_step=self.global_step, name='train_step') # applying the gradients
            self.init = tf.global_variables_initializer() # initializing all the variables


