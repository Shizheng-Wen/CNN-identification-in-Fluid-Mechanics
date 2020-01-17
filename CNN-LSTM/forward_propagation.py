# -*- coding: utf-8 -*-
"""
Created on August 29 2019
@author: Shizheng Wen
Instructions:
This code considers CNN+LSTM Neural Networks, all photos are sharing same convolutional kernels
"""

import tensorflow as tf
from tensorflow.contrib import rnn
#import data_preprocessing

IMAGE_SIZE_height = 150
IMAGE_SIZE_width = 200
NUM_CHANNELS = 1
CONV1_SIZE = 3
CONV1_KERNEL_NUM = 10
#CONV2_SIZE = 5
#CONV2_KERNEL_NUM = 20
FC_SIZE = 200
Units_LSTM = 200
OUTPUT_NODE = 3

#
batch_size = 1
time_steps = 200

def get_weight(shape, regularizer):
    w = tf.Variable(tf.truncated_normal(shape, stddev = 0.1))
    if regularizer != None:
        tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(regularizer)(w))
    return w

def get_bias(shape):
    b = tf.Variable(tf.zeros(shape))
    return b

def conv2d(x,w):
    return tf.nn.conv2d(x, w, strides = [1, 1, 1, 1], padding = 'SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')

def lstmNN(x, num_units):
    lstm_layer = rnn.BasicLSTMCell(num_units, forget_bias = 1.0, state_is_tuple = True, reuse = tf.get_variable_scope().reuse)
    lstm_layer = rnn.DropoutWrapper(cell = lstm_layer, output_keep_prob = 0.75)
    outputs, states = rnn.static_rnn(lstm_layer, x, dtype = 'float32')
    return outputs[-1], states
    
def forward(x, train, regularizer):
    #convolution layer 1
    conv1_w = get_weight([CONV1_SIZE,  CONV1_SIZE, NUM_CHANNELS, CONV1_KERNEL_NUM], regularizer)
    conv1_b = get_bias ([CONV1_KERNEL_NUM])
    conv1 = conv2d(x, conv1_w)
    relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_b))
    pool1 = max_pool_2x2(relu1)
    
    #convolution layer 2
    #conv2_w = get_weight([CONV2_SIZE, CONV2_SIZE, CONV1_KERNEL_NUM, CONV2_KERNEL_NUM], regularizer)
    #conv2_b = get_bias([CONV2_KERNEL_NUM])
    #conv2 = conv2d(pool1, conv2_w)
    #relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_b))
    #pool2 = max_pool_2x2(relu2)
    
    #reshape
    pool_shape = pool1.get_shape().as_list()
    # pool_shape= batch_size, height, width, num_channels = pool1.get_shape().as_list()
    nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]
    reshaped = tf.reshape(pool1, [pool_shape[0], nodes]) #batch 行，所有特征点作为列的二维张量
    #fully connected layer(one hidden layer)
    fc1_w = get_weight([nodes, FC_SIZE], regularizer)
    fc1_b = get_bias([FC_SIZE])
    fc1 = tf.nn.relu(tf.matmul(reshaped, fc1_w) + fc1_b)
    #reshape fc1 into the form of [batch_size, n_steps, n_input]
    fc1 = tf.reshape(fc1, [batch_size, time_steps, FC_SIZE])
    fc1 = tf.transpose(fc1, [1, 0, 2])
    fc1 = tf.reshape(fc1, [-1, FC_SIZE])
    fc1 = tf.split(fc1, time_steps, 0)
    #new shape: n_steps * (batch_size, n_hidden)
    #processing the input tensor from [bach_size, n_steps, n_input] to 'time_steps' number of [batch_size, n_input] tensors
    #define LSTM Neural Network
    lstm1, state1 = lstmNN(fc1, Units_LSTM)
  #  lstm_layer = rnn.BasicLSTMCell(Units_LSTM, forget_bias = 1.0, state_is_tuple = True, reuse = tf.get_variable_scope().reuse)
  #  lstm_layer = rnn.DropoutWrapper(cell = lstm_layer, output_keep_prob = 0.75)
  #  outputs, states = rnn.static_rnn(lstm_layer, fc1, dtype = 'float32')
  #  lstm1 = outputs[-1]
    # when training, dropout some neurons at hidden layer randomly.
    if train:
        lstm1 = tf.nn.dropout(lstm1, 0.5)
    
    fc2_w = get_weight([Units_LSTM, OUTPUT_NODE], regularizer)
    fc2_b = get_bias([OUTPUT_NODE])
    y = tf.matmul(lstm1, fc2_w) + fc2_b
    return y

