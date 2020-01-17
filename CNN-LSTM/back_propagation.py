# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 13:41:53 2019

@author: Shizheng Wen
Note: This code considers the structure of CNN_LSTM Neural Networks
"""

import tensorflow as tf
import forward_propagation
import os
import numpy as np


BATCH_SIZE = 1
time_steps = 200
LEARNING_RATE_BASE = 0.0005
LEARNING_RATE_DECAY = 0.99
REGULARIZER = 0.0001
STEPS = 10000
MOVING_AVERAGE_DECAY = 0.99
MODEL_SAVE_PATH="./model_LSTM/"
MODEL_NAME="model"


def backward(fluid_data, fluid_label):

    x = tf.placeholder(tf.float32, [
            BATCH_SIZE*time_steps,
            forward_propagation.IMAGE_SIZE_height,
            forward_propagation.IMAGE_SIZE_width,
            forward_propagation.NUM_CHANNELS])
    y_ = tf.placeholder(tf.float32, [None, forward_propagation.OUTPUT_NODE])
    y = forward_propagation.forward(x, True, REGULARIZER)
    global_step = tf.Variable(0, trainable=False)

    #loss function considering the regularization
    ce = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    cem = tf.reduce_mean(ce)
    loss = cem + tf.add_n(tf.get_collection('losses'))

    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,
        global_step,
        fluid_data[:,0,0,0].size / (BATCH_SIZE*time_steps), 
        LEARNING_RATE_DECAY,
        staircase=True)
    #learning_rate (exponential_decay)

    #train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step = global_step)
    #train_step = tf.train.MomentumOptimizer(learning_rate, momentum).minimize(loss, global_step = global_step)
    
    #model evaluation 
#    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
#    accuracy = tf.reduce_mean(tf.case(correct_prediction, tf.float32))

    ema = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    ema_op = ema.apply(tf.trainable_variables())
    with tf.control_dependencies([train_step, ema_op]):
        train_op = tf.no_op(name='train')

    saver = tf.train.Saver()

    #model evaluation on training dataset
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1)) 
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)) 
    
    
    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        
        ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
        
        
        for i in range(STEPS):
            start = (i*BATCH_SIZE*time_steps) % fluid_data[:,0,0,0].size
            end = start + BATCH_SIZE*time_steps
            _, loss_value, step, accuracy_score = sess.run([train_op, loss, global_step, accuracy], feed_dict={x: fluid_data[start:end,:,:,:], y_: fluid_label[int(start/200):int(end/200),:]})
            if i % 100 == 0:
                print("After %d training step(s), loss on training batch is %g." % (step, loss_value))
                print("accuracy for training dataset is %d" % (accuracy_score))
                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)


def main():    
    fluid_data = np.load('training_data_LSTM.npy')
    fluid_label = np.load('training_label_LSTM.npy')
    #fluid_data1 = fluid_data[:,:,:,0]
    #fluid_data2 = np.reshape(fluid_data1,(4000,100,100,1))
    #reduce the assumption of memory
    fluid_data = fluid_data.astype(np.float32)
    fluid_label = fluid_label.astype(np.float32)
    backward(fluid_data, fluid_label)

if __name__ == '__main__':
    main()