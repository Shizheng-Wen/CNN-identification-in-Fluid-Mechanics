# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 13:41:53 2019
Note: This code is used for calculating CNN (back propagation).
It should be noted that the input data for back propagation is the interpolated data.
@author: Shizheng Wen
email: szwen@nuaa.edu.cn
"""


import tensorflow as tf
import forward_propagation
import os
import numpy as np


BATCH_SIZE = 100
LEARNING_RATE_BASE = 0.0005
LEARNING_RATE_DECAY = 0.99
REGULARIZER = 0.0001
STEPS = 10000
MOVING_AVERAGE_DECAY = 0.99
MODEL_SAVE_PATH="./training_model_exp/"
MODEL_NAME="model"


def backward(fluid_data, fluid_label):

    x = tf.placeholder(tf.float32, [
            BATCH_SIZE,
            forward_propagation.IMAGE_SIZE_height,
            forward_propagation.IMAGE_SIZE_width,
            forward_propagation.NUM_CHANNELS])
    y_ = tf.placeholder(tf.float32, [None, forward_propagation.OUTPUT_NODE])
    y = forward_propagation.forward(x, True, REGULARIZER)
    global_step = tf.Variable(0, trainable=False)

    #loss function considering the method of regularization
    ce = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    cem = tf.reduce_mean(ce)
    loss = cem + tf.add_n(tf.get_collection('losses'))
    
    #method of learning rate decay
    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,
        global_step,
        fluid_data[:,0,0,0].size / BATCH_SIZE, 
        LEARNING_RATE_DECAY,
        staircase=True)
 
    #different optimizers
    #train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step = global_step)


    ema = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    ema_op = ema.apply(tf.trainable_variables())
    with tf.control_dependencies([train_step, ema_op]):
        train_op = tf.no_op(name='train')

    saver = tf.train.Saver()
    
    #accuracy on training dataset
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1)) 
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)) 
    
    
    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        
        ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
        
        
        for i in range(STEPS):
            start = (i*BATCH_SIZE) % 6000  #note!!: This number should change with the size of training dataset
            end = start + BATCH_SIZE
            _, loss_value, step, accuracy_score = sess.run([train_op, loss, global_step, accuracy], feed_dict={x: fluid_data[start:end,:,:,:], y_: fluid_label[start:end,:]})
            if i % 100 == 0:
                print("After %d training step(s), loss on training batch is %g." % (step, loss_value))
                print("accuracy for training dataset is %d" % (accuracy_score))                
                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)


def main():    
    fluid_data = np.load('G:/fluid_data/experiment/training_data.npy')
    fluid_label = np.load('G:/fluid_data/experiment/training_label.npy')
    fluid_data = fluid_data.astype(np.float32)
    fluid_label = fluid_label.astype(np.float32)
    backward(fluid_data, fluid_label)

if __name__ == '__main__':
    main()