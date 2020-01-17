# -*- coding: utf-8 -*-
"""
Created on Sun Jul 28 2019
This code is used for testing the accuracy of data
@author: Shizheng Wen 
email:szwen@nuaa.edu.cn
"""

import time
import tensorflow as tf
import forward_propagation
import back_propagation
import numpy as np

TEST_INTERVAL_SECS = 20

def test(fluid_data, fluid_label):
    with tf.Graph().as_default() as g: 
        x = tf.placeholder(tf.float32,[
            fluid_data[:,0,0,0].size,
            forward_propagation.IMAGE_SIZE_height,
            forward_propagation.IMAGE_SIZE_width,
            forward_propagation.NUM_CHANNELS])
        y_ = tf.placeholder(tf.float32, [None, forward_propagation.OUTPUT_NODE])
        y = forward_propagation.forward(x,False,None)

        ema = tf.train.ExponentialMovingAverage(back_propagation.MOVING_AVERAGE_DECAY)
        ema_restore = ema.variables_to_restore()
        saver = tf.train.Saver(ema_restore)
		
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1)) 
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)) 
        
        #a = y
        c = tf.argmax(y, 1)
        #b = tf.argmax(y_, 1)
        
        while True:
            with tf.Session() as sess:
                ckpt = tf.train.get_checkpoint_state(back_propagation.MODEL_SAVE_PATH)
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess, ckpt.model_checkpoint_path)
					
                    global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                    accuracy_score,c_ = sess.run([accuracy,c], feed_dict={x:fluid_data,y_:fluid_label}) 
                    print("After %s training step(s), test accuracy = %g" % (global_step, accuracy_score))
                    print(c_)
                else:
                    print('No checkpoint file found')
                    return
            time.sleep(TEST_INTERVAL_SECS) 

def main():
    fluid_data = np.load('G:/fluid_data/interpolated_data_near_the_airfoil/1M_9000_10000_data.npy')
    fluid_data = fluid_data[:,:,:,0:1]
    fluid_label = np.load('G:/fluid_data/test_label_3.npy')
    #fluid_data1 = fluid_data[:,:,:,0]
    #fluid_data2 = np.reshape(fluid_data1,(2000,100,100,1))
    #fluid_data = np.load('./fluid_data/nondimensionalize/test_1k_200k_data.npy')
    #fluid_label = np.load('./fluid_data/nondimensionalize/test_1k_200k_label.npy')
    
    fluid_data = fluid_data.astype(np.float32)
    fluid_label = fluid_label.astype(np.float32)
    
    test(fluid_data, fluid_label)

if __name__ == '__main__':
    main()
