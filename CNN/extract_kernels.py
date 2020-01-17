
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 28 2019

@author: Shizheng Wen
This code is used for extracting the kernel or feature maps in the trained model.
email: szwen@nuaa.edu.cn
"""


import tensorflow as tf
import forward_propagation
import back_propagation
import numpy as np
import matplotlib.pyplot as plt
import os
import imageio

fluid_data = np.load('G:/fluid_data/interpolated_data_near_the_airfoil/1M_8000_9000_data.npy')
fluid_data = fluid_data[:,:,:,0:1]
#x_coordinate = np.load('x_coordinate.npy')
#y_coordinate = np.load('y_coordinate.npy')
fluid_data = fluid_data[0:500,:,:,:]    
fluid_data = fluid_data.astype(np.float32)
Re = 1000000


with tf.Graph().as_default() as g: 
        #这个x中的第一个参数需要改动
        x = tf.placeholder(tf.float32,[
            fluid_data[:,0,0,0].size,
            forward_propagation.IMAGE_SIZE_height,
            forward_propagation.IMAGE_SIZE_width,
            forward_propagation.NUM_CHANNELS])
        #y_ = tf.placeholder(tf.float32, [None, forward_propagation.OUTPUT_NODE])
        conv1_w, conv1, relu1, fc1w, fc2w = forward_propagation.forward(x,False,None)
        #y_output = forward_propagation.forward(x, False, None)

        ema = tf.train.ExponentialMovingAverage(back_propagation.MOVING_AVERAGE_DECAY)
        ema_restore = ema.variables_to_restore()
        saver = tf.train.Saver(ema_restore)
		
        
        with tf.Session() as sess:
                ckpt = tf.train.get_checkpoint_state(back_propagation.MODEL_SAVE_PATH)
                saver.restore(sess, ckpt.model_checkpoint_path)
				#accuracy_score,b_,c_ = sess.run([accuracy,b,c], feed_dict={x:fluid_data,y_:fluid_label}) 
                
                #
#                lim=np.arange(np.min(fluid_data[0,:,:,0]),np.max(fluid_data[0,:,:,0]),0.001)
#                plt.contourf(x_coordinate,y_coordinate,fluid_data[0,:,:,0],lim)
                #plt.contour(a[:,:,1],a[:,:,2],a[:,:,3],lim,camp='jet')
#                plt.colorbar()
#                plt.show()              
                
                #lim=np.arange(np.min(fluid_data[0,:,:,1]),np.max(fluid_data[0,:,:,1]),0.001)
                #plt.contourf(x_coordinate,y_coordinate,fluid_data[0,:,:,1],lim)
                #plt.contour(a[:,:,1],a[:,:,2],a[:,:,3],lim,camp='jet')
                #plt.colorbar()
                #plt.show()
            
                
#                lim=np.arange(np.min(fluid_data[0,:,:,2]),np.max(fluid_data[0,:,:,2]),0.001)
#                plt.contourf(x_coordinate,y_coordinate,fluid_data[0,:,:,2],lim)
                #plt.contour(a[:,:,1],a[:,:,2],a[:,:,3],lim,camp='jet')
#                plt.colorbar()
#                plt.show()
                
                
                
                #
                conv1_kernel, conv1, conv_featuremaps, fc1_w, fc2_w = sess.run([conv1_w, conv1, relu1, fc1w, fc2w], feed_dict = {x:fluid_data})
                #y_shape = sess.run(y_output, feed_dict = {x:fluid_data})
              #  for i in range(10):
                    # pressure_kernel
              #      fig2,ax2 = plt.subplots(figsize=(8,6))
              #      ax2.imshow(fluid_data[0,:,:,0])
              #      plt.title('Re = ' +str(Re)+', t= ' +str(time_point)+', pressure')
              #      plt.show()
              #      fig2,ax2 = plt.subplots(figsize=(3,3))
              #      ax2.imshow(conv1_kernel[:,:,0,i])
              #      plt.title('pressure kernel, '+str(i))
              #      plt.show()
                    
                 #   fig2,ax2 = plt.subplots(figsize=(8,6))
                 #   ax2.imshow(fluid_data[0,:,:,1])
                 #   plt.title('Re = ' +str(Re)+', t= ' +str(time_point)+', V_x')
                 #   plt.show()
                 #   fig2,ax2 = plt.subplots(figsize=(3,3))
                 #   ax2.imshow(conv1_kernel[:,:,1,i])
                 #   plt.title('V_x kernel, '+str(i))
                 #   plt.show()
                    
                 #   fig2,ax2 = plt.subplots(figsize=(8,6))
                 #   ax2.imshow(fluid_data[0,:,:,2])
                 #   plt.title('Re = ' +str(Re)+', t= ' +str(time_point)+', V_y')
                 #   fig2, ax2 = plt.subplots(figsize=(3,3))
                 #   ax2.imshow(conv1_kernel[:,:,2,i])
                 #   plt.title('V_y kernel, '+str(i))
                 #   plt.show()
                    
              #      fig2,ax2 = plt.subplots(figsize=(8,6))
              #      ax2.imshow(conv1_featuremaps[0,:,:,i])
              #      plt.title('feature maps: '+str(i))
              #      plt.show()
                 #   aaa = conv1_kernel[:,:,0,i]
                  #  fig2,ax2 = plt.subplots(figsize = (2,2))
                   # ax2.imshow(aaa)
                    #plt.show()
                    #lim = np.arange(np.min(conv1_20[0,:,:,i]), np.max(conv1_20[0,:,:,i]),0.001)
                   # plt.contourf(a[:,:,1],a[:,:,2],conv1_20[0,:,:,i],lim)
                   # plt.colorbar()
                   # plt.show()
                   
'''
animation
        
for i in range(0, 500):
   fig2,ax2 = plt.subplots(figsize=(8,6))
   ax2.imshow(conv1_featuremaps[i,:,:,4])
   plt.title('Re = '+str(Re)+', t = '+str(16+i*0.002)+' s', fontsize = 26)
   plt.tick_params(labelsize=15)
   plt.savefig('./figure/frame'+str(i)+'.png', bbox_inches = 'tight')
   plt.close()
               
print("animating...",end="",flush=True)
with imageio.get_writer('re'+str(Re)+'.gif', mode='I',fps=30) as writer:
    for i in range(0,500):
        image = imageio.imread('./figure/frame'+str(i)+'.png')
        writer.append_data(image)
        os.remove('./figure/frame'+str(i)+'.png')
print("done.",flush=True)
 
   '''  
#fig2, ax2 = plt.subplots(figsize = (10,10))

#fluid_data = np.load('./fluid_data/nondimensionalize/test_200k_data.npy')
#ax2.imshow(fluid_data[0,:,:,0])
#plt.show()

#fluid_data_1 = np.load('./fluid_data/nondimensionalize/test_1k_data.npy')
#ax2.imshow(fluid_data_1[0,:,:,0])
#plt.show()
