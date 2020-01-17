# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 2019

@author: Shizheng Wen
@This code is used for reshaping the data structure for CNN
szwen@nuaa.edu.cn
"""
'''
'''

from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

#...................................................................................
# function for reading data_file
def open_file(file):
    My_file = open(file, 'r')
    result = []
    for line in My_file:
        line = line.strip('\n')
        result.append(line.split(','))
    My_file.close()
    del result[0]
    result =np.array(result, dtype = float)
    return result

#...................................................................................
#select the range near the airfoil
def range_select(result, x_min, x_max, y_min, y_max):
    result_range = []
    for i in range(len(result)):
        if result[i][1] < x_max and result[i][1] > x_min and result[i][2] < y_max and result[i][2] > y_min:
            result_range.append(result[i])
    result_range = np.matrix(result_range)
    return np.array(result_range)

#...................................................................................
#interpolate process
def interpolate_data (result_range, length_input, width_input, x_min, x_max, y_min, y_max):
    size_input = np.zeros((width_input, length_input, 6))
    length_x = x_max - x_min
    length_y = y_max - y_min
    delta_x = length_x / length_input
    delta_y = length_y / width_input
    #first, generating the input_array
    m = 0
    for i in range(width_input):
        for j in range(length_input):
            size_input[i,j,0] = m
            size_input[i,j,1] = x_min + delta_x + j * delta_x
            size_input[i,j,2] = y_max - delta_y - i * delta_y
            m += 1
    size_input.resize((length_input * width_input , 6))
    #secondly, start to interolate
    point_grid =size_input[:, 1:3]  #Interpolation point coordinates
    points = result_range[:,1:3] # real point coordinates
    values_1 = result_range[:, 3] #real point values of pressure
    values_2 = result_range[:, 4] #real point values of vx
    values_3 = result_range[:, 5] #real point values of vy
    
    size_input[:,3] = griddata(points, values_1, point_grid, method='nearest') #interpolation
    size_input[:,4] = griddata(points, values_2, point_grid, method='nearest') #interpolation
    size_input[:,5] = griddata(points, values_3, point_grid, method='nearest') #interpolation
    #size_input.resize((length_input, width_input, 6))
    
    return size_input
#..........................................................................................
#airfoil geometry range
def geo_function(decide_point):
    return (0.12/0.2)*(0.2969*decide_point**0.5 - 0.1260*decide_point - 0.3516*decide_point**2 + 0.2843*decide_point**3 - 0.1015*decide_point**4)


#........................................................................................
#the point inside the airfoil will be set into zero
def airfoil_zero(size_input):
    for i in range(len(size_input)):
        if size_input[i,1] > 0:
            y_up = geo_function(size_input[i,1])
            y_down = -y_up
            if  size_input[i, 2] < y_up and size_input[i,2] >y_down:
                size_input[i,3] = 0
                size_input[i,4] = 0
                size_input[i,5] = 0
    return size_input


#........................................................................................
# convert pressure into pressure coefficient
    
def Nondimensionalize(size_input):
    P_infinite = 101325
    M_infinite = 0.0583
    v_total = 20
    density_rou = 1.225
    kinetic_v = 0.5 * density_rou * v_total * v_total
    #k = 1.4
    for i in range(len(size_input)):
        #pressure
        size_input[i][3] = (size_input[i][3] - P_infinite) / kinetic_v
        #Mach number for x
        size_input[i][4] = (size_input[i][4] / v_total) * M_infinite
        #Mach number for y
        size_input[i][5] = (size_input[i][5] / v_total) * M_infinite
    return size_input

#........................................................................................
#main program
#select the point range (near-the-airfoil)
def main_program(file_name):
    x_min = -0.5
    x_max = 1.5
    y_min = -0.5
    y_max = 1
#size of input point
    x_sz = 0.01
    y_sz = 0.01
#the size for inputting
    length_input = int((x_max - x_min)/x_sz)
    width_input = int((y_max - y_min)/y_sz)
#read the file:
    result = open_file(file_name)
    result_range = range_select(result, x_min, x_max, y_min, y_max)
    size_input = interpolate_data(result_range, length_input, width_input, x_min, x_max, y_min, y_max)
    size_input = Nondimensionalize(size_input)
    size_input = airfoil_zero(size_input)
    size_input.resize((width_input, length_input, 6))
    size_input_3 = size_input[:,:, 3:6]
    a_1 = np.reshape(size_input_3, (1,width_input, length_input, 3))
    return a_1


''' The following code can be used to plot the figure and test'''
#test
#for i in range(8000,9000,10):
#    size_input = main_program('./DNSruns/40degDNS1k/40degDNS1k-'+ str(i))
#    fig2,ax2 = plt.subplots(figsize=(6,4))
#    ax2.imshow(size_input[:,:,3])
#    plt.show()

#size_input = main_program('./DNSruns/40degDNS100k/40degDNS100k-8000')
#fig2,ax2 = plt.subplots(figsize=(6,4))
#ax2.imshow(size_input[:,:,3])
#plt.show()
#lim=np.arange(np.min(size_input[:,:,3]),np.max(size_input[:,:,3]),0.05)
#plt.contourf(size_input[:,:,1],size_input[:,:,2],size_input[:,:,3],lim,cmap = 'jet')
#plt.contour(size_input[:,:,1],size_input[:,:,2],size_input[:,:,3],lim,cmap="jet")#jet
#plt.colorbar()
#plt.show()      