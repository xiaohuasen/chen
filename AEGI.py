#-*-coding:GBK -*-
# -*- coding: utf-8 -*-
"""
By Fei Wang, Jan 2022
Contact: WangFei_m@outlook.com
This code implements the ghost imaging reconstruction using deep neural network constraint (GIDC) algorithm
reported in the paper: 
Fei Wang et al. 'Far-field super-resolution ghost imaging with adeep neural network constraint'. Light Sci Appl 11, 1 (2022).  
https://doi.org/10.1038/s41377-021-00680-w
Please cite our paper if you find this code offers any help.

Inputs:
A_real: illumination patterns (pixels * pixels * pattern numbers)
y_real: single pixel measurements (pattern numbers)

Outputs:
x_out: reconstructed image by GIDC (pixels * pixels)
"""
import torch
import numpy as np
import torch.nn as nn
from torch.nn import functional as F
import os
import re
import glob
import scipy.io as sio
from PIL import Image
import scipy.io as io
import tensorflow.compat.v1 as tf
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
import GIDC_model_Unet
import os
#os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
from natsort import natsorted, ns
tf.disable_v2_behavior()



#file = open("o-BesselM1J-50-s100-1.txt")
file = open("i-s80.txt")
context = file.readlines()
data = []
i = 0
while (i < len(context)):
        line = context[i]
        data.append(float(line.rstrip()))
        i = i + 1
        
mat = np.array(data)

size = 1500
#src_dir =  '/home/magic/chenyifan/data/BesselM1J-50-s100-1/'
src_dir =  '/home/magic/chenyifan/data/080/'

date = list()

#all_file = os.walk(src_dir)
os.chdir(src_dir)
filelist = [f for f in os.listdir(src_dir) if f.endswith('.png')]
filelist = natsorted(filelist)


for i in range(size):
#        img = cv2.imread(filelist[i], cv2.IMREAD_GRAYSCALE)

#        file_ext = os.path.splitext(file)
#        front, ext = file_ext
        #print(front)  #获取图片名字前缀
        #print(ext)  # 获取图片后缀

        img = Image.open(filelist[i])
        #保存为.npy
        res = np.array(img, dtype='uint16')
        res = np.expand_dims(res, 0)
        res1 = torch.FloatTensor(res/1.0)
        date.append(res1)
        
     
date = torch.cat(date,0).permute(1, 2, 0).contiguous().numpy()

# load data
#data = loadmat('e-80.mat') 
result_save_path = '.\\results\\'

# create results save path
if not os.path.exists(result_save_path):
    os.makedirs(result_save_path) 

# define optimization parameters
img_W = 800
img_H = 1280
SR = 0.1                                      # sampling rate
batch_size = 1
lr0 = 0.05                                    # learning rate
TV_strength = 1e-9                            # regularization parameter of Total Variation
num_patterns = 1500  # number of measurement times  
Steps = 501                                   # optimization steps

A_real = date[:, :, 0:num_patterns]  # illumination patterns

#print(date.shape)
#data1 = torch.FloatTensor(data['data']/1.0).permute(1,0).contiguous().numpy()


y_real = mat[0:num_patterns]    # intensity measurements

#if (num_patterns > np.shape(data['data'])[-1]):
#    raise Exception('Please set a smaller SR')

# DGI reconstruction
print('DGI reconstruction...')
B_aver  = 0
SI_aver = 0
R_aver = 0
RI_aver = 0
count = 0
for i in range(num_patterns):    
    pattern = date[:,:,i]
    count = count + 1
    B_r = mat[i]

    SI_aver = (SI_aver * (count -1) + pattern * B_r)/count
    B_aver  = (B_aver * (count -1) + B_r)/count
    R_aver = (R_aver * (count -1) + sum(sum(pattern)))/count
    RI_aver = (RI_aver * (count -1) + sum(sum(pattern))*pattern)/count
    DGI = SI_aver - B_aver / R_aver * RI_aver
# DGI[DGI<0] = 0
print('Finished')

with tf.variable_scope('input'):           
    inpt = tf.placeholder(tf.float32,shape=[batch_size,img_W,img_H,1],name = 'inpt')
    y = tf.placeholder(tf.float32,shape=[batch_size,1,1,num_patterns],name = 'y') 
    A = tf.placeholder(tf.float32,shape=[batch_size,img_W,img_H,num_patterns],name = 'A')                
    x = tf.placeholder(tf.float32,shape=[batch_size,img_W,img_H,1],name = 'x')   
                
    isTrain = tf.placeholder(tf.bool,name = 'isTrain')
    lr = tf.placeholder(tf.float32, name = 'learning_rate')
    groable = tf.Variable(tf.constant(0))
    lrate = tf.train.exponential_decay(lr0,groable,100,0.90)

# Build the DNN structure (the physical model was embedded in the DNN) y = Ax, y:measurements(known) A:physical model(known) x:object(unknown)
x_pred,y_pred = GIDC_model_Unet.inference(inpt, A, batch_size, img_W, img_H, num_patterns, isTrain)

# define the loss function
TV_reg = TV_strength*tf.image.total_variation(tf.reshape(x_pred,[batch_size,img_W,img_H,1]))
loss_y = tf.reduce_mean(tf.square(y - y_pred))
loss = loss_y + TV_reg
loss = loss_y

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)      
with tf.variable_scope('train_step'):
    with tf.control_dependencies(update_ops):
        train_op = tf.train.AdamOptimizer(learning_rate=lr,beta1=0.5,beta2=0.9,epsilon=1e-08).minimize(loss)
           
init_op = (tf.local_variables_initializer(),tf.global_variables_initializer())

with tf.Session() as sess:
    sess.run(init_op)  
    y_real = np.reshape(y_real,[batch_size,1,1,num_patterns])    
    A_real = np.reshape(A_real,[batch_size,img_W,img_H,num_patterns])
    DGI = np.reshape(DGI,[batch_size,img_W,img_H,1])
    
    # preprocessing
    # DGI = np.transpose(DGI) # sometimes it gives better results     
    DGI = (DGI - np.mean(DGI))/np.std(DGI)
    y_real = (y_real - np.mean(y_real))/np.std(y_real)
    A_real = (A_real - np.mean(A_real))/np.std(A_real)                
    
    # prepare for surveillance             
    DGI_temp0 = np.reshape(DGI,[img_W,img_H],order='F')
    DGI_temp = np.transpose(DGI_temp0)
    y_real_temp = np.reshape(y_real,[num_patterns])
    inpt_temp = DGI
    
    print('GIDC reconstruction...')
  
    for step in range(Steps): 
        lr_temp = sess.run(lrate,feed_dict={groable:step}) 
                                        
        if step%10 == 0: 
            train_y_loss = sess.run(loss_y, feed_dict={inpt:inpt_temp,y:y_real,A:A_real,isTrain:True,lr:lr_temp})
            print('step:%d----y loss:%f----learning rate:%f----num of patterns:%d' % (step,train_y_loss,lr_temp,num_patterns))           
                        
            [y_out,x_out] = sess.run([y_pred,x_pred],feed_dict={inpt:inpt_temp,y:y_real,A:A_real,isTrain:True,lr:lr_temp})  
            x_out = np.reshape(x_out,[img_W,img_H],order='F')           
             
            y_out =  np.reshape(y_out,[num_patterns],order='F')   
                               
            plt.subplot(141)
            plt.imshow(DGI_temp0)
            plt.title('DGI')
            plt.yticks([])
            
            plt.subplot(142)
            plt.imshow(x_out)
            plt.title('GIDC')
            plt.yticks([])
           
            ax1 = plt.subplot(143)
            plt.plot(y_out)
            plt.title('pred_y')
            ax1.set_aspect(1.0/ax1.get_data_ratio(), adjustable='box')       
            plt.yticks([])
            
            ax2 = plt.subplot(144)
            plt.plot(y_real_temp)
            plt.title('real_y')
            ax2.set_aspect(1.0/ax2.get_data_ratio(), adjustable='box')
            plt.yticks([])

            plt.subplots_adjust(hspace=0.25, wspace=0.25)
 #           plt.subplots_adjust(hspace=0.25,wspace=0.25)
            plt.show()

            x_out = x_out - np.min(x_out)
            x_out = x_out*255/np.max(np.max(x_out))
            x_out = Image.fromarray(x_out.astype('uint8')).convert('L')
            x_out.save('N_%d_%d.bmp'%(num_patterns,step))
            
#            DGI_temp0 = DGI_temp0 - np.min(DGI_temp0)
#            DGI_temp0 = DGI_temp0*255/np.max(np.max(DGI_temp0))
#            DGI_temp0 = Image.fromarray(DGI_temp0.astype('uint8')).convert('L')
#            DGI_temp0.save('O_%d_%d.bmp'%(num_patterns,step))
        # optimize the weights in the DNN
        sess.run([train_op],feed_dict={inpt:inpt_temp,y:y_real,A:A_real,isTrain:True,lr:lr_temp})

print('Finished!')
