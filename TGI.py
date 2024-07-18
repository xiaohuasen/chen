#-*-coding:GBK -*-
# -*- coding: utf-8 -*-
import os
import numpy as np
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
import os
import cv2
from natsort import natsorted, ns
tf.disable_v2_behavior()


#file = open("n-BesselM1J-50-s100-1.txt")
file = open("bkd_3.txt")
context = file.readlines()
data = []
i = 0
while (i < len(context)):
        line = context[i]
        data.append(float(line.rstrip()))
        i = i + 1
        
bucket = np.array(data)



size = 1500
#src_dir =  '/home/magic/chenyifan/data/BesselM1J-50-s100-1/'
src_dir =  '/home/magic/chenyifan/data/010/'
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






ghost = np.zeros((800, 1280))
bucket_sum = 0
sum_field = ghost
corr_sum = ghost
number_sum = 0
ghost_sum = ghost
number_sum=[]


# # spatial_img = cv2.imread(filelist[0], cv2.IMREAD_GRAYSCALE)
# spatial_img = cv2.imread(filelist[0], 0)
# spatial_img1 = cv2.imread(filelist[0])

# # spatial_img1 = spatial_img.astype(np.float64)
# # spatial_img2 = plt.imread(filelist[0])

for i in range(np.size(data)):
    
    # spatial_img1 = cv2.imread(filelist[i], cv2.IMREAD_GRAYSCALE)
    spatial_img1 = date[:,:,i]
    spatial_img = spatial_img1.astype('float64')
    # print(filelist[i])
#     print(np.shape(img)) 
    sum_field = sum_field+spatial_img
    print(i)
    
    # Traditional GI
    
    # print('speatial_field =', spatial_img)
    # print('sum_field =', sum_field)
    mean_field = sum_field/(i+1)
    # print('mena_field =', mean_field)
    bucket_sum = bucket_sum+bucket[i]
    # print(bucket_sum)
    # print(bucket[i])
    mean_bucket = bucket_sum/(i+1)
    ghost_sum = ghost_sum + ((spatial_img-mean_field)*(bucket[i]-mean_bucket)) 
    # print((spatial_img - mean_field)*(bucket[i] - mean_bucket))
    # print(ghost_sum)
    ghost_final = ghost_sum/(i+1)
    # plt.show()
    # imshow(ghost_final)
    # plt.pause(0.05)
    if i == size:
        break
    if i%250 == 0:  
       DGI_temp0 = ghost_final
       DGI_temp0 = DGI_temp0 - np.min(DGI_temp0)
       DGI_temp0 = DGI_temp0*255/np.max(np.max(DGI_temp0))
       DGI_temp0 = Image.fromarray(DGI_temp0.astype('uint8')).convert('L')
       DGI_temp0.save('E_%d.png'%(i))
     
DGI_temp0 = ghost_final
plt.subplot(141)
plt.imshow(DGI_temp0)
plt.title('TGI')
plt.yticks([])

plt.subplots_adjust(hspace=0.25, wspace=0.25)
 #           plt.subplots_adjust(hspace=0.25,wspace=0.25)
plt.show()

DGI_temp0 = DGI_temp0 - np.min(DGI_temp0)
DGI_temp0 = DGI_temp0*255/np.max(np.max(DGI_temp0))
DGI_temp0 = Image.fromarray(DGI_temp0.astype('uint8')).convert('L')
DGI_temp0.save('O.png')