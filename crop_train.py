# -*- coding: utf-8 -*-
"""
Scaling/cropping all the images to 256x256 pixel images

Started on Sat May  5 13:59:07 2018
@author: Victor van Vucht
"""
import numpy as np
import pandas as pd
import os
import imageio as imio



#%%
Height=256 #Height we want to crop to
Width=255 #Width we want to crop to

Directory_Filtered = "Data/train_filtered/"
Directory_Cropped = "Data/train_filtered_cropped/"
if not os.path.exists(Directory_Cropped):
    os.makedirs(Directory_Cropped)
#load the filtered data
for file in os.listdir(Directory_Filtered):
    im = imio.imread(Directory_Filtered + file)
    middle=np.round([im.shape[0]/2,im.shape[1]/2],0) #middle point of the image
    
    H= np.array([middle[0]-(Height/2),middle[0]+(Height/2)]).astype(int) #Pixel range of height
    W= np.array([middle[1]-(Width/2),middle[1]+(Width/2)]).astype(int) #Pixel width of height
    newim = im[H[0]:H[1],W[0]:W[1],:]
    
    imio.imwrite(Directory_Cropped + file,newim)
    
