# -*- coding: utf-8 -*-
"""
Scaling/cropping all the images to 255x255 pixel images through random cropping,

Started on Sat May 19 11:00 2018
@author: Victor van Vucht
"""
import numpy as np
import pandas as pd
import os
import imageio as imio


def crop(im, Height, Width, option):
    if option == 'random':
        midpoint=[np.random.randint(Height//2,im.shape[0]+1-Height//2),np.random.randint(Width//2,im.shape[1]+1-Width//2)]
    else:
        midpoint=np.round([im.shape[0]/2,im.shape[1]/2],0) #middle point of the image
    
    H= np.array([midpoint[0]-(Height/2),midpoint[0]+(Height/2)]).astype(int) #Pixel range of height
    W= np.array([midpoint[1]-(Width/2),midpoint[1]+(Width/2)]).astype(int) #Pixel width of height
    return im[H[0]:H[1],W[0]:W[1],:]
    
    
#%%
Height=255 #Height we want to crop to
Width=255 #Width we want to crop to

Directory_Filtered = "Data/train/"
Directory_Trimmed = "Data/train_cropped/"
if not os.path.exists(Directory_Trimmed):
    os.makedirs(Directory_Trimmed)
#load the filtered data
for file in os.listdir(Directory_Filtered):
    im = imio.imread(Directory_Filtered + file)
    newim = crop(im,Height, Width,'random')
    imio.imwrite(Directory_Trimmed + file,newim)
    
    
