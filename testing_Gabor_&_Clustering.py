# -*- coding: utf-8 -*-
"""
Created on Sun Sep 29 22:20:47 2019

@author: luisf
"""

import cv2
import numpy as np
import time as t
import multiprocessing as mp
import psutil

# Local Energy
def localEnergy(args):
    img, wsize = args
    r_img , c_img = img.shape

    win_size = wsize
    
    r_ini = int( (win_size - 1)/2 )
    c_ini = int( (win_size - 1)/2 )
    r_end = r_img - r_ini
    c_end = c_img - c_ini
    
    rows = 0
    cols = 0
    
    # the new image is 10x10 smaller than the original
    img_eng = np.zeros((r_img - r_ini*2 ,c_img - c_ini*2), np.float32)
    
    #passing the window through all the image
    for i in range(r_ini, r_end ):
        cols = 0
        for j in range(c_ini, c_end):
            window = img[ i - r_ini : i + r_ini , j - c_ini : j + c_ini ]
            
            # calculate the mean and std of the window
            mean = cv2.mean(window)[0]
            #img_eng[rows,cols] = np.sum( np.abs( window - mean ) ) / (win_size*win_size)
            img_eng[rows,cols] = np.sum( np.abs( window - mean ) ) / (np.sum(window))

            cols = cols + 1
        
        rows = rows + 1
    
    #normalizing, rounding and casting to uint8 which is the format for images
    img_eng = cv2.copyMakeBorder(img_eng, r_ini, r_ini, c_ini, c_ini, cv2.BORDER_REPLICATE)
    
    return img_eng

if __name__=='__main__':
    # Reading the image
    path = 'D:/U de A/PDI/Project2/imagenes 2 UDEA/'
    name = 'img12.bmp'
    img = cv2.imread(path + name)
    
    img_g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    r,c = img_g.shape
    #Creating the Gabor filter bank
    #gabor = cv2.getGaborKernel(ksize, sigma, theta, lambdaa, gamma, psi, ktype)
    ksizes = [10, 50]
    
    sigmas = 0.7
    thetas = [np.pi, np.pi/2, 0]
    #thetas = [np.pi, np.pi/6]
    lambd = 8.0
    
    wsize = 101
    
    #filters = []
    filtered = []
    pfiltered = []
    nf = 0
    for size in ksizes:
        for th in thetas:
            gabor = cv2.getGaborKernel((size,size), sigmas, th, lambd, 5, 0, ktype = cv2.CV_32F) 
            filtered.append( cv2.filter2D(img_g, cv2.CV_8UC3, gabor) )
            nf = nf + 1
    
    num_cpus = psutil.cpu_count(logical=True)
    t11 = t.perf_counter()
    pool = mp.Pool(num_cpus)
    pfiltered = pool.map(localEnergy, zip(filtered, nf*[wsize]) )
    
    # nf: number of features
    # merges all the images as channels, so we end up with a r,c,nf (nf filters)
    features = cv2.merge(pfiltered)
    
    # reshapes the bank so we end up with a rXc,12. in other words, each new row
    # its a pixel, and each column its a feature, and we have nf features for each
    # pixel
    features = features.reshape((-1,nf))
    t12 = t.perf_counter()
    #%%
    #==============================================================================
    #                                   CLUSTERING
    # stop criteria
    t21 = t.perf_counter()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    
    clusters = 2
    attempts=10
    
    # applying k-means clustering. we get the labels for each cluster for th data
    # and the centers of the data
    ret,label,center=cv2.kmeans(features,clusters,None,criteria,attempts,cv2.KMEANS_PP_CENTERS)
    
    #center =np.uint8(center)
    t22 = t.perf_counter()
#%%
    mul = int(255 / (clusters-1) )
    img_seg = label.reshape((img_g.shape))
    img_seg = np.uint8(img_seg * mul)
    img_mask = img_seg
    #TODO: ISLAND REMOVAL
    
    
    _, contours,_ = cv2.findContours(img_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    img_toshow = [img_g,img_g,img_g]
    img_toshow = cv2.merge(img_toshow)
    cv2.drawContours(img_toshow, contours,-1, (0,0,255), 3)
    
    #cv2.rectangle(img_toshow, (0,0), (200,2000), (255,0,0), 10)
    img_toshow = cv2.resize(img_toshow, (int(c*0.15),int(r*0.15)), cv2.INTER_AREA)
    
    cv2.imshow('a',img_toshow)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()