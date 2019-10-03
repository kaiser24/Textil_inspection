# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 17:17:32 2019

@author: luisf
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Sep 15 19:48:45 2019

@author: luisf
"""

import cv2
import numpy as np
import os


path = 'D:/U de A/PDI/Project2/imagenes 2 UDEA/'

for name in os.listdir(path):
    img = cv2.imread(path + name)
    #img = cv2.GaussianBlur(img,(5,5),0)
    
    img_g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_bin = cv2.adaptiveThreshold(img_g, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 71,27)
    
    r,c = img_bin.shape
    
    img_mask = img_bin.copy()
    
    #kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (20,20))
    #kernel = np.ones((20,20) , np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20,20))
    
    
    img_bright = 255 - img_bin
    
    kernel2 = np.ones((5,5) , np.uint8)
    
    #img_mask = cv2.bitwise_or(img_dark,img_bright)
    img_mask = cv2.dilate(img_mask, kernel, iterations = 2)
    img_mask = cv2.erode(img_mask, kernel, iterations = 2)
    
    _, contours,_ = cv2.findContours(img_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    img_toshow = cv2.merge([img_g,img_g,img_g])
    
    cv2.drawContours(img_toshow, contours,-1, (0,0,255), 3)
    
    
    img_toshow = cv2.resize(img_toshow, (int(c*0.15),int(r*0.15)), cv2.INTER_AREA)
    
    cv2.imshow('s1',img_toshow)
    cv2.waitKey(0)
cv2.destroyAllWindows()