# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 11:06:46 2019

@author: luisf
"""

# =============================================================================
# DIP Project. Implementing Bitplane Decomposition
#
# =============================================================================

import cv2
import numpy as np
import matplotlib.pyplot as plt

def BitplaneDecomposition(img):
    r, c = img.shape
    bitplane = np.array([img,img,img,img,img,img,img,img])
    comp = np.ones((r,c), dtype = np.uint8)
    
    for i in range(8):
       bitplane[i] = cv2.bitwise_and(img, np.left_shift(comp,i))
       bitplane[i][bitplane[i]>0] = 255
        
    return bitplane
    

root = 'D:/U de A/PDI/Project2/'

img = cv2.imread(root + 'ejemplo1.JPG')

img_g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

cv2.imshow('Gray', img_g)

img_eq = cv2.equalizeHist(img_g)


#fig, ax = plt.subplots(2,2)
#ax[0,0].imshow(img_g)
#ax[0,1].hist( img_g.ravel() , 255 , [0,255] )
#ax[1,0].imshow(img_eq)
#ax[1,1].hist(img_eq.ravel() , 255 , [0,255])

bitplane = BitplaneDecomposition(img_eq)

#for i in range(8):
#    cv2.imshow('Bitplane '+str(i),bitplane[i])

img_bin = np.copy(bitplane[7])

kernel =  cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
#kernel = np.ones((3,3), dtype = np.uint8)

img_bin = cv2.morphologyEx(img_bin, cv2.MORPH_OPEN, kernel, iterations=3)

cv2.imshow('Morph', img_bin)

_, contours,_ = cv2.findContours(img_bin, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

img_toshow = img.copy()
cv2.drawContours(img_toshow, contours, -1, (0,0,255), 2)

cv2.imshow('Problems', img_toshow)


cv2.waitKey(0)
cv2.destroyAllWindows()