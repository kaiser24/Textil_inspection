# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 17:28:56 2019

@author: luisf
"""

import cv2
import numpy as np

def BitplaneDecomposition(img):
    r, c = img.shape
    bitplane = np.array([img,img,img,img,img,img,img,img])
    comp = np.ones((r,c), dtype = np.uint8)
    
    for i in range(8):
       bitplane[i] = cv2.bitwise_and(img, np.left_shift(comp,i))
       bitplane[i][bitplane[i]>0] = 255
        
    return bitplane

path = 'D:/U de A/PDI/Project2/imagenes UDEA/'
name = 'img8.bmp'
img = cv2.imread(path + name)
r,c,l = img.shape
img_go = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_g = img_go
img_g = cv2.equalizeHist(img_go)

bitplane = BitplaneDecomposition(img_g)
img_bin = bitplane[3].copy()


img_toshow = cv2.resize(img_bin, (int(c*0.15),int(r*0.15)), cv2.INTER_AREA)


cv2.imshow('bin',img_toshow)


cv2.waitKey(0)
cv2.destroyAllWindows()