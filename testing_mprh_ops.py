# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 16:26:57 2019

@author: luisf
"""
import cv2
import numpy as np


path = 'D:/U de A/PDI/Project2/imagenes 2 UDEA/'
name = 'img2.bmp'
img = cv2.imread(path + name)
    #img = cv2.GaussianBlur(img,(5,5),0)
    
img_g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#img_g = cv2.equalizeHist(img_g)
#1) mean 20 - 30 , 3 THEN e3 d3
#2) mean 71 21 ONLY THRESH

# 3) 51,5
# 4) 71, 27 ONLY THRESH
img_bin = cv2.adaptiveThreshold(img_g, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 71,35)

r,c = img_bin.shape

img_dark = img_bin.copy()

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
#kernel = np.ones((5,5) , np.uint8)    #squared

#dark zones
#img_dark = img_dark * 0
#img_dark = cv2.erode(img_dark, kernel, iterations = 4)
#img_dark = cv2.dilate(img_dark, kernel, iterations = 4 )


img_bright = 255 - img_bin

kernel2 = np.ones((5,5) , np.uint8)
#bright zones
#img_bright = cv2.erode(img_bright, kernel, iterations = 7)
#img_bright = cv2.dilate(img_bright, kernel, iterations = 7)
#img_bright = cv2.erode(img_bright, kernel2, iterations = 2)

img_bright = img_bright*0
#img_dark = img_dark*0

img_mask = cv2.bitwise_or(img_dark,img_bright)
#img_mask = cv2.dilate(img_mask, kernel, iterations = 3)
#img_mask = cv2.erode(img_mask, kernel, iterations = 3)

_, contours,_ = cv2.findContours(img_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

img_toshow = cv2.merge([img_g,img_g,img_g])

cv2.drawContours(img_toshow, contours,-1, (0,0,255), 3)


img_toshow = cv2.resize(img_mask, (int(c*0.15),int(r*0.15)), cv2.INTER_AREA)

cv2.imshow('s1',img_toshow)
cv2.waitKey(0)
cv2.destroyAllWindows()