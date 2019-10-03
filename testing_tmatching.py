# -*- coding: utf-8 -*-
"""
Created on Mon Sep 16 19:55:21 2019

@author: luisf
"""

import cv2
import numpy as np


path = 'D:/U de A/PDI/Project2/imagenes 2 UDEA/'
path2 = 'D:/U de A/PDI/Project2/templates/'
name = 'img12.bmp'
img = cv2.imread(path + name)

#128x250
#256x500
#64x175

template = cv2.imread(path2 + 'template11.jpg')

img_g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

#img_bin = cv2.adaptiveThreshold(img_g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 201,5)
#template_bin = cv2.adaptiveThreshold(template, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 201,5)


w, h = template.shape[::-1]
r,c = img_g.shape
tr,tc = template.shape
#img_g = cv2.equalizeHist(img_g)

match = cv2.matchTemplate(template, img_g, cv2.TM_CCORR_NORMED)
loc = np.where(match <= 0.9928)
#0.9928 T 11
img_toshow = cv2.merge([img_g,img_g,img_g])

img_mask = img_g*0

for pt in zip(*loc):
    cv2.rectangle(img_mask, (pt[1],pt[0]), (pt[1]+w,pt[0]+h), (255,255,255), -1)

_, contours,_ = cv2.findContours(img_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)


cv2.drawContours(img_toshow, contours,-1, (0,0,255), 3)

#cv2.rectangle(img_toshow, (0,0), (200,2000), (255,0,0), 10)
img_toshow = cv2.resize(img_toshow, (int(c*0.15),int(r*0.15)), cv2.INTER_AREA)
template_toshow = cv2.resize(template, (int(tc*0.15),int(tr*0.15)), cv2.INTER_AREA)


cv2.imshow('a',img_toshow)
cv2.imshow('t',template_toshow)

cv2.waitKey(0)
cv2.destroyAllWindows()