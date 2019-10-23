# -*- coding: utf-8 -*-
"""
Created on Sun Sep 29 21:53:52 2019

@author: luisf
"""

import cv2
import numpy as np
import os


path = '/media/felipe/A4763C84763C58EE/U de A/PDI/Project2/Textil_inspection_linux/Imagenes/'
path2 = '/media/felipe/A4763C84763C58EE/U de A/PDI/Project2/Textil_inspection_linux/templates/'

template = cv2.imread(path2 + 'template12.jpg')
template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

w, h = template.shape[::-1]

for name in os.listdir(path):
    img = cv2.imread(path + name)
    
    #img = cv2.GaussianBlur(img,(5,5),0)
    
    img_g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_go = img_g
    #img_g = cv2.equalizeHist(img_g)
    
    
    r,c = img_g.shape
    tr,tc = template.shape
    
    match = cv2.matchTemplate(template, img_g, cv2.TM_CCORR_NORMED)
    loc = np.where(match <= 0.99)
    #0.9928 T 11
    img_toshow = cv2.merge([img_g,img_g,img_g])
    
    img_mask = img_g*0
    
    for pt in zip(*loc):
        cv2.rectangle(img_mask, (pt[1],pt[0]), (pt[1]+w,pt[0]+h), (255,255,255), -1)
    kernel = np.ones((13,13), np.uint8)
    img_mask = cv2.erode(img_mask, kernel)
    
    contours,_ = cv2.findContours(img_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    
    cv2.drawContours(img_toshow, contours,-1, (0,0,255), 3)
    
    img_toshow = cv2.resize(img_toshow, (int(c*0.15),int(r*0.15)), cv2.INTER_AREA)
    
    cv2.imshow('s1',img_toshow)
    cv2.waitKey(0)
cv2.destroyAllWindows()