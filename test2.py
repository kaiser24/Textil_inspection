# -*- coding: utf-8 -*-
"""
Created on Fri Sep  6 10:49:45 2019

@author: luisf
"""

# =============================================================================
# DIP Project. Implementing Bitplane Decomposition
#
# =============================================================================

import cv2
import numpy as np
import matplotlib.pyplot as plt

root = 'D:/U de A/PDI/Project2/'

img = cv2.imread(root + 'ejemplo1.JPG')
template = cv2.imread(root + 'segment.jpg')

img_g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

cv2.imshow('Gray', img_g)

img_eq = cv2.equalizeHist(img_g)

result = cv2.matchTemplate(template, img_g, cv2.TM_CCORR_NORMED)

cv2.imshow('CROSS CORRELATION', result)

cv2.waitKey(0)
cv2.destroyAllWindows()