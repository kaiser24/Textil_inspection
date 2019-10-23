# -*- coding: utf-8 -*-
"""
Created on Mon Oct  7 17:43:10 2019

@author: luisf
"""
cimport cython
from libc.math cimport sqrt

# Energia Local
@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing.

                                                           #En cython, delcarar las variables a emplear 
                                                           #incrementa enormemente el desempeño
cpdef double[:, :] localEnergy(double[:, :] sum_1,double[:, :] sqsum_2, double[:, :] img_eng, int r_img, int c_img, int wsize):
    cdef int n, i, j
    cdef double s1, s2
    
    n = wsize**2
    r_end = r_img - (wsize -1)
    c_end = c_img - (wsize -1)

    
                                                           #Pasando la ventana por toda la imagen
    for i in range(0, r_end ):
        for j in range(0, c_end):
                                                           #Con la imagen integral la suma de los 
                                                           #elementos en una ventana es solo una suma de 4 elementos (en lugar de sumarlos todos).
                                                           #Lo que incrementa enormente e desempeño (Se reducen cientos de sumas a solo 4)
            s1 = sum_1[i + wsize,j + wsize ] + sum_1[i ,j] - sum_1[i,j + wsize ] - sum_1[i + wsize,j]
            s2 = sqsum_2[i + wsize,j + wsize ] + sqsum_2[i,j] - sqsum_2[i,j + wsize ] - sqsum_2[i + wsize,j]
                                                           
                                                           #Se guarda el calculo y se itera nuevamente
            img_eng[i,j] = sqrt( (s2 - (s1**2)/n)/n )

    
    return img_eng
#%%