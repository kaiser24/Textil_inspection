#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 18:31:20 2019

@author: felipe
"""

import cv2
import numpy as np
import multiprocessing as mp
import psutil
import os
from fast_localEnergy import localEnergy

#Tuvimos que implementar esta función en cython para optimizarla.
# de ~4 min a ~1s

# DEBE SER EJECUTADA EN LINUX DEBIDO A QUE EL ARCHIVO CYTHON (.pyx) FUE COMPILADO
# PARA LINUX (.so) (para windows genera un archivo .pyd)

#========================Función para eliminar sombras=========================
def shadows_remover(img):
                                                    #Se Dilata para eliminar formas
                                                    #y luego se aplica un filtro de media
                                                    #para generar un buen fondo
    img_d = cv2.dilate(img, np.ones((4,4), np.uint8), iterations = 2)
    img_b = cv2.medianBlur(img_d, 15)
                                                    #Se le resta el fondo a la imagen
    diff_img = 255 - cv2.absdiff(img, img_b)
                                                    #Se normaliza
    norm_img = diff_img.copy() # Needed for 3.x compatibility
    cv2.normalize(diff_img, norm_img, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
                                                    
                                                    #Se aplica un umbral truncado
                                                    #y se normaliza de nuevo
    _, thr_img = cv2.threshold(norm_img, 230, 0, cv2.THRESH_TRUNC)
    cv2.normalize(thr_img, thr_img, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    return thr_img

#================================Funcion de La energia Local===================
#Este es el punto más critico de la implementación, aqui se llama el archivo de
#cython
def fast_localEnergy(args): 
                                                    #Se recibe la imagen y se toman sus dimensiones    
    img, wsize = args
    r, c = img.shape
                                                    #Tamaño del borde a regenerar
    bor = int((wsize - 1) /2)
                                                    #Se obtienen la integral y la integral
                                                    #cuadrada de la imagen para realizar los
                                                    #calculos mucho mas rapido
    sum_1, sqsum_2 = cv2.integral2(img)
                                                    #Imagen donde se guardará el resultado
    img_eng = np.zeros((r - (wsize-1) ,c - (wsize -1)), np.float64)
                                                    
                                                    #Se llama la función implementada en cython
                                                    #y se transforma de nuevo a tipo array de numpy
    img_eng = np.array(localEnergy(sum_1.astype(np.float), sqsum_2,img_eng,r,c , wsize) ,np.float32)
                                                    #Se extienden los bordes para que quede del tamaño
                                                    #original
    img_eng = cv2.copyMakeBorder(img_eng, bor, bor,bor,bor, cv2.BORDER_REPLICATE)
    return img_eng


#============================== Procesamiento =================================
if __name__=='__main__':                            
    
#============================= Cargamos la ruta ===============================
    path = '/media/felipe/A4763C84763C58EE/U de A/PDI/Project2/Textil_inspection_linux/Imagenes/'
    
    for name in os.listdir(path):
        img_gi = cv2.imread(path + name,0)              #Iteramos sobre los archivos de la carpeta                
                                                        # y Leemos cada imagen
        #img_gi = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  #Se pasa a Gris
                                                        #Y eliminamos las sombras
        img_g = shadows_remover(img_gi)
        r,c = img_g.shape

                                                        #Parametros de los Filtros de gabor
        ksizes = [100]                                  #Tamaño de los filtros
        sigmas = 0.064                                  
        thetas = [np.pi, np.pi/2, 0]                    #Orientaciones de los filtros
        lambd = 15 
         
        wsize = 51                                      #Tamaño de la ventana para las caracteristicas 
        
#=============================Banco de Filtro de Gabor=========================
        filtered = []
        #pfiltered = []
        nf = 0
        for size in ksizes:                             #Iteramos variando la orientación del filtro
            for th in thetas:                           #y guardando la imagen generada
                gabor = cv2.getGaborKernel((size,size), sigmas, th, lambd, 0.075, 0, ktype = cv2.CV_32F) 
                filtered.append( cv2.filter2D(img_g, cv2.CV_8UC3, gabor) )
                nf = nf + 1
        
        num_cpus = psutil.cpu_count(logical=True)       #Se obtiene el numero de nucleos 
        pool = mp.Pool(num_cpus)                        #para distribuir el trabajo

#=================Integral de imagen y Calculo de Caracteristicas==============                                          
                                                        #Se distribuye a cada nucleo una parte de los calculos
        filtered = pool.map(fast_localEnergy, zip(filtered, nf*[wsize]) )
        
        pool.terminate()                                #Esto hace que se libere la memoria usada por los cores
                                                        #Sin esto la ram se va llenando cada iteracion
        
                                                        #El resultado de cada imagen obtenida son las
                                                        #caracteristicas de los pixeles que usaremos para clasificarlos
        features = cv2.merge(filtered)
        features = features.reshape((-1,nf))
        
#%%
#===========================CLUSTERING=========================================
                                                        #Criterios de parada y demás para el agrupamiento
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        
        clusters = 2                                    #Numero de Clusters. cada grupo es una textura
                                                        #En este caso solo 2. parte de la tela o no.     
        attempts=10
        
                                                        #Se aplica el metodo de clustering KMeans
        ret,label,center=cv2.kmeans(features,clusters,None,criteria,attempts,cv2.KMEANS_PP_CENTERS)
        

#%%                                                     #Se genera una mascara Según la clasificación anterior
        mul = int(255 / (clusters-1) )
        img_seg = label.reshape((img_g.shape))
        img_seg = np.uint8(img_seg * mul)
        img_mask = img_seg

#==========================Detección de Contornos==============================
                                                        #Se detectan los contornos en lamascara
                                                        #y se remarcan sobre la imagen original  
        contours,_ = cv2.findContours(img_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        
        img_toshow = [img_gi,img_gi,img_gi]
        img_toshow = cv2.merge(img_toshow)
        cv2.drawContours(img_toshow, contours,-1, (0,0,255), 8)

        img_toshow = cv2.resize(img_toshow, (int(c*0.15),int(r*0.15)), cv2.INTER_AREA)
        
        cv2.imshow('Procesado',img_toshow)
        cv2.waitKey(0)
        
        cv2.destroyAllWindows()