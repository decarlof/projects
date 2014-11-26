# -*- coding: utf-8 -*-
"""imagesc.py: Imagesc for text matrices"""
# to visualize the ring removal by:

__author__     = "Eduardo X Miqueles"
__copyright__  = "Copyright 2014, CNPEM/LNLS" 
__maintainer__ = "Eduardo X Miqueles"
__email__      = "edu.miqueles@gmail.com" 
__date__       = "January, 2014" 

from pylab import *
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def showi(img):
    imshow(img, extent=[0,100,0,1], aspect=100)
    colorbar()
    show()

def imagesc(*args):

    N = len(args)

    if N==0 or N==5:
        print ("imagesc error: max of 4 images")
        return
    
    if N==1:
        showi(args[0])
        return
    
    if N==2:
        plt.subplot(1, 2, 1)
        imshow(args[0], extent=[0,100,0,1], aspect=100)
        colorbar()
        plt.subplot(1, 2, 2)
        imshow(args[1], extent=[0,100,0,1], aspect=100)
        colorbar()
        show()
        return
        
    if N==3:
        plt.subplot(2, 2, 1)
        imshow(args[0], extent=[0,100,0,1], aspect=100)
        colorbar()
        plt.subplot(2, 2, 2)
        imshow(args[1], extent=[0,100,0,1], aspect=100)
        colorbar()
        plt.subplot(2, 2, 3)
        imshow(args[2], extent=[0,100,0,1], aspect=100)
        colorbar()
        show()
        
        return
        

    if N==4:
        plt.subplot(2, 2, 1)
        imshow(args[0], extent=[0,100,0,1], aspect=100)
        colorbar()
        plt.subplot(2, 2, 2)
        imshow(args[1], extent=[0,100,0,1], aspect=100)
        colorbar()
        plt.subplot(2, 2, 3)
        imshow(args[2], extent=[0,100,0,1], aspect=100)
        colorbar()
        plt.subplot(2, 2, 4)
        imshow(args[3], extent=[0,100,0,1], aspect=100)
        colorbar()
        show()
        
        return       

