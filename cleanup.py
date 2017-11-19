#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 12:19:16 2017

@author: tbrady
"""

import sys, os
sys.path.append("../..")
import numpy as np
from PIL import Image
import matplotlib.cm as cm
import matplotlib.pyplot as plt


def printVector(v, dims, title=None):
    plt.figure()
    e = v.reshape(dims)     # reshape to original image dimensions
    e = normalize(e,0,255)  # convert to grayscale range
    plt.axis('off')         # turn off axis
    plt.title(title)
    plt.imshow(e, cmap=cm.gray)
    
def normalize(arr, low, high):
    arr     = np.asarray(arr)
    minVal  = np.min(arr)   # find minimum value
    maxVal  = np.max(arr)   # find maximum value
    arr     = (arr - float(minVal)) / float((maxVal - minVal))  # normalize
    arr     = (arr * (maxVal - minVal)) + minVal                # scale from low to high
    return np.asarray(arr) 


def subplot(title, images, sptitle='subplot', colormap=cm.gray, num=None):
    fig = plt.figure()
    fig.suptitle(title, horizontalalignment='center')
    if num == None:
        for i in range(len(images)):
            fig.add_subplot(2, 4,(i+1))
            plt.title('#%d %s' % ((i+1), sptitle), fontsize = 10)
            plt.imshow(np.asarray(images[i]), cmap=colormap)
            plt.axis('off')
    else:
        for i in range(len(images)):
            fig.add_subplot(2, 4,(i+1))
            plt.title('%d %s' % (num[i], sptitle), fontsize = 10)
            plt.imshow(np.asarray(images[i]), cmap=colormap)
            plt.axis('off')


# read in all the images
def readImages(path): 
    c = 0
    X,y = [], []
    for dirname , dirnames , filenames in os.walk(path):
        for subdirname in dirnames:
            subject_path = os.path.join(dirname , subdirname)
            for filename in os.listdir(subject_path):
                try:
                    im = Image.open(os.path.join(subject_path, filename))
                    im = im.convert("L")
                    X.append(np.asarray(im, dtype=np.float))
                    y.append(c)
                except IOError as err:
                    print ("I/O error: {0}".format(err))
                except:
                    print ("Unexpected error:", sys.exc_info()[0])
                    raise
                c = c+1
    return [X,y]