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


def printVector(v, dims):
    e = v.reshape(dims)
    e = normalize(e,0,255)
    plt.axis('off')
    plt.imshow(e, cmap=cm.gray)
    
def normalize(X, low, high):
    X = np.asarray(X)
    minX , maxX = np.min(X), np.max(X)
    X = (X - float(minX)) / float((maxX - minX))        # normalize
    X = X * (high-low) + low                            # scale from low to high
    return np.asarray(X) 


def subplot(title, images, sptitle='subplot', colormap=cm.gray):
    fig = plt.figure()
    fig.suptitle(title, horizontalalignment='center')
    
    for i in range(len(images)):
        ax0 = fig.add_subplot(2, 4,(i+1))
        plt.title('#%d %s' % ((i+1), sptitle), fontsize = 10)
        plt.imshow(np.asarray(images[i]), cmap=colormap)
        plt.axis('off')


# read in all the images
def readImages(path, sz=None): 
    c = 0
    X,y = [], []
    for dirname , dirnames , filenames in os.walk(path):
        for subdirname in dirnames:
            subject_path = os.path.join(dirname , subdirname)
            for filename in os.listdir(subject_path):
                try:
                    im = Image.open(os.path.join(subject_path , filename))
                    im = im.convert("L")
                    # resize to given size (if given)
                    if (sz is not None):
                        im = im.resize(sz, Image.ANTIALIAS)
                    X.append(np.asarray(im, dtype=np.uint8))
                    y.append(c)
                except IOError as err:
                    print ("I/O error: {0}".format(err))
                except:
                    print ("Unexpected error:", sys.exc_info()[0])
                    raise
                c = c+1
    return [X,y]