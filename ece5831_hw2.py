#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 21:25:26 2017

@author: tombrady
"""
import sys, os
sys.path.append("../..")
import numpy as np
from PIL import Image
import matplotlib.cm as cm
import matplotlib.pyplot as plt


def normalize(X, low, high, dtype=None):
    X = np.asarray(X)
    minX , maxX = np.min(X), np.max(X)
    # normalize to [0...1].
    X = X - float(minX)
    X = X / float((maxX - minX))
    # scale to [low...high].
    X = X * (high-low)
    X = X + low
    if dtype is None:
        return np.asarray(X)
    return np.asarray(X, dtype=dtype)    

def subplot(title, images, rows, cols, sptitle="subplot", \
            sptitles=[], colormap=cm. gray, ticks_visible=True,\
            filename=None):
    fig = plt.figure()
    # main title
    fig.text(.5, .95, title, horizontalalignment='center')
    for i in range(len(images)):
        ax0 = fig.add_subplot(rows,cols,(i+1))
        plt.setp(ax0.get_xticklabels(), visible=False)
        plt.setp(ax0.get_yticklabels(), visible=False)
        if len(sptitles) == len(images):
            plt.title("%s #%s" % (sptitle, str(sptitles[i])))
        else:
            plt.title("%s #%d" % (sptitle, (i+1)))
        plt.imshow(np.asarray(images[i]), cmap=colormap)
    if filename is None:
        plt.show()
    else:
        fig.savefig(filename)

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


[trainingFaces,trainingFacesNum] = readImages('/Users/tbrady/drive/sw/att_faces/')

# create a matrix of y columns where each image is stored in one row
L = np.empty((trainingFaces[0].size, 0), dtype='float64')
for col in trainingFaces:
    L = np.hstack((L ,np.asarray(col).reshape(-1, 1)))

averageFaceVector   = np.array(L.mean(axis=1))

phiMatrix           = (L.transpose() - averageFaceVector).transpose()

eigenMatrix         = np.dot(phiMatrix.T,phiMatrix)
[eigenvalues ,eigenvectors] = np.linalg.eig(eigenMatrix)
idx                 = np.argsort(-eigenvalues)
eigenvalues         = eigenvalues[idx]
eigenvectors        = eigenvectors[:,idx]


numComponents       = 50
eigenvalues         = eigenvalues[0:numComponents].copy()
eigenvectors        = eigenvectors[:, 0:numComponents].copy()

eigenvectors        = np.dot(L, eigenvectors)
#norms               = np.linalg.norm(eigenvectors, axis=0)
#eigenvectors        /= norms



def printVector(v):
    e = v.reshape(trainingFaces[0].shape)
    e = normalize(e,0,255)
    plt.imshow(e)

E = []
for i in range(min(len(L), 4)):
    e = eigenvectors[:,i].reshape(trainingFaces[0].shape)
    E.append(normalize(e,0,255))
#    # plot them and store the plot to "python_eigenfaces.pdf"

subplot(title="Eigenfaces", images=E, rows=1, \
        cols=4, sptitle=" Eigenface", colormap=cm.binary, \
        filename="python_pca_eigenfaces.png")

##steps = [i for i in range(10, min(len(trainingFaceMatrix),320), 20)]
steps = [10,20]
E = []
for i in range(min(len(steps),16)):
    numEvs = steps[i]
#    P = project(eigenvectors[:,0:numEvs],trainingFaces[0].reshape(-1, 1) , averageFaceVector)
    dotFace = np.dot(L[:,0:numEVs] - averageFaceVector, eigenvectors[:,0:numEVs])
#    R = reconstruct(eigenvectors[:,0:numEvs ], P , averageFaceVector)
#    
#    
#    R = R.reshape(trainingFaces[0].shape)
#    E.append(normalize(R,0,255))






w = np.dot(eigenvectors[:,0], trainingFaceMatrix[:,0] - averageFaceVector)
printVector(averageFaceVector + np.dot(dotFace,eigenvectors[:,0].T))

#
#def project (W, X ,mu = None):
#    if mu is None :
#        return np.dot(X, W)
#    return np.dot(X - mu, W)
#
#def reconstruct (W, Y, mu = None ):
#    if mu is None:
#        return np.dot(Y, W.T)
#    return np.dot(Y, W.T) + mu
#
##steps = [i for i in range(10, min(len(trainingFaceMatrix),320), 20)]
#steps = [10,20]
#E = []
#for i in range(min(len(steps),16)):
#    numEvs = steps[i]
#    P = project(eigenvectors[:,0:numEvs],trainingFaces[0].reshape(-1, 1) , averageFaceVector)
#    R = reconstruct(eigenvectors[:,0:numEvs ], P , averageFaceVector)
#    
#    
#    R = R.reshape(trainingFaces[0].shape)
#    E.append(normalize(R,0,255))



























