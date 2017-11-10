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

def create_font(fontname='Tahoma', fontsize=10):
    return { 'fontname': fontname , 'fontsize':fontsize }

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
            plt.title("%s #%s" % (sptitle, str(sptitles[i])), create_font('Tahoma',8))
        else:
            plt.title("%s #%d" % (sptitle, (i+1)), create_font('Tahoma',8))
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
traingingFaceMatrix = np.empty((trainingFaces[0].size, 0), dtype='float64')
for col in trainingFaces:
    traingingFaceMatrix = np.hstack((traingingFaceMatrix ,np.asarray(col).reshape(-1, 1)))

averageFaceVector   = np.array(traingingFaceMatrix.mean(axis=1))
phiMatrix           = (traingingFaceMatrix.transpose() - averageFaceVector).transpose()
eigenMatrix         = np.dot(phiMatrix.T,phiMatrix)
[eigenvalues ,eigenvectors] = np.linalg.eigh(eigenMatrix)
idx                 = np.argsort(-eigenvalues)
eigenvalues         = eigenvalues[idx]
eigenvectors        = eigenvectors[:,idx]

### show the average face
#e = averageFace.reshape(imageMatrix[0].shape)
#eNorm = normalize(e,0,255)
#plt.imshow(eNorm)


#
#mat = np.dot(colMatrix.T,colMatrix)
#
#[eigenvalues ,eigenvectors] = np.linalg.eigh(mat)
#
#idx             = np.argsort(-eigenvalues)
#eigenvalues     = eigenvalues[idx]
#eigenvectors    = eigenvectors[:,idx]




## begin the PCA
## number of the columns elements
#num_components = colMatrix.shape[0]
## take the mean computed along the columns
#mu = colMatrix.mean(axis=1)
#

#
## subtract the mean from the columns
#colMatrix = (colMatrix.T - mu)
## take the dot product of the column matrix and the column matrix's transpose
#C = np.dot(colMatrix.T,colMatrix)
## calculate the eigenvalues and eigenvectors
#[eigenvalues, eigenvectors] = np.linalg.eigh(C)
## sort eigenvectors descending by their eigenvalue
#idx             = np.argsort(-eigenvalues)
#eigenvalues     = eigenvalues[idx]
#eigenvectors    = eigenvectors[:,idx]
## select only num_components
#eigenvalues     = eigenvalues[0:num_components].copy()
#eigenvectors    = eigenvectors[:, 0:num_components].copy()
#


    
    
def pca(X, y, num_components=0):
    [n,d] = X.shape
    if (num_components <= 0) or (num_components >n):
        num_components = n
    mu = X.mean(axis=0)
    X = X - mu
    if n>d:
        C = np.dot(X.T,X)
        [eigenvalues ,eigenvectors] = np.linalg.eigh(C)
        eigenvectors = np.dot(X,eigenvectors.T)
    else:
        C = np.dot(X,X.T)
        [eigenvalues ,eigenvectors] = np.linalg.eigh(C)
        eigenvectors = np.dot(X.T,eigenvectors)
        for i in range(n):
            eigenvectors[:,i] = eigenvectors[:,i]/np.linalg.norm(eigenvectors[:,i])

    idx = np.argsort(-eigenvalues)
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:,idx]
    eigenvalues = eigenvalues[0:num_components].copy()
    eigenvectors = eigenvectors[:,0:num_components].copy()
    return [eigenvalues , eigenvectors , mu]

def asRowMatrix(X):
    if len(X) == 0:
        return np.array([])
    mat = np.empty((0, X[0].size), dtype=X[0].dtype) 
    for row in X:
        mat = np.vstack((mat, np.asarray(row).reshape(1,-1)))
    return mat

def asColumnMatrix(X):
    if len(X) == 0:
        return np.array([])
    mat = np.empty((X[0].size, 0), dtype=X[0].dtype)
    for col in X:
        mat = np.hstack((mat, np.asarray(col).reshape(-1,1)))
    return mat

#[eigenvalues , eigenvectors , mu] = pca(asColumnMatrix(X), y)







#E = []
#for i in range(min(len(X), 16)):
#    e = eigenvectors[:,i].reshape(X[0].shape)
#    E.append(normalize(e,0,255))
#    # plot them and store the plot to "python_eigenfaces.pdf"
#    
#subplot(title="Eigenfaces", images=E, rows=4, \
#        cols=4, sptitle=" Eigenface", colormap=cm.binary, \
#        filename="python_pca_eigenfaces.png")



