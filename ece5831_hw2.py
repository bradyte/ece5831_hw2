#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 21:25:26 2017

@author: tombrady
"""

import numpy as np
from cleanup import *


if 1:
    # read in the faces and number of images
    [trainingFaces,trainingFacesCount] = readImages('/Users/tbrady/drive/sw/att_faces/')
    
    # dimensions of the picture size
    dims = trainingFaces[0].shape
    
    # initialize an empty vector of m x n length
    L = np.empty((trainingFaces[0].size, 0), dtype='float64')
    
    # create a matrix, gamma, of image vectors as long as the number of faces
    for col in trainingFaces:
        L = np.hstack((L ,np.asarray(col).reshape(-1, 1)))
    
    # average the vectors to create and average face
    meanVector  = np.array(L.mean(axis=1))
    
    # subtract the mean vector from the original images
    #t he double transpose is due to how the vectors are handled in numpy
    A       = (L.T - meanVector).T
    
    # the dot product of transposed difference matrix and original matrix
    C       = np.dot(A.T,A)
    
    # calculate the eigenvectors and eigenvalues
    [v, u]  = np.linalg.eig(C)
    
    # sort them by largest eigenvalues
    idx     = np.argsort(-v)
    
    # reorganize the eigenvalues by the sorted index
    v       = v[idx]
    
    # reorganize the eigenvectors by the sorted index
    u       = u[:,idx]
    
    # create the eigen faces
    U       = np.dot(A, u) 
    
    # normalize the eigenface matrix U   
    U       = U / np.linalg.norm(U, axis=0)

## display eigenfaces
#numFaces = 8
#E = []
#for i in range(0, numFaces):
#    e = U[:,i].reshape(dims)
#    E.append(normalize(e,0,255))
#
##printVector(meanVector, dims)
#
#subplot(title="Eigenfaces", images=E, rows=numFaces/4, \
#        cols=4, sptitle=" Eigenface", colormap=cm.gist_yarg, \
#        filename="python_pca_eigenfaces.png")


### reconstruct random test face from computed eigenfaces
#[testFace, count] = readImages('/Users/tbrady/drive/sw/test_faces/')
#testFace = np.array(testFace)
#T = np.empty((testFace[0].size, 0), dtype='float64')
#T = np.reshape(testFace, testFace[0].size)
#
#faces = 400
#
#reconFace = meanVector
##
#psi = T - meanVector
#w = np.dot(U[:,0].T, psi)
#w = np.dot(U[:,0:faces].T, psi)
#for i in range(0,faces):
#    reconFace += w[i]*U[:,i]
#
#printVector(reconFace, dims)


