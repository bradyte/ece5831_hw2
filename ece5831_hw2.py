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
    
#    for col in L:
#        L[:,int(col)] = (L[:,int(col)]/ np.linalg.norm(L[:,int(col)], axis=1))
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
    U       = (U.T / np.linalg.norm(U, axis=1)).T


numFaces = 8
E = []
for i in range(0, numFaces):
    e = U[:,i].reshape(dims)
    E.append(normalize(e,0,255))

#printVector(meanVector, dims)

subplot(title="Eigenfaces", images=E, rows=numFaces/4, \
        cols=4, sptitle=" Eigenface", colormap=cm.gist_yarg, \
        filename="python_pca_eigenfaces.png")


#faces = 30
#reconFace = meanVector
##
#psi = L[:,0] - meanVector
#w = np.dot(U[:,0].T, psi)
#w = np.dot(U[:,0:faces].T, psi)
#for i in range(0,faces):
#    reconFace += w[i]*U[:,i]
#
#printVector(reconFace, dims)

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

