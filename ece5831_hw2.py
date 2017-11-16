#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 21:25:26 2017

@author: tombrady
"""
import numpy as np
from cleanup import *


if 1:
## read in faces
    fpath   = '/Users/tbrady/drive/sw/att_faces/'
    [tFaces,tFacesCount] = readImages(fpath)    # read in the faces and number of images
    dims    = tFaces[0].shape                   # dimensions m, n of the picture size
    vecLen  = tFaces[0].size                    # length of the gamma vector m x n

## being PCA 
    L = np.empty((vecLen, 0), dtype='float64')  # initialize an empty vector of m x n length
    for col in tFaces:                          # create a matrix, gamma, of image vectors as long as the number of faces
        L = np.hstack((L ,np.asarray(col).reshape(-1, 1)))
    meanVector  = np.array(L.mean(axis=1))      # average the vectors to create and average face
#    printVector(meanVector, dims)               # display the average face
    A       = (L.T - meanVector).T              # subtract the mean vector from the original images
    C       = np.dot(A.T,A)                     # the dot product of transposed difference matrix and original matrix
    [v, u]  = np.linalg.eig(C)                  # calculate the eigenvectors and eigenvalues
    idx     = np.argsort(-v)                    # sort them by largest eigenvalues
    v       = v[idx]                            # reorganize the eigenvalues by the sorted index
    u       = u[:,idx]                          # reorganize the eigenvectors by the sorted index
    U       = np.dot(A, u)                      # create the eigenfaces
    U       = U / np.linalg.norm(U, axis=0)     # normalize the eigenface matrix U 

## display eigenfaces
numFaces    = 8                                 # number of faces to show
E           = []                                # new array containing faces
for i in range(0, numFaces):
    e       = U[:,i].reshape(dims)              # reshape the vector to the original picture size m x n
    E.append(normalize(e,0,255))                # append to the face array
subplot(title="Eigenfaces", images=E, sptitle=" Eigenface", colormap=cm.gist_yarg)

### reconstruct random test face from computed eigenfaces
#[testFace, count] = readImages('/Users/tbrady/drive/sw/test_faces/')
#testFace = np.array(testFace)
#T = np.empty((testFace[0].size, 0), dtype='float64')
#T = np.reshape(testFace, testFace[0].size)
#

## face to reconstruct 
#T = L[:,0]
## number of eigenfaces to use    
#faces = [10, 50, 100, 150, 200, 250, 300, 350]
## start with the average face
#reconFace = meanVector
#
#psi = T - meanVector
#
##for j in range(0, len(faces)):
#R = []
#reconFace = meanVector
#j = 3
#w = np.dot(U[:,0:faces[j]].T, psi)
#for i in range(0,faces[j]):
#    reconFace += w[i]*U[:,i]
#
##printVector(reconFace, dims)
