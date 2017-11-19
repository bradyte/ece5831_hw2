#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 21:25:26 2017

@author: tombrady
"""
import numpy as np
from cleanup import *
from matplotlib import pyplot as plt


## read in faces
fpath   = '/Users/tbrady/drive/sw/att_faces/'
[tFaces,tFacesCount] = readImages(fpath)    # read in the faces and number of images
dims    = tFaces[0].shape                   # dimensions m, n of the picture size
vecLen  = tFaces[0].size                    # length of the gamma vector m x n
## begin PCA 
L       = np.empty((vecLen, 0), dtype='float64') # initialize an empty vector of m x n length
for col in tFaces:                          # create a matrix, gamma, of image vectors as long as the number of faces
    L   = np.hstack((L ,np.asarray(col).reshape(-1, 1)))
meanVector  = np.array(L.mean(axis=1))      # average the vectors to create and average face
printVector(meanVector, dims, 'Average Face') # display the average face
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
E           = []                                # new list containing faces
for i in range(0, numFaces):
    e       = U[:,i].reshape(dims)              # reshape the vector to the original picture size m x n
    E.append(normalize(e,0,255))                # append to the face array
subplot(title="Eigenfaces", images=E, sptitle=" Eigenface", colormap=cm.gist_yarg)

## reconstruct face from training set
T           = L[:,0]

## reconstruct random test face from computed eigenfaces
#k          = 1
#[testFace, count] = readImages('/Users/tbrady/drive/sw/test_faces/')
#testFace    = np.array(testFace)
#T           = np.empty((testFace[0].size, 0), dtype='float64')
#T           = np.reshape(testFace[k], testFace[0].size)

## reconstruct faces from eigenfaces
# number of eigenfaces to use    
faces       = [10, 30, 50, 70, 90, 110, 130, 150]
#faces       = [50, 100, 150, 200, 250, 300, 350, 400]
R           = []                                # new list for reconstructed faces  
err         = []                                # new list for rmse values

for j in range(0,len(faces)):
    reconFace   = meanVector                    # start with the average face
    w       = 0
    rmse    = 0
    rerr    = 0
    w       = np.dot(U[:,0:faces[j]].T, T - meanVector) # calculate the weight w
    for i in range(0,faces[j]):
        reconFace += w[i]*U[:,i]                # sum together the weights x eigenvectors
    rerr    = normalize(reconFace,0,255)    #reconFace normalized vector
    r       = reconFace.reshape(dims)       # add face to list
    R.append(normalize(r,0,255))
## calculate the RMSE
    rmse    = np.sum((rerr.astype('float') - T.astype('float')) ** 2)
    rmse    /= float(rerr.shape[0] * T.shape[0])
    rmse    = np.sqrt(rmse)
    err.append(rmse)

subplot(title="Reconstructed Face", images=R, sptitle="Eigenfaces", num=faces)
plt.figure()
plt.title('Eigenface Reconstruction')
plt.ylabel('RMSE')
plt.xlabel('Sets of eigenfaces')
plt.plot(err)