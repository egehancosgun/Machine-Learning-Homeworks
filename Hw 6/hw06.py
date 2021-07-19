#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 11 21:31:38 2021

@author: egehancosgun
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import sqrtm
from scipy.linalg import eigh

X = np.genfromtxt("hw06_data_set.csv", delimiter = ",",skip_header=1)
N = X.shape[0]
D = X.shape[1]

#Plotting the dataset
plt.scatter(X[:,0],X[:,1],color = "Black")
plt.xlabel("x1")
plt.ylabel("x2")
plt.show()

#Calculating Distances and Connectivity Natrix

delta = 1.25
B = np.zeros((300,300))
for i in range(300):
    for j in range(300):
        if np.linalg.norm(X[i]-X[j]) <= 1.25 and i != j:
            B[i,j] = 1
        
#Plot Connectivity Matrix
plt.figure(figsize = (12,10))
for i in range(300):
    for j in range(300):
        if B[i,j] == 1 and i < j:
            x1_values = [X[i,0], X[j,0]]
            x2_values = [X[i,1], X[j,1]]
            plt.plot(x1_values, x2_values, color = "Black", marker = ".", markersize = 12)
plt.xlabel("x1")
plt.ylabel("x2")
plt.show()

#Calculating Laplacian 
D = np.zeros((300,300))
for i in range(300):
    D[i,i] = B.sum(axis=1)[i]
L_symmetric = np.identity(300) - np.matmul(np.matmul(sqrtm(np.linalg.inv(D)),B),sqrtm(np.linalg.inv(D)))

#Calculating Eigenvectors

eig_vals, eig_vecs = eigh(L_symmetric)
Z = eig_vecs[:5].transpose()
initial_points = [85,129,167,187,270]
centroids = Z[initial_points]

#Initial Clusters
initial_clusters = np.zeros((300,1))
for i in range(300):
    initial_clusters[i] = np.argmin([np.linalg.norm(Z[i]-centroids[j]) for j in range(5)])    

Z_clusters = np.hstack((Z,initial_clusters))

#Visualization of Clusters (At initialization)

plt.scatter(X[:,0],X[:,1], c = initial_clusters)
plt.xlabel("x1")
plt.ylabel("x2")
plt.xticks([-6,-4,-2,0,2,4,6])
plt.yticks([-6,-4,-2,0,2,4,6])
plt.show()

#kNN Algorithm
while True:
    old_centroids = centroids
    centroids = np.vstack([np.mean(Z[Z_clusters[:,5] == i],axis = 0) for i in range(5)])
    if np.alltrue(centroids == old_centroids):
        break
    new_clusters = np.zeros((300,))
    for i in range(300):
        new_clusters[i] = np.argmin([np.linalg.norm(Z[i]-centroids[j]) for j in range(5)])
    Z_clusters[:,5] = new_clusters
    if np.alltrue(new_clusters == initial_clusters):
        break
    
plt.scatter(X[:,0],X[:,1], c = new_clusters)
plt.xlabel("x1")
plt.ylabel("x2")
plt.xticks([-6,-4,-2,0,2,4,6])
plt.yticks([-6,-4,-2,0,2,4,6])
plt.show()
    


