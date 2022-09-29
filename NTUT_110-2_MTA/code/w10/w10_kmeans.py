# -*- coding: utf-8 -*-
"""
Created on Sun Mar 20 20:08:52 2022

@author: glanb
"""
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

filename='data_class.txt'
# 1. txt reader
with open(filename,'r', encoding="utf-8") as f:
    for line in f.readlines():
        print(line)
data=[]
label=[]
with open(filename,'r', encoding="utf-8") as f:
    for line in f.readlines():
        line = line.split('\t')
        data.append([float(line[0]),float(line[1])])
        if '男' in line[2]:
            label.append(1)
        else:
            label.append(0)
data = np.array(data)
label = np.array(label)

kmc = KMeans(n_clusters=2)
pred = kmc.fit(data)
pred_label = pred.labels_

plt.figure()
pos0= np.where(label==0)[0]
plt.plot(data[pos0,0],data[pos0,1],'b^')
pos1= np.where(label==1)[0]
plt.plot(data[pos1,0],data[pos1,1],'r^')

pos0= np.where(pred_label==0)[0]
plt.plot(data[pos0,0],data[pos0,1],'b+')
pos1= np.where(pred_label==1)[0]
plt.plot(data[pos1,0],data[pos1,1],'r+')
plt.show()

#
num_k=3
colors=['b','r','k']
kmc = KMeans(n_clusters=num_k)
pred = kmc.fit(data)
pred_label = pred.labels_
for i in range(num_k):
    pos= np.where(pred_label==i)[0]
    plt.plot(data[pos,0],data[pos,1],colors[i]+'*')
plt.show()

    
    
# kmeans.predict([[0, 0], [12, 3]])
# kmeans.cluster_centers_
