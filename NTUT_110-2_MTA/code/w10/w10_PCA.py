# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 22:04:15 2022

@author: glanb
"""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
import matplotlib.pyplot as plt

data = np.load('iris_x.npy')
label = np.load('iris_y.npy')

qda = QuadraticDiscriminantAnalysis(store_covariance=True)
y_pred = qda.fit(data, label).predict(data)
acc = (y_pred==label).sum()/len(label)
print('acc(QDA):{:.2f}%'.format(acc*100))

pca = PCA(n_components=4)
pca.fit(data)
print('PC:{}'.format(pca.explained_variance_ratio_))
print(pca.singular_values_)
data_pca = pca.transform(data)
QuadraticDiscriminantAnalysis()
qda_pca = QuadraticDiscriminantAnalysis(store_covariance=True)
y_pred_pca = qda_pca.fit(data_pca, label).predict(data_pca)
acc = (y_pred_pca==label).sum()/len(label)
print('acc(PCA+QDA):{:.2f}%'.format(acc*100))

c=0
L=3
colors=['r','g','b']
for i in range(4):
    for j in range(4):
        c+=1
        for l in range(L):
            plt.subplot(4,4,c)
            pos = np.where(label==l)[0]
            plt.plot(data[pos,i],data[pos,j], colors[l]+'*')
            plt.title('f{} vs f{}'.format(i+1,j+1))
plt.show()

colors=['r','g','b']
for i in range(L):
    pos= np.where(label==i)[0]
    plt.plot(data_pca[pos,0],data_pca[pos,1], colors[i]+'*')
plt.title('PCA')
plt.show()
