# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 22:04:15 2022

@author: glanb
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt


# 2.1
label_train,label_test=[],[]
with open('train_label.txt','r') as f:
    for line in f.readlines():
        label_train.append(int(line))
with open('test_label.txt','r') as f:
    for line in f.readlines():
        label_test.append(int(line))
label_train = np.array(label_train)
label_test = np.array(label_test)

# 2.2
data_train = np.zeros((60000,784))
data_test = np.zeros((10000,784))

cap = cv2.VideoCapture('mnist_train_image.avi')
c=0
while cap.isOpened():
    ret, frame = cap.read()
    if ret==False:
        break
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = np.reshape(frame,(1,784))
    data_train[c,:]=frame
    c+=1
print('number of training:{}'.format(c))
cap.release()
cv2.destroyAllWindows()


cap = cv2.VideoCapture('mnist_test_image.avi')
c=0
while cap.isOpened():
    ret, frame = cap.read()
    if ret==False:
        break
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame = np.reshape(frame,(1,784))
    data_test[c,:]=frame
    c+=1
cap.release()
cv2.destroyAllWindows()
print('number of test:{}'.format(c))

# 2.3
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.metrics import confusion_matrix
# Linear Discriminant Analysis
lda = LinearDiscriminantAnalysis(store_covariance=True)
lda = lda.fit(data_train, label_train)

frame = cv2.imread('15.jpg')
frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
frame = np.reshape(frame, (1, 784))
lda.covariance_
means = lda.means_
# print(means)
# print(means.shape)
y_pred = lda.predict(frame)
y_pred = lda.predict_proba(frame)
print(y_pred)
pred =[]
for i in range(10):
    tmp_mean=means[i,:]
    v = np.sqrt(np.sum((frame-tmp_mean)**2))
    pred.append(v)
print(np.argmin(pred))

plt.plot(pred)
plt.show()


