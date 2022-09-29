# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 22:04:15 2022

@author: glanb
"""
import numpy as np
import cv2

label_train,label_test=[],[]
with open('train_label.txt','r') as f:
    for line in f.readlines():
        label_train.append(int(line))
with open('test_label.txt','r') as f:
    for line in f.readlines():
        label_test.append(int(line))
label_train = np.array(label_train)
label_test = np.array(label_test)
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

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
import time
qda = QuadraticDiscriminantAnalysis(store_covariance=True)
qda.fit(data_train, label_train)
y_pred=qda.predict(data_train)
acc = (y_pred==label_train).sum()/len(label_train)
print('acc(train, QDA):{:.2f}%'.format(acc*100))
y_pred = qda.predict(data_test)
acc = (y_pred==label_test).sum()/len(label_test)
print('acc(test, QDA):{:.2f}%'.format(acc*100))
st = time.time()
pca = PCA(n_components=150)
pca.fit(data_train)
data_pca = pca.transform(data_train)
data_pca_test = pca.transform(data_test)
pos = 75
qda_pca = QuadraticDiscriminantAnalysis(store_covariance=True)
y_pred_pca = qda_pca.fit(data_pca[:, 0:pos + 1], label_train).predict(data_pca_test[:, 0:pos + 1])
acc = (y_pred_pca == label_test).sum() / len(label_test)
print('acc(test, PCA({}, by training ACC)+QDA):{:.2f}%'.format(pos + 1, acc * 100))
print('implement time (PCA+QDA) :{:.5f}s'.format(time.time()-st))


from sklearn.neural_network import MLPClassifier
clf_mlp = MLPClassifier(solver='lbfgs',
                        activation='logistic',
                        alpha=1e-5,
                        hidden_layer_sizes=(100,),
                        batch_size=1000,
                        max_iter=1000,
                        random_state=1)
'''
structure: 784 -> 2000 -> 10,  
Parameter: 784*2000 + 2000*10 = 1,588,000
acc(test, MLP):97.91%
implement time (MLP) :432.37997s
'''
'''
structure: 784 -> 100 -> 100 -> 10,  
Parameter: 784*100 + 100*100 + 100*10 = 89,400
acc(test, MLP):95.12%
implement time (MLP) :1277.10160s
'''
st = time.time()
clf_mlp.fit(data_train, label_train)
y_pred = clf_mlp.predict(data_test)
cm = confusion_matrix(y_pred, label_test)
acc = np.diag(cm).sum()/cm.sum()
print('acc(test, MLP):{:.2f}%'.format(acc * 100))
print('implement time (MLP) :{:.5f}s'.format(time.time()-st))

