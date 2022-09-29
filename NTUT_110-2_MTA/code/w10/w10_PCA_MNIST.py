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
# lda = LinearDiscriminantAnalysis(store_covariance=True)
# lda = lda.fit(data_train, label_train)
# y_pred = lda.predict(data_test)
# cm = confusion_matrix(label_test, y_pred)
# # print(cm)
# print('acc(LDA, test):{:.2f}%'.format(100*np.diag(cm).sum()/cm.sum()))
#
# y_pred = lda.predict(data_train)
# cm = confusion_matrix(label_train, y_pred)
# # print(cm)
# print('acc(LDA, train):{:.2f}%'.format(100*np.diag(cm).sum()/cm.sum()))
#
qda = QuadraticDiscriminantAnalysis(store_covariance=True)
qda.fit(data_train, label_train)
y_pred=qda.predict(data_train)
acc = (y_pred==label_train).sum()/len(label_train)
print('acc(train, QDA):{:.2f}%'.format(acc*100))
y_pred = qda.predict(data_test)
acc = (y_pred==label_test).sum()/len(label_test)
print('acc(test, QDA):{:.2f}%'.format(acc*100))



from sklearn.decomposition import PCA

import time
# st = time.time()
# for npc in range(15):
#     pca = PCA(n_components=npc+1)
#     pca.fit(data_train)
#     data_pca = pca.transform(data_train)
#     print(data_pca.shape)
#     qda_pca = QuadraticDiscriminantAnalysis(store_covariance=True)
#     y_pred_pca = qda_pca.fit(data_pca, label_train).predict(data_pca)
#     acc = (y_pred_pca==label_train).sum()/len(label_train)
#     print('acc(train, PCA({})+QDA):{:.2f}%'.format(npc+1, acc*100))
# print('implement time (PCA step by step):{:4f}s'.format(time.time()-st))
#
#
# st = time.time()
pca = PCA(n_components=15)
pca.fit(data_train)
# data_pca = pca.transform(data_train)
# for npc in range(15):
#     qda_pca = QuadraticDiscriminantAnalysis(store_covariance=True)
#     y_pred_pca = qda_pca.fit(data_pca[:,0:npc+1], label_train).predict(data_pca[:,0:npc+1])
#     acc = (y_pred_pca == label_train).sum() / len(label_train)
#     print('acc(train, PCA({})+QDA):{:.2f}%'.format(npc + 1, acc * 100))
# print('implement time (PCA once):{:4f}s'.format(time.time()-st))

print('PCA explained_variance_ratio:{}'.format(pca.explained_variance_ratio_))
count = 0
for npc in range(15):
    print('accumulate explained_variance_ratio(1:{}):{}'.format(npc+1,np.sum(pca.explained_variance_ratio_[0:npc+1])))

# #
# # st = time.time()
pca = PCA(n_components=150)
pca.fit(data_train)
data_pca = pca.transform(data_train)
data_pca_test = pca.transform(data_test)
# #
# ACC, ACC_test=[],[]
# for npc in range(50):
#     qda_pca = QuadraticDiscriminantAnalysis(store_covariance=True)
#     qda_pca.fit(data_pca[:,0:npc+1], label_train)
#     y_pred_pca = qda_pca.predict(data_pca[:, 0:npc + 1])
#     acc = (y_pred_pca == label_train).sum() / len(label_train)
#     ACC.append(acc)
#
#     y_pred_pca = qda_pca.predict(data_pca_test[:, 0:npc + 1])
#     acc = (y_pred_pca == label_test).sum() / len(label_test)
#     ACC_test.append(acc)
# ACC = np.array(ACC)
# ACC_test = np.array(ACC_test)
# pos = np.argmax(ACC)
# #
# # #
# qda_pca = QuadraticDiscriminantAnalysis(store_covariance=True)
# y_pred_pca = qda_pca.fit(data_pca[:, 0:pos + 1], label_train).predict(data_pca[:, 0:pos + 1])
# acc = (y_pred_pca == label_train).sum() / len(label_train)
# print('acc(train, PCA({}, by training ACC)+QDA):{:.2f}%'.format(pos + 1, acc * 100))
# y_pred_pca = qda_pca.fit(data_pca[:, 0:pos + 1], label_train).predict(data_pca_test[:, 0:pos + 1])
# acc = (y_pred_pca == label_test).sum() / len(label_test)
# print('acc(test, PCA({}, by training ACC)+QDA):{:.2f}%'.format(pos + 1, acc * 100))
# # #
# # #
# accum_variance=[]
# for npc in range(150):
#     accum_variance.append(np.sum(pca.explained_variance_ratio_[0:npc + 1]))
#
# plt.subplot(3,1,1)
# plt.plot(ACC,'b-*')
# plt.plot(pos, ACC[pos],'or')
# plt.title('ACC: train(blue), test(red)')
# plt.subplot(3,1,2)
# plt.plot(ACC_test,'r-*')
# plt.plot(pos, ACC_test[pos],'or')
# plt.subplot(3,1,3)
# plt.plot(accum_variance)
# plt.title('explained_variance_ratio')
# plt.show()




colors=[]
for i in [0,125,255]:
    for j in [0, 125, 255]:
        for k in [0, 125, 255]:
            colors.append([i/255,j/255,k/255])

ax = plt.figure()
for i in range(10):
    pos = np.where(label_train==i)[0]
    plt.plot(data_pca[pos,0],data_pca[pos,1], '*', color = colors[i])

plt.title('PCA')
plt.show()
