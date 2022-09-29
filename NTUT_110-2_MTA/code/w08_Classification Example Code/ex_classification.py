# -*- coding: utf-8 -*-
"""
@author: tommy huang
"""
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(202201)
x = np.load('iris_x.npy')
y = np.load('iris_y.npy')

###  將資料分成訓練集和測試集
## class 0
pos_lis_0 = np.array(range(0, 50))
print(pos_lis_0)
np.random.shuffle(pos_lis_0)
print(pos_lis_0)
# class 1 and class 2
pos_lis_1 = np.array(range(50, 100))
pos_lis_2 = np.array(range(100, 150))
np.random.shuffle(pos_lis_1)
np.random.shuffle(pos_lis_2)

x0 = x[pos_lis_0[0:40],:]
x1 = x[pos_lis_1[0:40],:]
x2 = x[pos_lis_2[0:40],:]
y0 = y[pos_lis_0[0:40]]
y1 = y[pos_lis_1[0:40]]
y2 = y[pos_lis_2[0:40]]
x_train = np.concatenate((x0,x1,x2))
y_train = np.concatenate((y0,y1,y2))

x0 = x[pos_lis_0[40:50],:]
x1 = x[pos_lis_1[40:50],:]
x2 = x[pos_lis_2[40:50],:]
y0 = y[pos_lis_0[40:50]]
y1 = y[pos_lis_1[40:50]]
y2 = y[pos_lis_2[40:50]]
x_test = np.concatenate((x0,x1,x2))
y_test = np.concatenate((y0,y1,y2))

# # sklearn module
# from sklearn.model_selection import train_test_split
# X_train_sk, X_test_sk, y_train_sk, y_test_sk = train_test_split(x, y, test_size=0.2, random_state=20220409)
# x_train = X_train_sk
# y_train = y_train_sk
# x_test = X_test_sk
# y_test = y_test_sk
# print(y_test)

# print(x_train.shape)
# from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
# from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import confusion_matrix
# # Quadratic Discriminant Analysis
# lda = LinearDiscriminantAnalysis(store_covariance=True)
# lda = lda.fit(x_train, y_train)
# y_pred = lda.predict(x_test)
# cm = confusion_matrix(y_test, y_pred)
# acc = np.diag(cm).sum()/cm.sum()
# print('confusion_matrix (LDA):\n{}'.format(cm))
# print('confusion_matrix (LDA,acc):{}'.format(acc))
#
# # Quadratic Discriminant Analysis
# qda = QuadraticDiscriminantAnalysis(store_covariance=True)
# qda = qda.fit(x_train, y_train)
# y_pred = qda.predict(x_test)
# cm = confusion_matrix(y_test, y_pred)
# acc = np.diag(cm).sum()/cm.sum()
# print('confusion_matrix (QDA):\n{}'.format(cm))
# print('confusion_matrix (QDA,acc):{}'.format(acc))

from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.linear_model import LogisticRegression, LinearRegression
# Fit the classifier
# clf = LogisticRegression()
# clf.fit(x_train, y_train)
# y_pred = clf.predict(x_test)
# y_pred_prob= clf.predict_proba(x_test)
# cm = confusion_matrix(y_test, y_pred)
# acc = np.diag(cm).sum()/cm.sum()
# print('confusion_matrix (LogisticRegression):\n{}'.format(cm))
# print('confusion_matrix (LogisticRegression,acc):{}'.format(acc))
# fig = plt.figure()
# ax = fig.gca(projection='3d')
# pos_0 = np.where(y_test==0)[0]
# ax.scatter(y_pred_prob[pos_0,0],y_pred_prob[pos_0,1],y_pred_prob[pos_0,2],'r', marker='^', label='class 0')
# pos_1 = np.where(y_test==1)[0]
# ax.scatter(y_pred_prob[pos_1,0],y_pred_prob[pos_1,1],y_pred_prob[pos_1,2],'b', marker='o', label='class 1')
# pos_2 = np.where(y_test==2)[0]
# ax.scatter(y_pred_prob[pos_2,0],y_pred_prob[pos_2,1],y_pred_prob[pos_2,2],'y', marker='*', label='class 2')
# ax.set_xlabel('prob. for class 0')
# ax.set_ylabel('prob. for class 1')
# ax.set_zlabel('prob. for class 2')
# ax.legend()
# plt.show()

lr = LinearRegression()
lr.fit(x_train, y_train)
y_pred = lr.predict(x_test)
mse = np.mean((y_pred-y_test)**2)
print(y_pred)
plt.figure()
for i in range(3):
    pos = np.where(y_test==i)[0]
    plt.plot(y_pred[pos],y_test[pos],'*')
plt.plot([0.5,0.5],[0, 2],'k:')
plt.plot([1.5,1.5],[0, 2],'k:')
plt.title('MSE:{:.6f}'.format(mse))
plt.show()