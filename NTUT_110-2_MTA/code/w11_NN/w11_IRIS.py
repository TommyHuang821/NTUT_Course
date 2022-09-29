# -*- coding: utf-8 -*-
"""
@author: tommy huang
"""
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(202201)
x = np.load('iris_x.npy')
y = np.load('iris_y.npy')

# sklearn module
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=20220409)



# print(x_train.shape)
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import confusion_matrix
# Quadratic Discriminant Analysis
lda = LinearDiscriminantAnalysis(store_covariance=True)
lda = lda.fit(x_train, y_train)
y_pred = lda.predict(x_test)
cm = confusion_matrix(y_test, y_pred)
acc = np.diag(cm).sum()/cm.sum()
print('confusion_matrix (LDA):\n{}'.format(cm))
print('confusion_matrix (LDA,acc):{}'.format(acc))
qda = QuadraticDiscriminantAnalysis(store_covariance=True)
qda = qda.fit(x_train, y_train)
y_pred = qda.predict(x_test)
cm = confusion_matrix(y_test, y_pred)
acc = np.diag(cm).sum()/cm.sum()
print('confusion_matrix (QDA):\n{}'.format(cm))
print('confusion_matrix (QDA,acc):{}'.format(acc))

from sklearn.neural_network import MLPClassifier
clf_mlp = MLPClassifier(solver='lbfgs',
                        alpha=1e-5,
                        hidden_layer_sizes=(10,),
                        max_iter=100,
                        random_state=1)
clf_mlp.fit(x_train, y_train)
y_pred = clf_mlp.predict(x_test)
cm = confusion_matrix(y_test, y_pred)
acc = np.diag(cm).sum()/cm.sum()
print('confusion_matrix (MLP):\n{}'.format(cm))
print('confusion_matrix (MLP,acc):{}'.format(acc))
