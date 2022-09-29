# -*- coding: utf-8 -*-
"""
@author: tommy huang
"""
import numpy as np
np.random.seed(202201)

class Gaussian_classifier():
    def ___init__(self):
        self.mu=np.array([])
        self.cov=np.array([])
    def fit(self, data_train, label_train):
        mu, cov=[],[]
        for i in range(np.max(label_train)+1):
            pos = np.where(label_train==i)[0]
            tmp_data = data_train[pos,:]
            tmp_cov = np.cov(np.transpose(tmp_data))
            tmp_mu = np.mean(tmp_data,axis=0)
            mu.append(tmp_mu)
            cov.append(tmp_cov)
        self.mu = np.array(mu)
        self.cov = np.array(cov)
    def predict(self,x_test):
        d_value =[]
        for tmp_mu, tmp_cov in zip(self.mu, self.cov):
            d = len(tmp_mu)
            zero_center_data = x_test - tmp_mu
            tmp = np.dot(zero_center_data.transpose(), np.linalg.inv(tmp_cov))
            tmp = -0.5*np.dot(tmp, zero_center_data)
            tmp1=(2 * np.pi)**(-d/2) * np.linalg.det(tmp_cov)**(-0.5)
            tmp = tmp1 * np.exp(tmp)
            d_value.append(tmp)
        d_value = np.array(d_value)
        return np.argmax(d_value), d_value



filename='data_class.txt'
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

x=np.array(data)
y=np.array(label)
gc = Gaussian_classifier()
gc.fit(x, y)
result, value = gc.predict([158.5, 50])
print(result)
print(value)



