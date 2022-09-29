# -*- coding: utf-8 -*-
"""
@author: tommy huang
"""
import numpy as np
def classifier_mean(x_male, x_female, x_test):
    d_male = x_test-np.mean(x_male)
    d_female = x_test-np.mean(x_female)
    if d_female < d_male:
        return 0, d_female
    else:
        return 1, d_male
data_female = np.array([22, 25, 30, 33, 35])
data_male = np.array([10, 15, 20, 25, 30])
result = classifier_mean(data_male, data_female, 100)
print(result)


class mean_classifier():
    def ___init__(self):
        self.th=[]
    def fit(self, data_train, label_train):
        mu=[]
        for i in range(np.max(label_train)+1):
            pos = np.where(label_train==i)[0]
            mu.append(np.mean(data_train[pos]))
        self.th = np.array(mu)
    def predict(self,x_test):
        d_value =[]
        for mu in self.th:
            d_value.append(x_test-mu)
        d_value = np.array(d_value)
        return np.argmin(d_value), np.min(d_value)

data_female = np.array([22, 25, 30, 33, 35])
label_female = np.array([0,0,0,0,0])
data_male = np.array([10, 15, 20, 25, 30])
label_male = np.array([1,1,1,1,1])

x = np.concatenate((data_female,data_male))
y = np.concatenate((label_female,label_male))

mc = mean_classifier()
mc.fit(x, y)
result, value = mc.predict(100)
print(result)
print(value)


