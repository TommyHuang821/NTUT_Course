# -*- coding: utf-8 -*-
"""
@author: tommy huang
"""
import numpy as np
np.random.seed(202201)

class Gaussian_classifier():
    def ___init__(self):
        self.mu=np.array([])
        self.sigma=np.array([])
    def fit(self, data_train, label_train):
        mu, sigma=[],[]
        for i in range(np.max(label_train)+1):
            pos = np.where(label_train==i)[0]
            mu.append(np.mean(data_train[pos]))
            sigma.append(np.std(data_train[pos]))
        self.mu = np.array(mu)
        self.sigma = np.array(sigma)

    def predict(self,x_test):
        d_value =[]
        for tmp_mu, tmp_std in zip(self.mu, self.sigma):
            tmp = (x_test - tmp_mu)**2 / (2*(tmp_std**2))
            tmp = 1/(np.sqrt(2*np.pi)*tmp_std) * np.exp(-tmp)
            d_value.append(tmp)
        d_value = np.array(d_value)
        return np.argmax(d_value), d_value

data_female = np.array([22, 25, 30, 33, 35])
label_female = np.array([0,0,0,0,0])
data_male = np.array([10, 15, 20, 25, 30])
label_male = np.array([1,1,1,1,1])
x = np.concatenate((data_female,data_male))
y = np.concatenate((label_female,label_male))
mc = Gaussian_classifier()
mc.fit(x, y)
result, value = mc.predict(15)
print(result)
print(value)



