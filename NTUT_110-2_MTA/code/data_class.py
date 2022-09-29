# -*- coding: utf-8 -*-
"""
Created on Sun Mar 20 20:08:52 2022

@author: glanb
"""
filename='data_class.txt'

# 1. txt reader
# with open(filename,'r') as f:
#     for line in f.readlines():
#         print(line)


# 2. 因為有中文需要解碼
with open(filename,'r', encoding="utf-8") as f:
    for line in f.readlines():
        print(line)


data_heigh=[]
data_weight=[]
label=[]
with open(filename,'r', encoding="utf-8") as f:
    for line in f.readlines():
        line = line.split('\t')
        data_heigh.append(float(line[0]))
        data_weight.append(float(line[1]))
        if '男' in line[2]:
            label.append(1)
        else:
            label.append(0)

# 1. correlation
import numpy as np
import matplotlib.pyplot as plt
data_heigh = np.array(data_heigh)
data_weight = np.array(data_weight)
r = np.corrcoef(data_heigh, data_weight)

# plt.figure()
# plt.plot(data_heigh, data_weight,'*')
# plt.xlabel('Heigth')
# plt.ylabel('Weight')
# plt.title('ALL correlation = {:.2f}'.format(r[0][1]))
# plt.show()

# # 2. correlation (Male and Female)
# label=np.array(label)
# pos_male = np.where(label==1)[0]
# pos_female = np.where(label==0)[0]
# plt.figure()
# plt.subplot(1,2,1)
# r_male = np.corrcoef(data_heigh[pos_male],data_weight[pos_male])
# plt.plot(data_heigh[pos_male], data_weight[pos_male],'*')
# plt.xlabel('Heigth')
# plt.ylabel('Weight')
# plt.title('Male \n correlation = {:.2f}'.format(r_male[0][1]))
# plt.subplot(1,2,2)
# r_female = np.corrcoef(data_heigh[pos_female],data_weight[pos_female])
# plt.plot(data_heigh[pos_female], data_weight[pos_female],'*')
# plt.xlabel('Heigth')
# plt.ylabel('Weight')
# plt.title('Female \n correlation = {:.2f}'.format(r_female[0][1]))
# plt.show()

# # Linear regression
# x = data_heigh
# y = data_weight
# x_bar=np.mean(x)
# y_bar=np.mean(y)
# b1 = np.sum((y-y_bar)*(x-x_bar))/np.sum((x-x_bar)**2)
# b0 = y_bar-b1*x_bar

# x_star = np.linspace(150,190,100)
# y_hat = x_star*b1+b0
# plt.figure()
# plt.plot(x, y,'*')
# plt.plot(x_star, y_hat,'r-')
# plt.xlabel('Heigth')
# plt.ylabel('Weight')
# plt.title('ALL correlation = {:.2f}\n $y={:.4f}x+{:.4f}$'.format(r[0][1],b1,b0))
# plt.show()

# # Linear regression (female)
# x = data_heigh[pos_female]
# y = data_weight[pos_female]
# x_bar=np.mean(x)
# y_bar=np.mean(y)
# b1 = np.sum((y-y_bar)*(x-x_bar))/np.sum((x-x_bar)**2)
# b0 = y_bar-b1*x_bar

# x_star = np.linspace(150,170,100)
# y_hat = x_star*b1+b0
# plt.figure()
# plt.plot(x, y,'*')
# plt.plot(x_star, y_hat,'r-')
# plt.xlabel('Heigth')
# plt.ylabel('Weight')
# plt.title('Female correlation = {:.2f}\n $y={:.4f}x+{:.4f}$'.format(r_female[0][1],b1,b0))

# plt.show()


# # Linear regression (male)
# x = data_heigh[pos_male]
# y = data_weight[pos_male]
# x_bar=np.mean(x)
# y_bar=np.mean(y)
# b1 = np.sum((y-y_bar)*(x-x_bar))/np.sum((x-x_bar)**2)
# b0 = y_bar-b1*x_bar

# x_star = np.linspace(160,190,100)
# y_hat = x_star*b1+b0
# plt.figure()
# plt.plot(x, y,'*')
# plt.plot(x_star, y_hat,'r-')
# plt.xlabel('Heigth')
# plt.ylabel('Weight')
# plt.title('Male correlation = {:.2f}\n $y={:.4f}x+{:.4f}$'.format(r_male[0][1],b1,b0))

# plt.show()


from sklearn.linear_model import LinearRegression
x = np.expand_dims(data_heigh,1)
y = np.expand_dims(data_weight,1)
LR = LinearRegression()
LR.fit(x,y)
print('coefficient by sklearn')
print(LR.coef_)
print(LR.intercept_)


x_bar=np.mean(x)
y_bar=np.mean(y)
b1 = np.sum((y-y_bar)*(x-x_bar))/np.sum((x-x_bar)**2)
b0 = y_bar-b1*x_bar
print('coefficient by close form')
print(b1)
print(b0)
