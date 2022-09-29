# -*- coding: utf-8 -*-
"""
Created on Sun Mar 20 20:56:11 2022

@author: glanb
"""
import numpy as np
import matplotlib.pyplot as plt
x = np.linspace(-20,10,100)
y=x**2+10*x
plt.figure()
plt.plot(x,y)
plt.title('$y=x^2+10x$')
plt.show()


plt.figure()
plt.plot(x,y)
plt.plot(-5,(-5)**2-50,'r*',markersize=15)
plt.title('$y=x^2+10x$')
plt.show()


plt.figure()
plt.plot(x,y)
plt.plot([5,5],[0,200],'k')
plt.plot(-5,(-5)**2-50,'r*',markersize=15)
plt.plot(5,(-5)**2+50,'r*',markersize=15)
plt.title('$y=x^2+10x$')
plt.show()


