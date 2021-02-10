import random, os
import numpy as np
from numpy import genfromtxt
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets
import csv

X = []
y = []
lossGraph = []

with open("Salary_Data.csv") as csv_file:
    csv_reader = csv.reader(csv_file)
    cnt = 0;
    for lines in csv_reader: # Exclude first line, which are the column names
        if cnt==0:
            cnt+=1
            continue
        X.append(lines[0])
        y.append(lines[1])

# Load data
print("Data Loaded...")
X = np.asarray(X, dtype='float32')
y = np.asarray(y, dtype='float32')

# Plot data
plt.scatter(X, y, color='b', label='0')
plt.legend();
plt.show()
print("Data Plotted...")

# Class
class LinearRegression:
    def __init__(self, lr = 0.1, num_iter = 100000):
        self.lr = lr
        self.num_iter = num_iter

    def __hypothesis(self, X):
        return self.theta*X + self.bias


    def __loss(self, h, y):
        diff = y-h
        return np.sum(diff*diff)/diff.size
    
    def fit(self, X, y):
        
        self.theta = np.ones(X.shape[0])
        self.bias = np.ones(1)

        for i in range(self.num_iter):
            h = self.__hypothesis(X)
            n = float(len(y))
            d_theta = (-2/n)*sum(X*(y-h))
            d_bias = (-2/n)*sum(y-h)

            self.theta = self.theta - d_theta*self.lr
            self.bias = self.bias - d_bias*self.lr
            
            loss = self.__loss(h, y)
            lossGraph.append(loss)

            if i%10000 == 0:
                print("loss: ",loss)

    def predict(self, X):
        return self.__hypothesis(X)

model = LinearRegression(lr=0.001, num_iter=90000)
model.fit(X, y)

y_pred = model.predict(X)
plt.scatter(X, y) 
plt.plot([min(X), max(X)], [min(y_pred), max(y_pred)], color='red')  # regression line
# plt.plot(lossGraph)
plt.show()