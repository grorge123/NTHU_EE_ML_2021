#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import random
import math
from scipy.stats import norm
#setting random seed
# Seed = 8415
# random.seed(Seed)
# np.random.seed(Seed)

#read csv
df = pd.read_csv('Wine.csv')
data = []
for i in df.iloc:
    data.append(list(i))
#random training and testing data
random.shuffle(data)
test = []
test_label = []
train = []
train_label = []
#slice training data and testing data
slice_num = [0, 0, 0]
for i in data:
    if slice_num[int(i[0])-1] < 18:
        test.append(i[1:])
        test_label.append(i[0])
        slice_num[int(i[0])-1] += 1
    else:
        train.append(i[1:])
        train_label.append(i[0])
train = np.array(train)
test = np.array(test)
print("traing data shape:",train.shape)
print("testing data shape:", test.shape)
#set prior from training data
config = [train_label.count(1)/124, train_label.count(2)/124, train_label.count(3)/124]
#set how many feature be used
feature_num = train.shape[1]
#write csv
with open("train.csv", "w") as f:
    for i in range(len(train)):
        f.write(str(train_label[i]))
        for i in train[i]:
            f.write(',')
            f.write(str(i))
        f.write('\n')
with open("test.csv", "w") as f:
    for i in range(len(test)):
        f.write(str(test_label[i]))
        for i in test[i]:
            f.write(',')
            f.write(str(i))
        f.write('\n')


# In[2]:


#Gaussian distribution
def normal_distribution(mean, sigma):
    return lambda x: np.exp(-1*((x-mean)**2)/(2*(sigma**2)))/(math.sqrt(2*np.pi) * sigma)


# In[3]:


# get likelihood
def likelihood(data, goal, label, no_slice=False):
    train = []
    if no_slice == False:
        for i in range(len(data)):
            if label[i] == goal:
                train.append(data[i])
    else:
        train = data
    mean = []
    scale = []
    #get mean and std to build gaussian distribution
    for i in range(feature_num):
        mean.append(np.mean(np.array([train_slice[i] for train_slice in train])))
        scale.append(np.std(np.array([train_slice[i] for train_slice in train])))
    return [norm(mean[i], scale[i]).pdf for i in range(13)]


# In[4]:


#predict
def prediction(data, label):
    pred = []
    li = [likelihood(data, 1, label), likelihood(data, 2, label), likelihood(data, 3, label)]
    
    for i in data:
        prob_n = []
        for num in range(3):
            prob = 1
            #likelihood * prior
            for q in range(len(i)):
                prob *= li[num][q](i[q])
            prob *= config[num]
            prob_n.append(prob)
        #check which probability is highest
        pred.append(prob_n.index(max(prob_n))+1)
    #check accurate
    acc = 0
    for i in range(len(pred)):
        if pred[i] == label[i]:
            acc += 1
    acc /= len(pred)
    return acc, pred


# In[5]:

acc, label = prediction(train, train_label)
print("train acc:",acc*100,"%")
acc, label = prediction(test, test_label)
print("test acc:",acc*100,"%")


#draw picture
"""
# In[6]:


li = likelihood(train, 1, train_label)
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
for i in range(13):
    plt.suptitle(df.columns[i+1])
    for q in range(len(train_label)):
        plt.scatter(train[q][i], train_label[q], s = 30 ,c="r")
    plt.subplots()


# In[7]:



from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca.fit(train)
X_pca = pca.transform(train)
print("transformed shape:", X_pca.shape)
# plt.plot(X_pca, train_label, 'o', lw=5, alpha=0.6, label='pca')
plt.suptitle("train 2D PCA")
fig = plt.figure(figsize=[10,10])
for i in range(len(X_pca)):
    plt.scatter(X_pca[i][0], X_pca[i][1], s = 30 ,marker=' ox^'[int(train_label[i])], c=" rgb"[int(train_label[i])])


# In[8]:



from sklearn.decomposition import PCA
pca = PCA(n_components=3)
pca.fit(train)
X_pca = pca.transform(train)
print("transformed shape:", X_pca.shape)
# plt.plot(X_pca, train_label, 'o', lw=5, alpha=0.6, label='pca')
fig = plt.figure(figsize=[10,10])
ax = fig.gca(projection='3d')
plt.suptitle("train 3D PCA")
for i in range(len(X_pca)):
    if train_label[i] == 1:
        ax.scatter(X_pca[i][0], X_pca[i][1], X_pca[i][2] , marker='o', c='r')
    elif train_label[i] == 2:
        ax.scatter(X_pca[i][0], X_pca[i][1], X_pca[i][2] , marker='x', c='g')
    elif train_label[i] == 3:
        ax.scatter(X_pca[i][0], X_pca[i][1], X_pca[i][2] , marker='^', c='b')


# In[11]:



from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pca.fit(test)
X_pca = pca.transform(test)
print("transformed shape:", X_pca.shape)
# plt.plot(X_pca, train_label, 'o', lw=5, alpha=0.6, label='pca')
plt.suptitle("test 2D PCA")
fig = plt.figure(figsize=[5,5])
for i in range(len(X_pca)):
    plt.scatter(X_pca[i][0], X_pca[i][1], s = 30 ,marker=' ox^'[int(test_label[i])],c=" rgb"[int(test_label[i])])


# In[10]:



from sklearn.decomposition import PCA
pca = PCA(n_components=3)
pca.fit(test)
X_pca = pca.transform(test)
print("transformed shape:", X_pca.shape)
# plt.plot(X_pca, train_label, 'o', lw=5, alpha=0.6, label='pca')
fig = plt.figure(figsize=[5,5])
ax = fig.gca(projection='3d')
plt.suptitle("test 3D PCA")
for i in range(len(X_pca)):
    if test_label[i] == 1:
        ax.scatter(X_pca[i][0], X_pca[i][1], X_pca[i][2] , marker='o', c='r')
    elif test_label[i] == 2:
        ax.scatter(X_pca[i][0], X_pca[i][1], X_pca[i][2] , marker='x', c='g')
    elif test_label[i] == 3:
        ax.scatter(X_pca[i][0], X_pca[i][1], X_pca[i][2] , marker='^', c='b')

"""
