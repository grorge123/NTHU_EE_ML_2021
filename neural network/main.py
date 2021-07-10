#!/usr/bin/env python
# coding: utf-8

# In[1]:

##導入library和設定random seed
import numpy as np
import math
from sklearn.decomposition import PCA
from PIL import Image
import os
import random

seed = 1227
random.seed = seed
np.random.seed = seed


# In[2]:

##讀取圖片
tr_path = "Data/Data_train/"
tt_path = "Data/Data_test/"
tr_data = [[], [], []]
tt_data = [[], [], []]

for name in os.listdir(tr_path + "Carambula"):
    image = Image.open(tr_path + "Carambula/" + name)
    tr_data[0].append(np.array(image).reshape(32*32*2))
for name in os.listdir(tr_path + "Lychee"):
    image = Image.open(tr_path + "Lychee/" + name)
    tr_data[1].append(np.array(image).reshape(32*32*2))
for name in os.listdir(tr_path + "Pear"):
    image = Image.open(tr_path + "Pear/" + name)
    tr_data[2].append(np.array(image).reshape(32*32*2))
    
for name in os.listdir(tt_path + "Carambula"):
    image = Image.open(tt_path + "Carambula/" + name)
    tt_data[0].append(np.array(image).reshape(32*32*2))
for name in os.listdir(tt_path + "Lychee"):
    image = Image.open(tt_path + "Lychee/" + name)
    tt_data[1].append(np.array(image).reshape(32*32*2))
for name in os.listdir(tt_path + "Pear"):
    image = Image.open(tt_path + "Pear/" + name)
    tt_data[2].append(np.array(image).reshape(32*32*2))
    
for i in range(3):
    tr_data[i] = np.array(tr_data[i])
    tt_data[i] = np.array(tt_data[i])

for i in range(3):
    pca = PCA(n_components=2)
    pca.fit(tr_data[i])
    tr_data[i] = pca.transform(tr_data[i])
    tt_data[i] = pca.transform(tt_data[i])
    

tr_in = []
tt_in = []

for i in range(3):
    for q in range(len(tr_data[i])):
        label = np.zeros((3))
        label[i] = 1
        tr_in.append([label, np.append(tr_data[i][q], np.array([1]))])
    for q in range(len(tt_data[i])):
        label = np.zeros((3))
        label[i] = 1
        tt_in.append([label, np.append(tt_data[i][q], np.array([1]))])
        
random.shuffle(tr_in)


# In[3]:


##標準化
te = np.array(tr_in)
for i in range(2):
    me = np.mean(te[:,1,i])
    std = np.std(te[:,1,i])
    for q in range(len(tr_in)):
        tr_in[q][1][i] -= me
        tr_in[q][1][i] /= std
    for q in range(len(tt_in)):
        tt_in[q][1][i] -= me
        tt_in[q][1][i] /= std


# In[4]:

##繪圖
# import matplotlib.pyplot as plt 
# plt.suptitle("train data")
# for i in tr_data[0]:
#     plt.plot(i[0], i[1], 'o', color='black');
# for i in tr_data[1]:
#     plt.plot(i[0], i[1], 'o', color='red');
# for i in tr_data[2]:
#     plt.plot(i[0], i[1], 'o', color='blue');


# In[5]:


# import matplotlib.pyplot as plt 
# plt.suptitle("test data")
# for i in tt_data[0]:
#     plt.plot(i[0], i[1], 'o', color='black');
# for i in tr_data[1]:
#     plt.plot(i[0], i[1], 'o', color='red');
# for i in tr_data[2]:
#     plt.plot(i[0], i[1], 'o', color='blue');


# In[6]:


# plt.suptitle("train data normalized")
# for i in tr_in:
#     if list(i[0]).index(1) == 0:
#         plt.plot(i[1][0], i[1][1], 'o', color='black');
#     elif list(i[0]).index(1) == 1:
#         plt.plot(i[1][0], i[1][1], 'o', color='red');
#     elif list(i[0]).index(1) == 2:
#         plt.plot(i[1][0], i[1][1], 'o', color='blue');


# In[7]:


# plt.suptitle("test data normalized")
# for i in tt_in:
#     if list(i[0]).index(1) == 0:
#         plt.plot(i[1][0], i[1][1], 'o', color='black');
#     elif list(i[0]).index(1) == 1:
#         plt.plot(i[1][0], i[1][1], 'o', color='red');
#     elif list(i[0]).index(1) == 2:
#         plt.plot(i[1][0], i[1][1], 'o', color='blue');


# In[8]:

##設定函數
def softmax(x):
    exp = np.exp(x-x.max())
    return exp/exp.sum()
#     return np.exp(x) / np.sum(np.exp(x), axis=0)
def liner(x):
    return x;
def init(k, m):
#     return np.random.random((k, m))
    return np.random.uniform(-math.sqrt(6/k+m), math.sqrt(6/k+m), (k, m))
def cross(pred, y):
    loss = 0
    for i in range(len(pred)):
            loss -= y[i] * math.log2(pred[i])
    return loss
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def d_sigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))


# In[9]:

##建立three_layer模型
class three_layer:
    def __init__(self, inpdim, lr):
        self.lr = lr
        self.inpdim = inpdim
        self.hid2 = 32
        self.hid3 = 32
        self.dim = [inpdim, self.hid2, self.hid3, 3]
        self.act = [sigmoid, sigmoid, softmax]
        self.w = np.array([init(inpdim, self.hid2), init(self.hid2, self.hid3), init(self.hid3, 3)])
    def forward(self, data):
        z1 = data.dot(self.w[0])
        self.a1 = sigmoid(z1)
        z2 = self.a1.dot(self.w[1])
        self.a2 = sigmoid(z2)
        z3 = self.a2.dot(self.w[2])
        out = softmax(z3)
        return out
    def backward(self, data, pred, y):
        dw = np.array([np.zeros((self.inpdim, self.hid2)), np.zeros((self.hid2, self.hid3)), np.zeros((self.hid3, 3))])
        for i in range(3):
            dl = pred[i] - y[i]
            for q in range(self.hid3):
                dw[2][q][i] += dl * self.a2[q]
            for q in range(self.hid3):
                for k in range(self.hid2):
                    dw[1][k][q] += dl * self.w[2][q][i] * self.a2[q] * (1 - self.a2[q]) * self.a1[k]
            for q in range(self.inpdim):
                for k in range(self.hid2):
                    tmp = 0.0
                    for j in range(self.hid3):
                        tmp += self.w[2][j][i] * self.a2[j] * (1 - self.a2[j]) * self.w[1][k][j]
                    dw[0][q][k] += dl * tmp * self.a1[k] * (1 - self.a1[k]) * data[q]
        return dw
    def train(self, in_data, mode="train"):
        if mode == "train":
            loss = 0
            dw = np.array([np.zeros((self.inpdim, self.hid2)), np.zeros((self.hid2, self.hid3)), np.zeros((self.hid3, 3))])
            for i in in_data:
                pred = self.forward(i[1])
                loss += cross(pred, i[0])
                dw += self.backward(i[1], pred, i[0])
            self.w -= self.lr * dw
            return loss
        else:
            re = []
            for i in in_data:
                re.append(self.forward(i[1]))
            return re;


# In[10]:

##訓練three_layer模型
epoch = 20
batch_size = 1
lr = 0.01

model = three_layer(tr_in[0][1].shape[0], lr)
loss_record = []
acc_record = []

for _ in range(epoch):
    loss = 0
    acc = 0
    for bti in range(math.ceil(len(tr_in)/batch_size)):
        now_in = tr_in[bti * batch_size : min((bti + 1) * batch_size, len(tr_in))]
        tmploss = model.train(now_in)
        loss += tmploss
        pred = model.train(now_in, mode="test")
        acc += (np.argmax(pred, 1) == np.argmax(np.array([i[0] for i in now_in]), 1)).sum()
    loss_record.append(loss)
    acc_record.append(acc/len(tr_in))
    print("three_layer epoch:", _, "loss:", loss, "Acc:", acc/len(tr_in))
    acc = 0
    pred = model.train(tt_in, mode="test")
    acc += (np.argmax(pred, 1) == np.argmax(np.array([i[0] for i in tt_in]), 1)).sum()
    print("Test data acc:", acc / len(tt_in))


# In[11]:

##繪圖
# import matplotlib.pyplot as plt 
# plt.suptitle("Loss")
# plt.plot(range(len(loss_record)),loss_record,'s-',color = 'r')
# plt.subplots()
# plt.suptitle("Acc")
# plt.plot(range(len(acc_record)),acc_record,'s-',color = 'r')


# In[60]:

##建立two_layer模型
class two_layer:
    def __init__(self, inpdim, lr):
        self.lr = lr
        self.inpdim = inpdim
        self.hid2 = 16
        self.act = [liner, softmax]
        self.w = np.array([init(inpdim, self.hid2), init(self.hid2, 3)])
    def forward(self, data):
        z1 = data.dot(self.w[0])
        self.a1 = sigmoid(z1)
        z2 = self.a1.dot(self.w[1])
        out = softmax(z2)
        return out
    def backward(self, data, pred, y):
        dw = np.array([np.zeros((self.inpdim, self.hid2)), np.zeros((self.hid2, 3))])
        for i in range(3):
            dl = pred[i] - y[i]
            for q in range(self.hid2):
                dw[1][q][i] += dl * self.a1[q]
            for q in range(self.hid2):
                for k in range(self.inpdim):
                    dw[0][k][q] += dl * self.w[1][q][i] * self.a1[q] * (1 - self.a1[q]) * data[k]
        return dw
    def train(self, in_data, mode="train"):
        if mode == "train":
            loss = 0
            dw = np.array([np.zeros((self.inpdim, self.hid2)), np.zeros((self.hid2, 3))])
            for i in in_data:
                pred = self.forward(i[1])
                loss += cross(pred, i[0])
                dw += self.backward(i[1], pred, i[0])
            self.w -= self.lr * dw
            return loss
        else:
            re = []
            for i in in_data:
                re.append(self.forward(i[1]))
            return re;


# In[23]:

##訓練two_layer模型
epoch = 20
batch_size = 1
lr = 0.1

model = two_layer(tr_in[0][1].shape[0], lr)
loss_record = []
acc_record = []

for _ in range(epoch):
    loss = 0
    acc = 0
    for bti in range(math.ceil(len(tr_in)/batch_size)):
        now_in = tr_in[bti * batch_size : min((bti + 1) * batch_size, len(tr_in))]
        tmploss = model.train(now_in)
        loss += tmploss
#         print("Loss:", tmploss)
        pred = model.train(now_in, mode="test")
        acc += (np.argmax(pred, 1) == np.argmax(np.array([i[0] for i in now_in]), 1)).sum()
    print("two_layer epoch:", _, "loss:", loss, "Acc:", acc/len(tr_in))
    loss_record.append(loss)
    acc_record.append(acc/len(tr_in))
    acc = 0
    pred = model.train(tt_in, mode="test")
    acc += (np.argmax(pred, 1) == np.argmax(np.array([i[0] for i in tt_in]), 1)).sum()
    print("Test data acc:", acc / len(tt_in))


# In[18]:

##繪圖
# import matplotlib.pyplot as plt 
# plt.suptitle("Loss")
# plt.plot(range(len(loss_record)),loss_record,'s-',color = 'r')
# plt.subplots()
# plt.suptitle("Acc")
# plt.plot(range(len(acc_record)),acc_record,'s-',color = 'r')
