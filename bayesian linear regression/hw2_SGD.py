'''
NTHU EE Machine Learning HW2
Author: 
Student ID: 109062123
'''
import numpy as np
import pandas as pd
import math
import scipy.stats
import argparse
import random

def initw(sz):
    # re = np.array([])
    # re = np.resize(re, (sz))
    # for i in range(sz):
    #     re[i] = random.uniform(1, 0)
    # return re;
    return np.random.uniform(0, 1, sz)

def initx(x1_max, x2_max, x1_min, x2_min, O1, O2, x):
    re = np.array([])
    re = np.resize(re, (x.shape[0], O1 * O2 + 2))
    s1 = (x1_max - x1_min) / (O1 - 1)
    s2 = (x2_max - x2_min) / (O2 - 1)
    for sz in range(x.shape[0]):
        for i in range(1, O1+1):
            for j in range(1, O2+1):
                ui = s1 * (i - 1) + x1_min
                uj = s2 * (j - 1) + x2_min
                nx = math.exp(-((x[sz][0] - ui) ** 2 / (2 * s1 ** 2)) - ((x[sz][1] - uj) ** 2 / (2 * s2 ** 2)))
                re[sz][O2 * (i - 1) + j - 1] = nx
        re[sz][O1*O2] = x[sz][2]
        re[sz][O1*O2+1] = 1
        
    return re

def SGD(w, x, y):
    re = np.array([])
    re = np.resize(re, w.shape)
    for i in range(x.shape[0]):
        for q in range(x.shape[0]):
            re[i] += w[q] * x[q] * x[i]
        re[i] -= y * x[i]
    return re

def SGD2(w, x, y):
    re = np.array([])
    re = np.resize(re, w.shape)
    for i in range(x.shape[0]):
        for q in range(x.shape[0]):
            re[i] += w[q] * x[q] * x[i]
            re[i] += w[q]
        re[i] -= y * x[i]
    return re

def pred(w, x):
    re = np.array([])
    re = np.resize(re, x.shape[0])
    for q in range(x.shape[0]):
        for i in range(w.shape[0]):
            re[q] += w[i] * x[q][i]
    return re

def CalMSE(data, prediction):

    squared_error = (data - prediction) ** 2
    sum_squared_error = np.sum(squared_error)
    mean__squared_error = sum_squared_error/prediction.shape[0]

    return mean__squared_error

# do not change the name of this function
def BLR(train_data, test_data_feature, O1=2, O2=2, test=False):  # remember to set best choice O1 and O2 as default
    '''
    output: ndarray with size (length of test_data, )
    '''
    data_train_feature = train_data[:, :3]
    data_train_label = train_data[:, 3]
    x1_min = float(min(data_train_feature[0,:]))
    x2_min = float(min(data_train_feature[1,:]))
    x1_max = float(max(data_train_feature[0,:]))
    x2_max = float(max(data_train_feature[1,:]))
    
    x = initx(x1_max, x2_max, x1_min, x2_min, O1, O2, data_train_feature)
    w = initw(x[0].shape[0])
    epoch = 5000
    lr = 1e-3
    data_size = len(data_train_label)
    batch_size = 1
    greatloss = 5e6
    lastupdate = -1

    for _ in range(epoch):
        for i in range(data_size):
            _w = SGD2(w, x[i], data_train_label[i]);
            w -= lr * _w
            last = i
        if _ % 10 == 0:
            pre = pred(w, x)
            loss = CalMSE(pre, data_train_label)
            # print("Loss:", loss)
            if greatloss - loss > 1e-4:
                greatloss = loss
                lastupdate = _
            if greatloss < 0.01 or _ - lastupdate >= 40:
                print("early break")
                break
    
    x = initx(x1_max, x2_max, x1_min, x2_min, O1, O2, test_data_feature)
    re = pred(w, x)
    if test:
        return re, greatloss
    return re 

    return y_BLRprediction 

# do not change the name of this function
def MLR(train_data, test_data_feature, O1=2, O2=2, test=False):  # remember to set best choice O1 and O2 as default
    '''
    output: ndarray with size (length of test_data, )
    '''
    data_train_feature = train_data[:, :3]
    data_train_label = train_data[:, 3]
    x1_min = float(min(data_train_feature[0,:]))
    x2_min = float(min(data_train_feature[1,:]))
    x1_max = float(max(data_train_feature[0,:]))
    x2_max = float(max(data_train_feature[1,:]))
    
    x = initx(x1_max, x2_max, x1_min, x2_min, O1, O2, data_train_feature)
    w = initw(x[0].shape[0])
    epoch = 5000
    lr = 1e-2
    data_size = len(data_train_label)
    batch_size = 1
    greatloss = 5e6
    lastupdate = -1

    for _ in range(epoch):
        for i in range(data_size):
            _w = SGD(w, x[i], data_train_label[i]);
            w -= lr * _w
            last = i
        if _ % 10 == 0:
            pre = pred(w, x)
            loss = CalMSE(pre, data_train_label)
            # print("Loss:", loss)
            if greatloss - loss > 1e-4:
                greatloss = loss
                lastupdate = _
            if greatloss < 0.01 or _ - lastupdate >= 40:
                print("early break")
                break
    
    x = initx(x1_max, x2_max, x1_min, x2_min, O1, O2, test_data_feature)
    re = pred(w, x)
    if test:
        return re, greatloss
    return re 

def testO(data_train, data_test_feature, data_test_label):
    for O_1 in range(2,8,1):
        for O_2 in range(2, 8, 1):
            pre, tloss = MLR(data_train, data_test_feature, O1=O_1, O2=O_2, test=True)
            print("O_1: {} O_2: {} train_loss: {} test_loss: {}".format(O_1, O_2, tloss, CalMSE(pre, data_test_label)))

def main():
    Seed = 8154
    random.seed(Seed)
    np.random.seed(Seed)
    parser = argparse.ArgumentParser()
    parser.add_argument('-O1', '--O_1', type=int, default=2)
    parser.add_argument('-O2', '--O_2', type=int, default=2)
    args = parser.parse_args()
    O_1 = args.O_1
    O_2 = args.O_2
    
    data_train = pd.read_csv('Training_set.csv', header=None).to_numpy()
    data_test = pd.read_csv('Validation_set.csv', header=None).to_numpy()
    data_test_feature = data_test[:, :3]
    data_test_label = data_test[:, 3]

    # testO(data_train, data_test_feature, data_test_label)
    
    predict_BLR = BLR(data_train, data_test_feature, O1=O_1, O2=O_2)
    predict_MLR = MLR(data_train, data_test_feature, O1=O_1, O2=O_2)
    print('MSE of BLR = {e1}, MSE of MLR= {e2}.'.format(e1=CalMSE(predict_BLR, data_test_label), e2=CalMSE(predict_MLR, data_test_label)))


if __name__ == '__main__':
    main()

