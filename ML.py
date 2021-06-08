import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegressionCV,LogisticRegression
from sklearn.metrics import classification_report,roc_curve,auc
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier as dtc
from  sklearn.ensemble import RandomForestClassifier
import csv
from sklearn.metrics import f1_score
import random


myseed = 7122
random.seed(myseed)
np.random.seed(myseed)

tr_path = 'deal3_train.csv'  # path to training data
tt_path = 'deal2_test.csv'   # path to te|sting data

x_va = []
y_va = []
with open(tr_path, 'r', encoding="Big5") as fp:
            data = list(csv.reader(fp))
            data = np.array([list(map(float,i)) for i in data[1:]]).astype(float)
            random.shuffle(data)
            x_tr = [data[i][2:] for i in range(len(data)) if i % 10 <= 8]
            y_tr = [data[i][1] for i in range(len(data)) if i % 10 <= 8]
            x_va = [data[i][2:] for i in range(len(data)) if i % 10 > 8]
            y_va = [data[i][1] for i in range(len(data)) if i % 10 > 8]
feature_filter = [0,43,37,80,38,81]
feature = []
for i in range(len(x_tr[0])):
    if(i not in feature_filter):
        feature.append(i)
x_tr = list(np.array(x_tr)[:, feature])
x_va = list(np.array(x_va)[:, feature])
print("TEST:", len(x_tr), len(x_tr[0]))
model = [LogisticRegression(), GaussianNB(), RandomForestClassifier(), dtc()]
for i in model:
    mo = i
    mo.fit(x_tr, y_tr)
    print(i, mo.score(x_tr, y_tr))
    y_pred = mo.predict(x_tr)
    print(f1_score(y_tr, y_pred, average='binary'))
    y_pred = mo.predict(x_va)
    print(i, mo.score(x_va, y_va))
    print(f1_score(y_va, y_pred, average='binary'))
pred = RandomForestClassifier()
pred.fit(x_tr, y_tr)
with open(tt_path, 'r', encoding="Big5") as fp:
            data = list(csv.reader(fp))
            data = np.array([list(map(float,i)) for i in data[1:]]).astype(float)
            x_t = data[:, 2:]
            label = data[:, 0]
x_t = list(np.array(x_t)[:, feature])
predict_final = pred.predict(x_t)
print(list(predict_final).count(1))

with open("pred.csv", 'w', newline='') as fp:
    writer = csv.writer(fp)
    writer.writerow(['PerNo', 'PerStatus'])
    for i in range(len(label)):
        writer.writerow([label[i],predict_final[i]])