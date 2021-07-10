import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegressionCV,LogisticRegression
from sklearn.metrics import classification_report,roc_curve,auc
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier as dtc
from sklearn.tree import ExtraTreeClassifier as etc
from  sklearn.ensemble import RandomForestClassifier, VotingClassifier
import csv
# from sklearn.metrics import f1_score
import random
from xgboost import XGBClassifier
from sklearn.metrics import precision_recall_fscore_support as f1_score
from sklearn import svm
from sklearn.cluster import DBSCAN
times = 20
f1 = 0

for _ in range(times):
    myseed = random.randint(0,100000)
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
                x_tr = [data[i][2:] for i in range(len(data)) if i % 10 <= 6]
                y_tr = [data[i][1] for i in range(len(data)) if i % 10 <= 6]
                x_va = [data[i][2:] for i in range(len(data)) if i % 10 > 6]
                y_va = [data[i][1] for i in range(len(data)) if i % 10 > 6]

    normal = [];
    for i in normal:
        mean = np.mean(x_tr[i])
        std = np.std(x_tr[i])
        x_tr[i] =(x_tr[i] - mean ) / std
        x_va[i] =(x_va[i] - mean ) / std


    feature_filter = [0,37,38,42,43,80,81]
    # feature_filter.extend(range(0,85))
    # feature_filter.extend(range(85,len(x_tr[0])))
    # feature_filter = [41,84,18,61,53,10,13,56,93,98,100,103,105,110,113,115,119,123,60,90,95,108,120,28,46,85,88]
    feature = []
    for i in range(85):
        if(i not in feature_filter and i not in feature):
            feature.append(i)
            # if(i < 85 and i + 43 < 85):
            #     feature.append(i+43)
            # elif i >= 85:
            #     for q in range(1,8):
            #         if(i + q*5 < len(x_tr[0])):
            #             feature.append(i+q*5)
    _x_tr = x_tr
    _x_va = x_va
    # for q in range(0,42):
    #     feature = []
    #     feature.append(q)
    #     feature.append(q+43)
    x_tr = list(np.array(_x_tr)[:, feature])
    x_va = list(np.array(_x_va)[:, feature])

    # model = [LogisticRegression(), GaussianNB(), RandomForestClassifier(), dtc()]
    # for i in model:
    mo1 = GaussianNB()
    mo2 = dtc(random_state=50,splitter="best",max_depth=15,min_samples_leaf=5)
    mo3 = etc(random_state=50,splitter="best",max_depth=15,min_samples_leaf=5)
    mo4 = RandomForestClassifier(n_estimators = 80, oob_score = True, n_jobs = -1,random_state =7122,max_features = None, max_depth=15)
    mo5 = XGBClassifier(eta=0.2, min_child_weight=0, reg_lambda=0, objective="multi:softmax", num_class=2,use_label_encoder=False,eval_metric="rmsle")
    # mo = svm.SVC(C=0.8, kernel='rbf', gamma=10, decision_function_shape='ovr')
    mo = VotingClassifier(estimators=[
         ('GaussianNB',mo1),('dtc',mo2),('etc', mo3), ('rf', mo4), ('xg', mo5)], voting='soft')
    mo.fit(np.array(x_tr), np.array(y_tr))
    # print(i, mo.score(x_tr, y_tr))
    y_pred = mo.predict(np.array(x_tr))
    # print(f1_score(y_tr, y_pred, average='binary'))
    y_pred = mo.predict(np.array(x_va))
    # print(i, mo.score(x_va, y_va))
    print(f1_score(y_va, y_pred, average='binary', beta=1.5)[2])
    f1 += f1_score(y_va, y_pred, average='binary', beta=1.5)[2]
    # pred = XGBClassifier()
    # pred.fit(np.array(x_tr), np.array(y_tr))
print("All scoure", f1/times)
pred = mo;
with open(tt_path, 'r', encoding="Big5") as fp:
            data = list(csv.reader(fp))
            data = np.array([list(map(float,i)) for i in data[1:]]).astype(float)
            x_t = data[:, 2:]
            label = data[:, 0]
x_t = list(np.array(x_t)[:, feature])
predict_final = pred.predict(np.array(x_t))
print(list(predict_final).count(1))

with open("pred.csv", 'w', newline='') as fp:
    writer = csv.writer(fp)
    writer.writerow(['PerNo', 'PerStatus'])
    for i in range(len(label)):
        writer.writerow([label[i],predict_final[i]])