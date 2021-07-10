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
import pandas as pd # 引用套件並縮寫為 pd
import csv
import math
import random
import numpy as np
from sklearn.metrics import precision_recall_fscore_support as f1_score

# def f1_score(pred, label,average):
#     tp = 0.0
#     fn = 0.0
#     fp = 0.0
#     for i in range(len(pred)):
#         if pred[i] == 1 and label[i] == 1:
#             tp += 1
#         elif pred[i] == 0 and label[i] == 1:
#             fn += 1
#         elif pred[i] == 1 and label[i] == 0:
#             fp += 1
#         elif pred[i] == 0 and label[i] == 0:
#             pass
#         else:
#             print("FAIL")
#     pre = tp/(tp+fp)
#     rec = tp/(tp+fn)
#     return (1+1.5**2)*pre*rec/((1.5**2*pre)+rec)

tt_path = "deal_test.csv"
tr_path = "deal_train.csv"
sea_path = "season.csv"

df = pd.read_csv(tt_path, encoding="Big5")
df2 = pd.read_csv(tr_path, encoding="Big5")
data = {};
for i in range(len(df["PerStatus"])):
    if(df.iloc[i]["PerNo"] not in data):
        data[df.iloc[i]["PerNo"]] = {df.iloc[i]["yyyy"]:df.iloc[i][3:]}
    else:
        data[df.iloc[i]["PerNo"]][df.iloc[i]["yyyy"]] = df.iloc[i][3:]

for i in range(len(df2["PerStatus"])):
    if(df2.iloc[i]["PerNo"] not in data):
        data[df2.iloc[i]["PerNo"]] = {df2.iloc[i]["yyyy"]:df2.iloc[i][3:]}
    else:
        data[df2.iloc[i]["PerNo"]][df2.iloc[i]["yyyy"]] = df2.iloc[i][3:]

sea = pd.read_csv(sea_path, encoding="utf-8")
for i in range(len(sea["PerNo"])):
    if(sea.iloc[i]["PerNo"] not in data):
        data[sea.iloc[i]["PerNo"]] = {sea.iloc[i]["periodQ"]:sea.iloc[i][3:]}
    else:
        data[sea.iloc[i]["PerNo"]][sea.iloc[i]["periodQ"]] = sea.iloc[i][3:]

dtr = []
dtt = []
for idx, value in data.items():
    for yy in range(2014, 2016):
        if yy in value and yy+1 in value:
            now = [idx]
            now.extend(value[yy+1])
            now.extend(value[yy])
            now.extend(value[str(yy+1)+"Q1"])
            now.extend(value[str(yy+1)+"Q2"])
            now.extend(value[str(yy+1)+"Q3"])
            now.extend(value[str(yy+1)+"Q4"])
            now.extend(value[str(yy)+"Q1"])
            now.extend(value[str(yy)+"Q2"])
            now.extend(value[str(yy)+"Q3"])
            now.extend(value[str(yy)+"Q4"])
            dtr.append(now)
for idx, value in data.items():
    if 2018 in value:
        now = [idx]
        now.extend(value[2018])
        now.extend((value[2017] if(2017 in value) else [-1 for _ in range(43)]))
        now.extend(value[str(2018)+"Q1"] if(str(2018)+"Q1" in value) else [-1 for _ in range(5)])
        now.extend(value[str(2018)+"Q2"] if(str(2018)+"Q1" in value) else [-1 for _ in range(5)])
        now.extend(value[str(2018)+"Q3"] if(str(2018)+"Q1" in value) else [-1 for _ in range(5)])
        now.extend(value[str(2018)+"Q4"] if(str(2018)+"Q1" in value) else [-1 for _ in range(5)])
        now.extend(value[str(2017)+"Q1"] if(str(2017)+"Q1" in value) else [-1 for _ in range(5)])
        now.extend(value[str(2017)+"Q2"] if(str(2017)+"Q1" in value) else [-1 for _ in range(5)])
        now.extend(value[str(2017)+"Q3"] if(str(2017)+"Q1" in value) else [-1 for _ in range(5)])
        now.extend(value[str(2017)+"Q4"] if(str(2017)+"Q1" in value) else [-1 for _ in range(5)])
        dtt.append(now)

name = ["PerNo","PerStatus","sex","工作分類","職等","廠區代碼","管理層級","工作資歷1","工作資歷2","工作資歷3","工作資歷4","工作資歷5","專案時數","專案總數","當前專案角色","特殊專案佔比","工作地點","訓練時數A","訓練時數B","訓練時數C","生產總額","榮譽數","是否升遷","升遷速度","近三月請假數A","近一年請假數A","近三月請假數B","近一年請假數B","出差數A","出差數B","出差集中度","年度績效等級A","年度績效等級B","年度績效等級C","年齡層級","婚姻狀況","年資層級A","年資層級B","年資層級C","任職前工作平均年數","畢業科系類別","眷屬量","通勤成本","歸屬部門",
        "PerStatus","sex","工作分類","職等","廠區代碼","管理層級","工作資歷1","工作資歷2","工作資歷3","工作資歷4","工作資歷5","專案時數","專案總數","當前專案角色","特殊專案佔比","工作地點","訓練時數A","訓練時數B","訓練時數C","生產總額","榮譽數","是否升遷","升遷速度","近三月請假數A","近一年請假數A","近三月請假數B","近一年請假數B","出差數A","出差數B","出差集中度","年度績效等級A","年度績效等級B","年度績效等級C","年齡層級","婚姻狀況","年資層級A","年資層級B","年資層級C","任職前工作平均年數","畢業科系類別","眷屬量","通勤成本","歸屬部門",
       "加班數","出差數A","出差數B","請假數A","請假數B","加班數","出差數A","出差數B","請假數A","請假數B","加班數","出差數A","出差數B","請假數A","請假數B","加班數","出差數A","出差數B","請假數A","請假數B",
        "加班數","出差數A","出差數B","請假數A","請假數B","加班數","出差數A","出差數B","請假數A","請假數B","加班數","出差數A","出差數B","請假數A","請假數B","加班數","出差數A","出差數B","請假數A","請假數B",
       ]
with open("deal2_train.csv", 'w', newline='', encoding="Big5") as Csv:
    wcsv = csv.writer(Csv)
    wcsv.writerow(name)
    for i in dtr:
        wcsv.writerow(i)
with open("deal2_test.csv", 'w', newline='', encoding="Big5") as Csv:
    wcsv = csv.writer(Csv)
    wcsv.writerow(name)
    for i in dtt:
        wcsv.writerow(i)
max_f1 = 0
while True:
    myseed = random.randint(0,100000),
    # myseed =5445
    random.seed(myseed)
    np.random.seed(myseed)
    random.shuffle(dtr)


    k = 0
    new_dtr = []
    for i in dtr:
        if i[1] == 1 or k < math.ceil(304 * 0.3):
            new_dtr.append(i)
            if i[1] == 0:
                k += 1

    with open("deal3_train.csv", 'w', newline='', encoding="Big5") as Csv:
        wcsv = csv.writer(Csv)
        wcsv.writerow(name)
        for i in new_dtr:
            wcsv.writerow(i)

    f1 = 0

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


    feature_filter = []
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
    # print(f1_score(y_va, y_pred, average='binary'))
    f1 = f1_score(y_va, y_pred, average='binary', beta=1.5)[2]
    # pred = XGBClassifier()
    # pred.fit(np.array(x_tr), np.array(y_tr))





    if f1 > max_f1:
        max_f1 = f1
        print("Successful:", f1, "seed:", myseed)
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