tt_path = "test.csv"
tr_path = "train.csv"
import pandas as pd # 引用套件並縮寫為 pd  
import csv


##將空的label標成-1
df = pd.read_csv(tt_path, encoding="Big5")


for i in range(len(df["PerStatus"])):
    df["PerStatus"][i] = -1

for i in range(len(df["PerStatus"])):
    if(df["sex"][i] != 0 and df["sex"][i] != 1):
        for q in df.columns.values.tolist()[3:]:
            df[q][i] = -1


df.to_csv("deal_test.csv",  encoding="Big5")


df = pd.read_csv(tr_path, encoding="Big5")

for i in range(len(df["PerStatus"])):
    if(df["sex"][i] != 0 and df["sex"][i] != 1):
        for q in df.columns.values.tolist()[3:]:
            df[q][i] = -1

df.to_csv("deal_train.csv",  encoding="Big5")

###########################################
#每兩年數據連起來

tt_path = "deal_test.csv"
tr_path = "deal_train.csv"
import pandas as pd # 引用套件並縮寫為 pd
import csv

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
       
dtr = []
dtt = []
for idx, value in data.items():
    for yy in range(2014, 2016):
        if yy in value and yy+1 in value:
            now = [idx]
            now.extend(value[yy+1])
            now.extend(value[yy])
            dtr.append(now)
for idx, value in data.items():
    if 2018 in value:
        now = [idx]
        now.extend(value[2018])
        now.extend((value[2017] if(2017 in value) else [-1 for _ in range(43)]))
        dtt.append(now)

# name = ["PerNo","PerStatus","sex","工作分類","職等","廠區代碼","管理層級","工作資歷1","工作資歷2","工作資歷3","工作資歷4","工作資歷5","專案時數","專案總數","當前專案角色","特殊專案佔比","工作地點","訓練時數A","訓練時數B","訓練時數C","生產總額","榮譽數","是否升遷","升遷速度","近三月請假數A","近一年請假數A","近三月請假數B","近一年請假數B","出差數A","出差數B","出差集中度","年度績效等級A","年度績效等級B","年度績效等級C","年齡層級","婚姻狀況","年資層級A","年資層級B","年資層級C","任職前工作平均年數","畢業科系類別","眷屬量","通勤成本","歸屬部門",
#         "PerStatus","sex","工作分類","職等","廠區代碼","管理層級","工作資歷1","工作資歷2","工作資歷3","工作資歷4","工作資歷5","專案時數","專案總數","當前專案角色","特殊專案佔比","工作地點","訓練時數A","訓練時數B","訓練時數C","生產總額","榮譽數","是否升遷","升遷速度","近三月請假數A","近一年請假數A","近三月請假數B","近一年請假數B","出差數A","出差數B","出差集中度","年度績效等級A","年度績效等級B","年度績效等級C","年齡層級","婚姻狀況","年資層級A","年資層級B","年資層級C","任職前工作平均年數","畢業科系類別","眷屬量","通勤成本","歸屬部門",
#         "PerStatus","sex","工作分類","職等","廠區代碼","管理層級","工作資歷1","工作資歷2","工作資歷3","工作資歷4","工作資歷5","專案時數","專案總數","當前專案角色","特殊專案佔比","工作地點","訓練時數A","訓練時數B","訓練時數C","生產總額","榮譽數","是否升遷","升遷速度","近三月請假數A","近一年請假數A","近三月請假數B","近一年請假數B","出差數A","出差數B","出差集中度","年度績效等級A","年度績效等級B","年度績效等級C","年齡層級","婚姻狀況","年資層級A","年資層級B","年資層級C","任職前工作平均年數","畢業科系類別","眷屬量","通勤成本","歸屬部門",
#         "PerStatus","sex","工作分類","職等","廠區代碼","管理層級","工作資歷1","工作資歷2","工作資歷3","工作資歷4","工作資歷5","專案時數","專案總數","當前專案角色","特殊專案佔比","工作地點","訓練時數A","訓練時數B","訓練時數C","生產總額","榮譽數","是否升遷","升遷速度","近三月請假數A","近一年請假數A","近三月請假數B","近一年請假數B","出差數A","出差數B","出差集中度","年度績效等級A","年度績效等級B","年度績效等級C","年齡層級","婚姻狀況","年資層級A","年資層級B","年資層級C","任職前工作平均年數","畢業科系類別","眷屬量","通勤成本","歸屬部門"
#        ]
name = ["PerNo","PerStatus","sex","工作分類","職等","廠區代碼","管理層級","工作資歷1","工作資歷2","工作資歷3","工作資歷4","工作資歷5","專案時數","專案總數","當前專案角色","特殊專案佔比","工作地點","訓練時數A","訓練時數B","訓練時數C","生產總額","榮譽數","是否升遷","升遷速度","近三月請假數A","近一年請假數A","近三月請假數B","近一年請假數B","出差數A","出差數B","出差集中度","年度績效等級A","年度績效等級B","年度績效等級C","年齡層級","婚姻狀況","年資層級A","年資層級B","年資層級C","任職前工作平均年數","畢業科系類別","眷屬量","通勤成本","歸屬部門",
        "PerStatus","sex","工作分類","職等","廠區代碼","管理層級","工作資歷1","工作資歷2","工作資歷3","工作資歷4","工作資歷5","專案時數","專案總數","當前專案角色","特殊專案佔比","工作地點","訓練時數A","訓練時數B","訓練時數C","生產總額","榮譽數","是否升遷","升遷速度","近三月請假數A","近一年請假數A","近三月請假數B","近一年請假數B","出差數A","出差數B","出差集中度","年度績效等級A","年度績效等級B","年度績效等級C","年齡層級","婚姻狀況","年資層級A","年資層級B","年資層級C","任職前工作平均年數","畢業科系類別","眷屬量","通勤成本","歸屬部門",
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






