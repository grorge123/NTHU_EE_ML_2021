tt_path = "test.csv"
tr_path = "train.csv"
import pandas as pd # 引用套件並縮寫為 pd  


##將test的label標成-1
df = pd.read_csv(tt_path, encoding="Big5")

print(df)

for i in range(len(df["PerStatus"])):
    df["PerStatus"][i] = -1

# dell = []
# for i in range(len(df["PerStatus"])):
#     if(df["sex"][i] != 0 and df["sex"][i] != 1):
#         dell.append(i)
#         print(i, df["sex"][i])
# df = df.drop(dell)


for i in range(len(df["PerStatus"])):
    if(df["sex"][i] != 0 and df["sex"][i] != 1):
        for q in df.columns.values.tolist()[3:]:
            df[q][i] = -1


df.to_csv("deal_test.csv",  encoding="Big5")


df = pd.read_csv(tr_path, encoding="Big5")

print(df)
for i in range(len(df["PerStatus"])):
    if(df["sex"][i] != 0 and df["sex"][i] != 1):
        for q in df.columns.values.tolist()[3:]:
            df[q][i] = -1

df.to_csv("deal_train.csv",  encoding="Big5")
