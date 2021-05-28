# NTHU_EE_ML_2021
## 處理方法
1. 因為資料權重有點偏差，所以先將有離職的員工和沒離職的員工以1:1重組<br>
2. 假設年分是沒有關聯的，只抓2017年的年份來做單年的預測離職<br>
3. 因為最高學歷	畢業學校類別有很多資料缺失所以不計算<br>
4. 測試組合session的資料對正確率的提升<br>
5. 假設年分是有關連的，將三年資料做輸入，輸出為最後一年的離職機率<br>
6. 將資料缺失的員工剃除<br>
7. 將時數和請假數等資料做標準化<br>
8. 訓練資料足夠預測模型使用DNN全連接模型<br>

## 使用說明
test.csv和train.csv為題目給定資料<br>
deal_test.csv和deal_train.csv為clearspace.py處理後的結果<br>
main.ipynd為jupyter notebook可執行的主要python程式