# Final Project Competition
**繳交內容**
1. 訓練程式碼: 以一個jpynb繳交。
2. 訓練好的模型: 繳交onnx和pytorch的.pt。
3. Inference程式碼: 以一個jpynb繳交，以for loop進行單張圖片讀取，並且利用onnxruntime執行你訓練好的onnx模型。<br>
    **不要用pytorch dataloader執行，也不要用pytorch，請以onnxruntime跑模型**

**模型請以onnx檔案評估大小，需壓制在20MB以下。**

-------------------------------------------------------------
### 繳交期限2023/01/08，超過不給補交。<br>
助教會在1/8-1/11評比，並且於1/12上課前公告，大家對於成績有疑問可以於1/12做最後發問。<br>

----------------------------------------------------------------
分三個大組，皆為分類任務 <br>
兩組是前方ADAS影像(58度)，圖片是灰階(RCCC)<br>
一組是後方魚眼影像(180度)，圖片是彩色(RGB)<br>

### 資料數量
資料數量每一個類有100張圖片，測試資料只有助教有每類都200張。<br>


## 第一大組
前方ADAS影像(58度)，圖片是灰階(RCCC)，以24bit RGB圖片儲存，R、G、B三通道數值接一致<br>
![REC19700101-233103-59 0067](https://user-images.githubusercontent.com/25295252/207522232-05028661-9715-45c9-9b21-acd07d3c96eb.jpg)
任務是情境分類
類別有
|類別|編碼|
|:-:|:-:|
|Day_rain|0|
|day_sunny|1|
|day-cloudy|2|
|dusk_cloudy|3|
|dusk_rain|4|
|dusk_sunny|5|
|night_cloudy|6|
|night_rain|7|
|night_sunny|8|

## 第二大組
前方ADAS影像(58度)，圖片是灰階(RCCC)，以24bit RGB圖片儲存，R、G、B三通道數值接一致: 和第一大組圖片一樣<br>
![image](https://user-images.githubusercontent.com/25295252/207523557-d7689091-774b-455a-8a94-a849effb442c.png)
任務是場景分類
類別有
|類別|編碼|
|:-:|:-:|
|country|0|
|freeway|1|
|urban|2|

## 第三大組
後方魚眼影像(180度)，圖片是彩色(RGB)，以24bit RGB圖片儲存<br>
![24 REC19700101-003019-25 0166](https://user-images.githubusercontent.com/25295252/207524107-e3b6e28c-22c9-4b47-9775-e3230c3eaae8.jpg)
任務是情境分類
類別有
|類別|編碼|
|:-:|:-:|
|day-cloudy|0|
|day_rain|1|
|day_sunny|2|
|dusk_cloudy|3|
|dusk_sunny|4|
|night_cloudy|5|
|night_sunny|6|


### 比賽成績評比
最終成績以圖片分類總正確率(%)和F1-Score評比。<br>
以助教成績當作baseline:助教完成訓練會公告其training set和Test set成績。<br>

超過助教成績85分起跳。<br>
每一大組有五小組，在大家都有繳交情況下<br>
假設5組都超過助教比賽成績，分數為90、89、88、87、86。<br>
假設4都超過助教比賽成績，分數為90、89、88、87。<br>
假設3組都超過助教比賽成績，分數為90、89、88。<br>
假設2組都超過助教比賽成績，分數為90、89。<br>
假設1組都超過助教比賽成績，分數為90。<br>
假設0組都超過助教比賽成績，分數就**80**上下微調。<br>
以上成績會依照程式碼完整度和報告完整度進行調整，上述評比只做為基本參考，非最終打分數方式。<br>
