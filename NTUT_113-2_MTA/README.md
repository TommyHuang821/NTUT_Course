# 國立台北科技大學 113學年度第2學期 多媒體技術與應用
## 公告:<br>  
📥 [下載此課程資料](https://download-directory.github.io/?url=https://github.com/TommyHuang821/NTUT_Course/tree/main/NTUT_113-2_MTA)




此github為黃志勝業師授課部分公告教教材和作業課程資料用，陳彥霖老師部分請到北科大i學園plus<br>  
此次上課內容為
- Numba
- 生成式AI起手式: Language model、VAE、Diffusiion model、Stabe Diffusiion(文生圖)、圖生圖等。
  
所有GAI都是*單機版*，會給範例程式，你可以回家練習。


## 本課程以實體課程方式進行: 地點→北科科研大樓 1222

* **授課教師：** <br>
陳彥霖教授、黃志勝博士 <br>

* **業師評分部分：** <br>
作業: 90% <br>
出席: 10% <br>
課程問答: 加分<br>

**繳交要求**: 作業為個人作業<br>
**作業繳交項目**: 程式檔 ipynb <br> 繳交期限為5/29<br>
超過時間遲交每隔一週（含一週內）分數打 8 折，採累計連乘方式  <br>
舉例：<br>
遲交三天－以遲交一週計算<br>
遲交的項目分數 >*0.8 = 該項目得到的分數<br>
遲交九天－以遲交兩週計算<br>
遲交的項目分數 >*0.8 *0.8 = 該項目得到的分數<br>
**遲交兩週以上作業不予補繳**



* **程式語言** <br>
Python

* **教材** <br>
教師自行製作教材


|週次|上課日期|課程進度、內容、主題|備註| 地點 |
|:---:|:---:|:---|:---|:---|
|第01週|2024/05/07 | [Numba](https://github.com/TommyHuang821/NTUT_Course/blob/main/NTUT_113-2_MTA/code/main_numba.ipynb) ||科研1222|
|第01週|2024/05/12 | 作業一: [Numba練習](https://github.com/TommyHuang821/NTUT_Course/blob/main/NTUT_113-2_MTA/code/main_numba_pratice.ipynb) ||科研1222|
|第02週|2024/05/14 |- [生成式AI起手式-Language Model_part1 (Tokenizer為主, Transformer額外](https://github.com/TommyHuang821/NTUT_Course/blob/main/NTUT_113-2_MTA/ppt/%E7%94%9F%E6%88%90%E5%BC%8FAI%E8%B5%B7%E6%89%8B%E5%BC%8F-Language%20Model_part1.pdf) <br>- [NLP Tokenizer](https://github.com/TommyHuang821/NTUT_Course/blob/main/NTUT_113-2_MTA/code/main_Tokenizer.ipynb) <br>- [LLM互動](https://github.com/TommyHuang821/NTUT_Course/blob/main/NTUT_113-2_MTA/code/llmresponse.ipynb)|[Seq2Seq詳細介紹看這份投影片](https://github.com/TommyHuang821/NTUT_Course/blob/main/NTUT_113-1/ppt/week03_RNN%26Transformer.pdf)|科研1222|
|第02週|2024/05/19 |作業二: [llmresponse練習](https://github.com/TommyHuang821/NTUT_Course/blob/main/NTUT_113-2_MTA/code/llmresponse_homework.ipynb)|LLM的回答請加上限制<br>「請以中文回答，幫我以50字以內回答」|科研1222|
|第03週|2024/05/21 |[生成式AI起手式2-VAE&DDPM&SD](https://github.com/TommyHuang821/NTUT_Course/blob/main/NTUT_113-2_MTA/ppt/%E7%94%9F%E6%88%90%E5%BC%8FAI%E8%B5%B7%E6%89%8B%E5%BC%8F2-VAE%26DDPM%26SD.pdf)<br>code: [VAE](https://github.com/TommyHuang821/NTUT_Course/blob/main/NTUT_113-2_MTA/code/main_vae.ipynb)<br> code: [DDPM (MNIST)](https://github.com/TommyHuang821/NTUT_Course/blob/main/NTUT_113-2_MTA/code/main_Diffusion_handexample.ipynb) (文件內title打錯應該是DDPM)<br>code: [Stable Diffusion](https://github.com/TommyHuang821/NTUT_Course/blob/main/NTUT_113-2_MTA/code/main_diffusers_StableDiffusion.ipynb)||科研1222|
|第03週|2024/05/26 |作業三: 交出一份.ipynb<br> 1. 生成一張圖片(程式碼、Prompt和參數要寫清楚)80%: 生成一個人，主體只有上半身，至少要一隻手在畫面上，主題不限性別不限<br>2. 寫出這兩周的GAI課程在操作上你的心得，從GPU或是CPU操作的感想來比較 20%||科研1222|




**作業注意事項** <br>
作業一: 題目三和四需要GPU加速，如果無法操作GPU請使用CPU版本(@jit修飾)<br>
用CPU寫分數不變，用GPU寫會額外+10分。

作業三: 
第一題.
- 生成內容五官清楚+5隻手指很清楚100分
- 生成內容五官清楚+手指不清楚95分
- 生成內容五官不清楚+手指清楚90分
- 生成內容五官不清楚+手指不清楚85分
- 都生不出來 → 嘗試到做到生出來吧 → 真的不行找助教協助跑你寫的程式，但Prompt需自行提供給助教(產生的內容評分同上)

------
# OLLAMA下載
[<img src="https://ollama.com/public/assets/c889cc0d-cb83-4c46-a98e-0d0e273151b9/42f6b28d-9117-48cd-ac0d-44baaf5c178e.png" width="100" alt="縮小圖片">](https://ollama.com/download) (ollama下載直接點圖)

[OLLAMA語言模型下載](https://ollama.com/library)

