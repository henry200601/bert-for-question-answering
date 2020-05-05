# 檔案說明:
在自然語言處理中，fine-tune pre-trained model 是非常重要的技術。在這份程式的目的為訓練一個模型，
讓模型看一段中文文章，和一個中文問題，讓機器可以找出答案在文章中所在的位置，屬於span selection model。  
此模型用Bert model做fine-tune。
# 使用方法:
## 下載所需檔案 
bash ./download.sh

## 模型預測
```
bash ./run.sh TEST_PATH PREDICT_PATH 
```
## earlybird 模型預測
```
bash ./early.sh TEST_PATH PREDICT_PATH 
```
## 訓練模型
```
bash ./train.sh TRAIN_PATH DEV_PATH  
```
路徑下生成模型檔 testtt01.h5
# 畫圖
## 第五題
```
bash ./plot_answer_len.sh TRAIN_PATH  
```
路徑下生成圖檔 answer_length.png
## 第六題

### 產生5個不同threshold的預測檔案
```
bash ./threshold.sh TRAIN_PATH
```
### 請先下載安裝ckiptagger model，切換到tensorflow = 1.15.0的環境，再執行下式
```
bash ./threshold_score.sh DEV_PATH ckiptagger_model_path
```
ckiptagger_model_path為模型所在的資料夾
### 在路徑下產生圖檔ans_threshold.png
