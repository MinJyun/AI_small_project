# AI_small_project

## Sentiment Classifier (情緒分類器)

這是一個簡單的情緒分類模型，使用 Python 與 Scikit-learn 建立，能根據輸入文字判斷為正向（Positive）或負向（Negative）情緒。

---

### 功能簡介

- 使用 **TF-IDF 向量化** 將文字轉為數值特徵
- 使用 **邏輯迴歸（Logistic Regression）** 建立分類模型
- 提供混淆矩陣與分類報告來評估模型表現
- 可輸入句子進行即時情緒預測

---

### 資料說明

- **欄位 `label`**:
  - `0` 表示 **Positive**
  - `1` 表示 **Negative**
- **欄位 `text`**: 使用者評論文字內容

---

### 環境需求

請先安裝以下套件：

```bash
pip install pandas scikit-learn
