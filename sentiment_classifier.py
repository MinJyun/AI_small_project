import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# 載入資料集
url = "https://raw.githubusercontent.com/dD2405/Twitter_Sentiment_Analysis/master/train.csv"
df = pd.read_csv(url)[['label', 'tweet']]
print(df['label'].value_counts())

df = df.rename(columns={"label": "sentiment", "tweet": "text"})
df['sentiment'] = df['sentiment'].map({0: 'positive', 1: 'negative'})
df = df.dropna().drop_duplicates().reset_index(drop=True)

# 平衡資料
min_count = df['sentiment'].value_counts().min()
df_balanced = df.groupby('sentiment').sample(n=min_count, random_state=42)

# 檢查
print(df_balanced['sentiment'].value_counts())
# print(df_balanced['text'].values)

# 切分
X_train, X_test, y_train, y_test = train_test_split(
    df_balanced['text'], df_balanced['sentiment'], test_size=0.2, random_state=42
)

# 向量化
vectorizer = TfidfVectorizer(stop_words='english')
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# 訓練模型
model = LogisticRegression()
model.fit(X_train_vec, y_train)

# 模型評估
y_pred = model.predict(X_test_vec)
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# 測試輸入
def predict_sentiment(text):
    vec = vectorizer.transform([text])
    pred = model.predict(vec)[0]
    return pred

sample = "I hate this product. Worst ever."
print(f"Prediction: '{sample}' → {predict_sentiment(sample)}")
