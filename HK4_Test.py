

import pandas as pd
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, MaxPooling1D, LSTM, Dense, Dropout, Flatten
import matplotlib.pyplot as plt

# Step 1: Data Preparation

# Load dataset and take the first 10,000 rows
file_path = r'C:\Users\USER\Desktop\Reviews.csv'
df = pd.read_csv(file_path)
df = df.head(10000)[['Text', 'Score']]

# Convert 'Score' values to binary (positive/negative)
df['Score'] = df['Score'].apply(lambda x: 1 if x >= 4 else 0)

# Split text into words
df['Text'] = df['Text'].astype(str).str.split()

# Remove stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
df['Text'] = df['Text'].apply(lambda x: [word for word in x if word.lower() not in stop_words])

# Join words back for TF-IDF vectorization
df['Text'] = df['Text'].apply(lambda x: ' '.join(x))

# TF-IDF Vectorization
tfidf = TfidfVectorizer(max_features=5000)
X = tfidf.fit_transform(df['Text']).toarray()
y = df['Score'].values

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 2: Modeling with CNN and LSTM

# CNN Model
cnn_model = Sequential()
cnn_model.add(Embedding(input_dim=5000, output_dim=128))
cnn_model.add(Conv1D(filters=128, kernel_size=5, activation='relu'))
cnn_model.add(MaxPooling1D(pool_size=2))
cnn_model.add(Dropout(0.7))
cnn_model.add(Flatten())
cnn_model.add(Dense(10, activation='relu'))
cnn_model.add(Dense(1, activation='sigmoid'))
cnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train CNN Model
cnn_history = cnn_model.fit(X_train, y_train, epochs=10, batch_size=64, validation_split=0.2)

print(cnn_history)

# LSTM Model
lstm_model = Sequential()
lstm_model.add(Embedding(input_dim=5000, output_dim=128))
lstm_model.add(LSTM(128))
lstm_model.add(Dropout(0.7))
lstm_model.add(Dense(1, activation='sigmoid'))
lstm_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train LSTM Model
lstm_history = lstm_model.fit(X_train, y_train, epochs=10, batch_size=64, validation_split=0.2)

print(lstm_history)


# Plot CNN Accuracy and Loss
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(cnn_history.history['accuracy'], label='Train Accuracy')
plt.plot(cnn_history.history['val_accuracy'], label='Validation Accuracy')
plt.title('CNN Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(cnn_history.history['loss'], label='Train Loss')
plt.plot(cnn_history.history['val_loss'], label='Validation Loss')
plt.title('CNN Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

# Plot LSTM Accuracy and Loss
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(lstm_history.history['accuracy'], label='Train Accuracy')
plt.plot(lstm_history.history['val_accuracy'], label='Validation Accuracy')
plt.title('LSTM Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(lstm_history.history['loss'], label='Train Loss')
plt.plot(lstm_history.history['val_loss'], label='Validation Loss')
plt.title('LSTM Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

# Step 3: Model Evaluation with Test Data
cnn_test_loss, cnn_test_accuracy = cnn_model.evaluate(X_test, y_test)
lstm_test_loss, lstm_test_accuracy = lstm_model.evaluate(X_test, y_test)

print(f"CNN Test Accuracy: {cnn_test_accuracy:.2f}")
print(f"LSTM Test Accuracy: {lstm_test_accuracy:.2f}")




作業流程：

1. 資料前處理(可延用HW2之方法)：

a. 讀取 csv 檔後取前 1 萬筆資料

僅保留"Text"、"Score"兩個欄位

並將 "Score" 欄位內值大於等於4的轉成1，其餘轉成0

1: positive

0: negative

並將text欄位內的文字利用分割符號切割


b. 去除停頓詞stop words 

c. 文字轉向量（Tfidf 、Ｗord2vec …等 ）



2. 建模

a. 分別用CNN與LSTM對train的資料進行建模，可自行設計神經網路的架構

b. 加入Dropout Layer設定Dropout參數(建議0.7)進行比較

c. plot出訓練過程中的Accuracy與Loss值變化


3. 評估模型

a. 利用kaggle上test的資料對2.所建立的模型進行測試，並計算Accuracy



