




import pandas as pd
import numpy as np
# Load the Excel file and parse the required sheet
file_path = r'C:\Users\10435\Downloads\Reviews.csv'

data = pd.read_csv(file_path)


# Take the first 10,000 rows and keep only "Text" and "Score" columns
data = data[['Text', 'Score']].head(10000)

# Transform "Score" - 1 if Score >= 4, else 0
data['Score'] = data['Score'].apply(lambda x: 1 if x >= 4 else 0)


from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
import nltk

# Download stop words if not already available
nltk.download('stopwords')

# Define stop words
stop_words = set(stopwords.words('english'))

# Split text based on whitespace, remove stop words, and join back into a string
data['Text'] = data['Text'].apply(lambda x: ' '.join([word for word in str(x).split() if word.lower() not in stop_words]))

# Initialize TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(data['Text'])

# Converting to DataFrame for better readability (using dense form for the first 5 rows to preview)
tfidf_df = pd.DataFrame(tfidf_matrix[:5].todense(), columns=tfidf_vectorizer.get_feature_names_out())

# Displaying the TF-IDF transformed data to the user
#tools.display_dataframe_to_user(name="TF-IDF Vectorized Text (Sample)", dataframe=tfidf_df)




import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, MaxPooling1D, Flatten, Dense, LSTM, Bidirectional, Dropout
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

# Prepare data for modeling
X = data['Text'].values
y = data['Score'].values

# Tokenize the text data
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X)
X_seq = tokenizer.texts_to_sequences(X)

# Define maximum sequence length and pad sequences
max_length = 100
X_padded = pad_sequences(X_seq, maxlen=max_length, padding='post')

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_padded, y, test_size=0.2, random_state=42)

# Get vocabulary size
vocab_size = len(tokenizer.word_index) + 1

# CNN Model
cnn_model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=128, input_length=max_length),
    Conv1D(filters=64, kernel_size=5, activation='relu'),
    MaxPooling1D(pool_size=2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

# Compile CNN model
cnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# LSTM Model
lstm_model = Sequential([
    Embedding(input_dim=vocab_size, output_dim=128, input_length=max_length),
    Bidirectional(LSTM(64, return_sequences=True)),
    LSTM(32),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

# Compile LSTM model
lstm_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Summary of models
print(cnn_model.summary())
print(lstm_model.summary())


