import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import pickle

# Load and preprocess the dataset
dataset = pd.read_csv('news_dataset.csv')
en_data = dataset[["label"]]
ohe = OneHotEncoder(drop="first")
ar = ohe.fit_transform(en_data).toarray()
dataset["label"] = pd.DataFrame(ar, columns=[['label_FAKE']])

# Preprocessing function
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = text.lower()
    text = ''.join([char for char in text if char not in string.punctuation])
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)

# Apply preprocessing
dataset["text"] = dataset["text"].astype(str)
dataset["text"] = dataset["text"].apply(preprocess_text)

# Vectorize the text data
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(dataset["text"])
y = dataset['label']

# Train the model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression()
model.fit(X_train, y_train)

# Test accuracy
y_pred = model.predict(X_test)
# Save the model and vectorizer to files
with open("logistic_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)
