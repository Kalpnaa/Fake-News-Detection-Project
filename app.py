from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import re
import pickle
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string

app = Flask(__name__)
CORS(app)

# Load the trained model and vectorizer
with open("logistic_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

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

# URL-based fake news detection
def predict_url_reliability(url):
    reliable_sources = [
        'https://bbc.com', 'https://cnn.com', 'https://nytimes.com',
        'https://reuters.com', 'https://theguardian.com', 'https://apnews.com',
        'https://aljazeera.com', 'https://washingtonpost.com', 'https://npr.org',
        'https://independent.co.uk', 'https://bloomberg.com', 'https://time.com',
        'https://forbes.com', 'https://theatlantic.com', 'https://economist.com',
        'https://usatoday.com', 'https://newsweek.com', 'https://latimes.com',
        'https://wsj.com', 'https://ft.com'
    ]

    for source in reliable_sources:
        if source in url:
            return 'Reliable'
    return 'Unreliable'

# Text-based fake news detection
def make_prediction(text):
    processed_text = preprocess_text(text)
    text_vector = vectorizer.transform([processed_text])
    prediction = model.predict(text_vector)
    return 'FAKE' if prediction[0] == 0 else 'REAL'

# Routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/url_detection.html')
def url_detection():
    return render_template('url_detection.html')

@app.route('/text_detection.html')
def text_detection():
    return render_template('text_detection.html')



@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    if 'url' in data:
        url = data['url']
        result = predict_url_reliability(url)
        return jsonify({"prediction": result})
    elif 'text' in data:
        text = data['text']
        result = make_prediction(text)
        return jsonify({"prediction": result})
    else:
        return jsonify({"error": "No input provided"}), 400

if __name__ == '__main__':
    app.run(debug=True,port=5001)
