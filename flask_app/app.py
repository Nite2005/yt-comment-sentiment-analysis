from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import io
import mlflow
import numpy as np
import joblib
import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from  nltk.stem import WordNetLemmatizer
from  mlflow.tracking import MlflowClient
import json

app = Flask(__name__)
CORS(app)

def preprocess_comment(comment):
    """Apply preprocessing transformations to a comment."""
    try:
        # Convert to lowercase
        comment = comment.lower()

        # Remove trailing and leading whitespaces
        comment = comment.strip()

        # Remove newline characters
        comment = re.sub(r'\n', ' ', comment)

        # Remove non-alphanumeric characters, except punctuation
        comment = re.sub(r'[^A-Za-z0-9\s!?.,]', '', comment)

        # Remove stopwords but retain important ones for sentiment analysis
        stop_words = set(stopwords.words('english')) - {'not', 'but', 'however', 'no', 'yet'}
        comment = ' '.join([word for word in comment.split() if word not in stop_words])

        # Lemmatize the words
        lemmatizer = WordNetLemmatizer()
        comment = ' '.join([lemmatizer.lemmatize(word) for word in comment.split()])

        return comment
    except Exception as e:
        print(f"Error in preprocessing comment: {e}")
        return comment


def load_model_and_vectorizer(model_name, model_version, vectorizer_path):
    # Set MLflow tracking URI to your server
    mlflow.set_tracking_uri("http://15.206.117.247:5000/")
    client = MlflowClient()
    model_uri = f"models:/{model_name}/{model_version}"
    model = mlflow.pyfunc.load_model(model_uri)
    vectorizer = joblib.load(vectorizer_path)  # Load the vectorizer
    return model, vectorizer

# Initialize the model and vectorizer
model, vectorizer = load_model_and_vectorizer("yt_chrome_plugin_model", "1", "./tfidf_vectorizer.pkl")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    comments = data.get("comments")  # List of comments

    # Preprocess all comments (list of strings)
    preprocessed_comments = [preprocess_comment(c) for c in comments]

    # Vectorize (directly accepts list of strings)
    X = vectorizer.transform(preprocessed_comments)

    # Convert sparse matrix to DataFrame
    X_df = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())

    # Predict sentiments
    predictions = model.predict(X_df)

    # Build response
    response = [
        {"comment": c, "sentiment": int(p)} 
        for c, p in zip(comments, predictions)
    ]
    return jsonify(response)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=6000)
    
