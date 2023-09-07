import os
import pickle
import re
import sys

import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from src.spamDetection.exception import CustomException

# Download NLTK resources if not already downloaded
nltk.download('punkt')
nltk.download('stopwords')


nltk.download('wordnet')


# Function for preprocessing nlp pipeline
# Step 1: Remove Punctuation Marks and URLs
def remove_punctuation_and_urls(text):
    # Remove punctuation marks
    text = re.sub(r'[^\w\s]', '', text)

    # Remove URLs
    text = re.sub(r'http\S+|www.\S+', '', text)

    return text


# Step 2: Remove Stop Words and Lowercase
stop_words = set(stopwords.words('english'))


def remove_stopwords_and_lowercase(text):
    # Tokenize the text
    tokens = word_tokenize(text)

    # Remove stopwords and lowercase the tokens
    filtered_tokens = [word.lower() for word in tokens if word.lower() not in stop_words]

    return ' '.join(filtered_tokens)


# Step 3: Tokenization, Stemming, and Lemmatization
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()


def tokenize_stem_lemmatize(text):
    # Tokenize the cleaned text
    tokens = word_tokenize(text)

    # Apply stemming and lemmatization
    stemmed_tokens = [stemmer.stem(word) for word in tokens]
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in stemmed_tokens]

    return ' '.join(lemmatized_tokens)


def preprocess_text(df: pd.DataFrame):
    df['text'] = df['text'].apply(remove_punctuation_and_urls)
    df['text'] = df['text'].apply(remove_stopwords_and_lowercase)
    df['text'] = df['text'].apply(tokenize_stem_lemmatize)
    return df


def save_object(filepath, obj):
    try:
        dir_path = os.path.dirname(filepath)
        os.makedirs(dir_path, exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(obj, f)

    except Exception as e:
        raise CustomException(e, sys)


def load_object(filepath):
    try:
        with open(filepath, 'rb') as f:
            return pickle.load(f)

    except Exception as e:
        raise CustomException(e, sys)
