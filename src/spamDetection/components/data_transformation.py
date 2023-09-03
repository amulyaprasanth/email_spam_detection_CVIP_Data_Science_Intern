import re

import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder

from src.spamDetection.logger import logging
from src.spamDetection.utils import *

# nltk.download('punkt')
nltk.download('stopwords')
# nltk.download('wordnet')


from dataclasses import dataclass


class TextPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_transformed = X.apply(self._preprocess_text)
        return X_transformed

    def _preprocess_text(self, text):
        # Remove punctuation marks and URLs
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'http\S+|www.\S+', '', text)

        # Tokenize the text
        tokens = word_tokenize(text)

        # Remove stopwords and lowercase the tokens
        filtered_tokens = [word.lower() for word in tokens if word.lower() not in self.stop_words]

        # Apply stemming and lemmatization
        stemmed_tokens = [self.stemmer.stem(word) for word in filtered_tokens]
        lemmatized_tokens = [self.lemmatizer.lemmatize(word) for word in stemmed_tokens]

        return ' '.join(lemmatized_tokens)


@dataclass
class DataTransformationConfig:
    def __init__(self):
        self.preprocessor_obj_path = os.path.join("artifacts", "preprocessor.pkl")


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_preprocessor_obj(self):
        try:
            preprocessor = ColumnTransformer([
                ('preprocessor', TextPreprocessor(), ['text'])
            ])

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initate_data_transformation(self, train_path, test_path):
        try:
            logging.info("Data Transformation initiated")
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Getting preprocessor object")

            preprocessor_obj = self.get_preprocessor_obj()
            label_enc = LabelEncoder()

            input_features_train_df = train_df["text"]
            input_features_test_df = test_df["text"]

            target_train_feature_df = train_df["class"]
            target_test_feature_df = test_df["class"]

            logging.info("Applying preprocessor object to the training and testing datasets")

            input_features_train_arr = preprocessor_obj.fit_transform(input_features_train_df)
            input_features_test_arr = preprocessor_obj.transform(input_features_test_df)

            target_train_feature_arr = label_enc.fit_transform(target_train_feature_df)
            target_test_feature_arr = label_enc.transform(target_test_feature_df)

            train_arr = np.c_[input_features_train_arr, target_train_feature_arr]
            test_arr = np.c_[input_features_test_arr, target_test_feature_arr]

            logging.info("Data Transformation completed")
            logging.info("Saving preprocessor object")
            save_object(self.data_transformation_config.preprocessor_obj_path,
                        preprocessor_obj)
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_path
            )
        except Exception as e:
            raise CustomException(e, sys)
