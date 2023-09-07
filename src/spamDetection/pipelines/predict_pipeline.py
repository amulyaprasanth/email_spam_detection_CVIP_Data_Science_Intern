import sys

import pandas as pd

from src.spamDetection.components.model_trainer import ModelTrainerConfig
from src.spamDetection.exception import CustomException
from src.spamDetection.logger import logging
from src.spamDetection.utils import load_object, preprocess_text
import tensorflow as tf


class PredictPipeline():
    def __init__(self):
        pass

    def predict(self, input_text):
        try:
            model_path = ModelTrainerConfig().pretrained_model_path

            logging.info("Loading model and preprocessor")
            reloaded_artifact = tf.saved_model.load(model_path)
            print("Loading completed")

            print("Generating predictions...")
            input_data = preprocess_text(input_text)
            prediction = reloaded_artifact.serve(input_data)

            return prediction

        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    def __init__(self,
                 text_input: str):
        self.text_input = text_input

    def get_data_as_dataframe(self):
        try:
            custom_data_dict = {
                "text": [self.text_input]
            }

            return pd.DataFrame(custom_data_dict)
        except Exception as e:
            raise CustomException(e, sys)
