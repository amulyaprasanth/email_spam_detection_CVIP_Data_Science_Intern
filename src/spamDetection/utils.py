import os
import pickle
import sys

from src.spamDetection.exception import CustomException


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
