import sys

from src.spamDetection.components.data_ingestion import DataIngestion
from src.spamDetection.components.data_transformation import DataTransformation
from src.spamDetection.exception import CustomException


def main():
    try:
        obj = DataIngestion()
        train_set, test_set = obj.initiate_data_ingestion()
        data_transformation = DataTransformation()
        train_arr, test_arr, _ = data_transformation.initate_data_transformation(train_set, test_set)
    except Exception as e:
        raise CustomException(e, sys)


if __name__ == '__main__':
    main()
