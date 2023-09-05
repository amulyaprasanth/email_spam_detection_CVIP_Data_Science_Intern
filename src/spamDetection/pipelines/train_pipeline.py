import sys

from src.spamDetection.components.data_ingestion import DataIngestion
from src.spamDetection.components.data_transformation import DataTransformation
from src.spamDetection.components.model_trainer import ModelTrainer
from src.spamDetection.exception import CustomException


def main():
    try:
        obj = DataIngestion()
        train_set, test_set = obj.initiate_data_ingestion()
        data_transformation = DataTransformation()
        train_sentences, train_labels, test_sentences, test_labels = data_transformation.initiate_data_transformation(
            train_set, test_set)
        model_trainer = ModelTrainer()
        model_trainer.create_and_train(train_sentences, train_labels, test_sentences, test_labels)
    except Exception as e:
        raise CustomException(e, sys)


if __name__ == '__main__':
    main()
