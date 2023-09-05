import os
import sys
from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split

from src.spamDetection.exception import CustomException
from src.spamDetection.logger import logging


@dataclass
class DataIngestionConfig:
    raw_data_path = os.path.join("artifacts", "raw_data.csv")
    train_data_path = os.path.join("artifacts", "train_data")
    test_data_path = os.path.join("artifacts", "test_data")


class DataIngestion:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        """
        This function initiates the data ingestion process
        """
        try:
            logging.info("Data Ingestion initiated")
            data = pd.read_csv("data/data_cleaned.csv")
            logging.info("Read data as DataFrame")

            os.makedirs(os.path.dirname(self.data_ingestion_config.raw_data_path), exist_ok=True)
            data.to_csv(self.data_ingestion_config.raw_data_path, index=False, header=True)

            logging.info("Train Test split initiated")
            # Splitting the data into training and test splits
            train_set, test_set = train_test_split(data, test_size=0.2, random_state=42)

            logging.info("Saving training and test datasets")
            # Saving training and test datafiles
            train_set.to_csv(self.data_ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.data_ingestion_config.test_data_path, index=False, header=True)

            logging.info("Data Ingestion completed")
            return (
                self.data_ingestion_config.train_data_path,
                self.data_ingestion_config.test_data_path
            )
        except Exception as e:
            raise CustomException(e, sys)
