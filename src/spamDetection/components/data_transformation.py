from dataclasses import dataclass

import numpy as np
from sklearn.preprocessing import LabelEncoder

from src.spamDetection.logger import logging
from src.spamDetection.utils import *


@dataclass
class DataTransformationConfig:
    def __init__(self):
        self.preprocessor_obj_path = os.path.join("artifacts", "preprocessor.pkl")


class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def initiate_data_transformation(self, train_path, test_path):
        try:
            logging.info("Data Transformation initiated")
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            label_enc = LabelEncoder()

            input_features_train_df = train_df.drop("class", axis=1)
            input_features_test_df = test_df.drop("class", axis=1)

            target_train_feature_df = train_df["class"]
            target_test_feature_df = test_df["class"]

            logging.info("Applying custom preprocessor function to the training and testing datasets")

            input_features_train_arr = preprocess_text(input_features_train_df)
            input_features_test_arr = preprocess_text(input_features_test_df)

            target_train_feature_arr = label_enc.fit_transform(target_train_feature_df)
            target_test_feature_arr = label_enc.transform(target_test_feature_df)

            train_arr = np.c_[input_features_train_arr, target_train_feature_arr]
            test_arr = np.c_[input_features_test_arr, target_test_feature_arr]

            logging.info("Data Transformation completed")
            return (
                train_arr,
                test_arr,
            )
        except Exception as e:
            raise CustomException(e, sys)
