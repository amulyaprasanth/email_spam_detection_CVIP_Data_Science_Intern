import os
import sys
from dataclasses import dataclass

import keras.callbacks
import tensorflow as tf
from keras import layers

from src.spamDetection.exception import CustomException
from src.spamDetection.logger import logging
from src.spamDetection.utils import save_object

tf.random.set_seed(42)


@dataclass
class ModelTrainerConfig:
    pretrained_model_path = os.path.join("artifacts", "pretrained_model.keras")


MAX_VOCAB_LENGTH = 10000
MAX_LEN = 25


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def create_and_train(self, train_sentences, train_labels,
                         test_sentences, test_labels,
                         epochs: int = 1,
                         batch_size=1,
                         early_stopping_patience=10,
                         summary: bool = False):
        try:
            logging.info("Creating model")
            text_vectorizer = layers.TextVectorization(max_tokens=MAX_VOCAB_LENGTH,
                                                       output_mode="int",
                                                       output_sequence_length=MAX_LEN)
            text_vectorizer.adapt(train_sentences)
            embedding = layers.Embedding(input_dim=MAX_VOCAB_LENGTH,
                                         output_dim=128,
                                         embeddings_initializer="uniform",
                                         input_length=MAX_LEN,
                                         name="embedding")

            # Build model with the Functional API
            inputs = layers.Input(shape=(1,), dtype="string")
            x = text_vectorizer(inputs)
            x = embedding(x)
            x = layers.GlobalAveragePooling1D()(x)
            outputs = layers.Dense(1, activation="sigmoid")(x)
            model = tf.keras.Model(inputs, outputs, name="dense_model")

            logging.info("Compile the model")
            # compile the model
            model.compile(loss="binary_crossentropy",
                          optimizer=tf.keras.optimizers.Adam(learning_rate=2e-3),
                          metrics=["accuracy"])

            logging.info("Model creation and compilation done successfully")

            if summary:
                print(model.summary())

            logging.info(f"Training the model for {epochs} epochs")
            model.fit(train_sentences, train_labels,
                      epochs=epochs,
                      batch_size=batch_size,
                      validation_data=(test_sentences, test_labels),
                      callbacks=[keras.callbacks.EarlyStopping(patience=early_stopping_patience,
                                                               restore_best_weights=True)])
            logging.info("Model training completed")
            logging.info("Model Evaluation intitiated")
            logging.info("accuracy: {}%".format(round(model.evaluate(test_sentences, test_labels)[1] * 100, 2)))
            logging.info("Saving the model")
            model.save(self.model_trainer_config.pretrained_model_path)
            return model

        except Exception as e:
            raise CustomException(e, sys)
