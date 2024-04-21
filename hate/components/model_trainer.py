import os
import pickle
import sys
import inspect
import pandas as pd
import nltk
import inspect
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
nltk.download('stopwords')
from hate.logger import logging
from hate.exception import CustomException
from hate.entity.config_entity import DataTransformationConfig, ModelTrainerConfig
from hate.entity.artifact_entity import DataTransformationArtifacts, DataIngestionArtifacts, DataValidationArtifacts, ModelTrainerArtifacts
from keras.preprocessing.text import Tokenizer
import tensorflow as tf
from keras.utils import pad_sequences
from hate.ml.model import ModelArchitecture
from hate.constants import *

class ModelTrainer:
    def __init__(self, model_trainer_config: ModelTrainerConfig, 
                 data_transformation_artifacts: DataTransformationArtifacts):
        self.model_trainer_config = model_trainer_config
        self.data_transformation_artifacts = data_transformation_artifacts
        
    def splitting_data(self, csv_path):
        current_function_name = inspect.stack()[0][3]
        logging.info(f"Entered the {current_function_name} method of {self.__class__.__name__} class")
        try:        
            df = pd.read_csv(csv_path, index_col=False)
            logging.info("Splitting the data into x & y")
            x = df[TWEET].astype(str)
            y = df[LABEL]
            
            logging.info("Applying the train_test_split on the data")
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=RANDOM_STATE)
            
            logging.info(f"Exited the {current_function_name} method of {self.__class__.__name__} class")
                        
            return x_train, x_test, y_train, y_test
        except Exception as e:
            raise CustomException(e, sys) from e  
    
    def tokenizing_data(self, x_train):
        current_function_name = inspect.stack()[0][3]
        logging.info(f"Entered the {current_function_name} method of {self.__class__.__name__} class")
        try:        
            tokenizer = Tokenizer(num_words=self.model_trainer_config.MAX_WORDS)
            logging.info("Applying Tokenization on data")
            tokenizer.fit_on_texts(x_train)
            sequences = tokenizer.texts_to_sequences(x_train)
            
            
            logging.info(f"Converting text to sequences: {sequences}")
            sequences_matrix = pad_sequences(sequences, maxlen=self.model_trainer_config.MAX_LEN)
            logging.info(f"The sequence matrix is: {sequences_matrix}")
            logging.info(f"Exited the {current_function_name} method of {self.__class__.__name__} class")
                        
            return sequences_matrix, tokenizer
        except Exception as e:
            raise CustomException(e, sys) from e  
    
    def initiate_model_trainer(self,) -> ModelTrainerArtifacts:
        current_function_name = inspect.stack()[0][3]
        logging.info(f"Entered the {current_function_name} method of {self.__class__.__name__} class")
        
        """
        Method Name :   initiate_model_trainer
        Description :   This function initiates a model trainer steps
        
        Output      :   Returns model trainer artifact
        On Failure  :   Write an exception log and then raise an exception
        """

        try:
            x_train,x_test,y_train,y_test = self.splitting_data(csv_path=self.data_transformation_artifacts.transformation_data_file_path)
            model_architecture = ModelArchitecture()   

            model = model_architecture.get_model()



            logging.info(f"Xtrain size is : {x_train.shape}")

            logging.info(f"Xtest size is : {x_test.shape}")

            sequences_matrix, tokenizer =self.tokenizing_data(x_train)


            logging.info("Entered into model training")
            model.fit(sequences_matrix, y_train, 
                        batch_size=self.model_trainer_config.BATCH_SIZE, 
                        epochs = self.model_trainer_config.EPOCH, 
                        validation_split=self.model_trainer_config.VALIDATION_SPLIT, 
                        )
            logging.info("Model training finished")

            
            with open('tokenizer.pickle', 'wb') as handle:
                pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
            os.makedirs(self.model_trainer_config.TRAINED_MODEL_DIR,exist_ok=True)



            logging.info("saving the model")
            model.save(self.model_trainer_config.TRAINED_MODEL_PATH)
            x_test.to_csv(self.model_trainer_config.X_TEST_DATA_PATH)
            y_test.to_csv(self.model_trainer_config.Y_TEST_DATA_PATH)

            x_train.to_csv(self.model_trainer_config.X_TRAIN_DATA_PATH)

            model_trainer_artifacts = ModelTrainerArtifacts(
                trained_model_path = self.model_trainer_config.TRAINED_MODEL_PATH,
                x_test_path = self.model_trainer_config.X_TEST_DATA_PATH,
                y_test_path = self.model_trainer_config.Y_TEST_DATA_PATH)
            logging.info("Returning the ModelTrainerArtifacts")
            return model_trainer_artifacts

        except Exception as e:
            raise CustomException(e, sys) from e