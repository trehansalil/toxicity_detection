import os
import sys
import inspect
import pandas as pd
import nltk
import re
import string
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
nltk.download('stopwords')
from hate.logger import logging
from hate.exception import CustomException
from hate.entity.config_entity import DataTransformationConfig
from hate.entity.artifact_entity import DataTransformationArtifacts, DataIngestionArtifacts, DataValidationArtifacts


class DataTransformation:
    def __init__(self, data_transformation_config: DataTransformationConfig, data_ingestion_artifacts: DataIngestionArtifacts):
        self.data_transformation_config = data_transformation_config
        self.data_ingestion_artifacts = data_ingestion_artifacts
        
    def imbalance_data_cleaning(self):
        current_function_name = inspect.stack()[0][3]
        try:
            logging.info(f"Entered the {current_function_name} method of {self.__class__.__name__} class")

            IMBALANCE_DATA_DF = pd.read_csv(self.data_ingestion_artifacts.imbalance_data_file_path)
            IMBALANCE_DATA_DF.drop(columns=[self.data_transformation_config.ID], 
                                   inplace=self.data_transformation_config.INPLACE)
            
            logging.info(f"Exited the {current_function_name} method of {self.__class__.__name__} class")
            
            return IMBALANCE_DATA_DF
        except Exception as e:
            raise CustomException(e, sys) from e
        
    def raw_data_cleaning(self):
        current_function_name = inspect.stack()[0][3]
        try:
            logging.info(f"Entered the {current_function_name} method of {self.__class__.__name__} class")

            RAW_DATA_DF = pd.read_csv(self.data_ingestion_artifacts.raw_data_file_path)
            RAW_DATA_DF[self.data_transformation_config.LABEL] = RAW_DATA_DF[self.data_transformation_config.CLASS].map(self.data_transformation_config.MAPPING_CLASS_COL_DICT)
            RAW_DATA_DF.drop(columns=self.data_transformation_config.DROP_COLUMNS, 
                                   inplace=self.data_transformation_config.INPLACE)
            
            logging.info(f"Exited the {current_function_name} method of {self.__class__.__name__} class")
            
            return RAW_DATA_DF
        except Exception as e:
            raise CustomException(e, sys) from e        

    def concatenate_data(self):
        current_function_name = inspect.stack()[0][3]
        logging.info(f"Concatenating the data using the {current_function_name} method of {self.__class__.__name__} class")
        try:
            df = pd.concat([self.imbalance_data_cleaning()[[self.data_transformation_config.LABEL, self.data_transformation_config.TWEET]], 
                            self.raw_data_cleaning()[[self.data_transformation_config.LABEL, self.data_transformation_config.TWEET]]])
            print(df.head())
            
            logging.info(f"Returned the concatenated data using the {current_function_name} method of {self.__class__.__name__} class")
            
            return df
        
        except Exception as e:
            raise CustomException(e, sys) from e
        
    def data_cleaning(self, words):
        try:
            stemmer = nltk.SnowballStemmer('english')
            stopword = set(stopwords.words('english'))
            words = str(words).lower()
            words = re.sub('\[.*?\]', '', words)
            words = re.sub('https?://\S+|www\.\S+', '', words)
            words = re.sub('<.*?>+', '', words)
            words = re.sub('[%s]' % re.escape(string.punctuation), '', words)
            words = re.sub('\n', '', words)
            words = re.sub('\w*\d\w*', '', words)
            words = [word for word in words.split(' ') if word not in stopword]
            words = ' '.join(words)
            words = [stemmer.stem(word) for word in words.split(' ')]
            words = ' '.join(words)

            return words
        
        except Exception as e:
            raise CustomException(e, sys) from e    
        
    def initiate_data_transformation(self) -> DataTransformationArtifacts:
        current_function_name = inspect.stack()[0][3]
        logging.info(f"Entered the {current_function_name} method of {self.__class__.__name__} class")
        
        try:
            df = self.concatenate_data()
            df[self.data_transformation_config.TWEET] = df[self.data_transformation_config.TWEET].apply(self.data_cleaning)
            print(self.data_transformation_config.DATA_TRANSFORMATION_ARTIFACTS_DIR)
            os.makedirs(self.data_transformation_config.DATA_TRANSFORMATION_ARTIFACTS_DIR, exist_ok=True)
            df.to_csv(self.data_transformation_config.TRANSFORMED_FILE_PATH, index=False, header=True)
            
                        
            data_transformation_artifacts = DataTransformationArtifacts(
                transformation_data_file_path = self.data_transformation_config.TRANSFORMED_FILE_PATH, 
            )
            
            logging.info(f"Returing the DataTransformationArtifacts using {current_function_name} method of {self.__class__.__name__} class")
            
            return data_transformation_artifacts
        except Exception as e:
            raise CustomException(e, sys) from e 