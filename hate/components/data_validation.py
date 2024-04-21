import os
import sys
import inspect
import pandas as pd
from hate.logger import logging
from hate.exception import CustomException
from hate.entity.config_entity import DataValidationConfig
from hate.entity.artifact_entity import DataValidationArtifacts


class DataValidation:
    def __init__(self, data_validation_config: DataValidationConfig):
        self.data_validation_config = data_validation_config
        
    def get_data_from_artifacts(self):
        current_function_name = inspect.stack()[0][3]
        try:
            logging.info(f"Entered the {current_function_name} method of {self.__class__.__name__} class")
            # os.makedirs(self.data_validation_config.IMBALANCE_DATA_DIR, exist_ok=True)
            # os.makedirs(self.data_validation_config.RAW_DATA_DIR, exist_ok=True)

            IMBALANCE_DATA_DF = pd.read_csv(self.data_validation_config.IMBALANCE_DATA_DIR)
            RAW_DATA_DF = pd.read_csv(self.data_validation_config.RAW_DATA_DIR)
            
            logging.info(f"Exited the {current_function_name} method of {self.__class__.__name__} class")
            
            return IMBALANCE_DATA_DF, RAW_DATA_DF
        except Exception as e:
            raise CustomException(e, sys) from e

    def compare_lists(self, list1, list2):
        # Convert lists to sets
        set1 = set(list1)
        set2 = set(list2)
        
        # Check if both sets are equal
        if set1 == set2:
            return True
        else:
            print(list1)
            print(list2)
            return False

    def validate_data(self):
        current_function_name = inspect.stack()[0][3]
        logging.info(f"Entered the {current_function_name} method of {self.__class__.__name__} class")
        try:
            IMBALANCE_DATA_DF, RAW_DATA_DF = self.get_data_from_artifacts()
            
            IMBALANCE_BOOL = self.compare_lists(IMBALANCE_DATA_DF.columns, self.data_validation_config.IMBALANCE_DATA_COLUMNS)
            RAW_BOOL = self.compare_lists(RAW_DATA_DF.columns, self.data_validation_config.RAW_DATA_COLUMNS)
            
            
            logging.info(f"Exited the {current_function_name} method of {self.__class__.__name__} class")
            
            return IMBALANCE_BOOL, RAW_BOOL
        except Exception as e:
            raise CustomException(e, sys) from e
    
    def initiate_data_validation(self) -> DataValidationArtifacts:
        current_function_name = inspect.stack()[0][3]
        logging.info(f"Entered the {current_function_name} method of {self.__class__.__name__} class")
        
        try:
            imbalance_data_valid, raw_data_valid = self.validate_data()
            logging.info(f"Unzipped file & split into train and validation set using {current_function_name} method of {self.__class__.__name__} class")
                        
            data_validation_artifacts = DataValidationArtifacts(
                imbalance_data_valid = imbalance_data_valid, 
                raw_data_valid = raw_data_valid
            )
            
            return data_validation_artifacts
        except Exception as e:
            raise CustomException(e, sys) from e    