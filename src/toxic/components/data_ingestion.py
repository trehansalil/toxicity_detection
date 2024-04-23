import os
import zipfile
import gdown
import inspect
from src.toxic import logging
from src.toxic.utils.common import get_size
from src.toxic.entity.config_entity import DataIngestionConfig
from src.toxic.configuration.kaggle_syncer import KaggleSync


class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig):
        self.data_ingestion_config = data_ingestion_config
        self.kagglesync = KaggleSync()
        
    def get_data_from_cloud(self) -> None:
        current_function_name = inspect.stack()[0][3]
        try:
            logging.info(f"Entered the {current_function_name} method of {self.__class__.__name__} class")
            # os.makedirs(self.data_ingestion_config.DATA_INGESTION_ARTIFACTS_DIR, exist_ok=True)
            dataset_url = self.data_ingestion_config.source_url
            unzip_dir = self.data_ingestion_config.unzip_dir
            os.makedirs(self.data_ingestion_config.root_dir, exist_ok=True)
            logging.info(f"Downloading data from {dataset_url} into folder {unzip_dir}")

            self.kagglesync.sync_folder_from_kaggle(
                self.data_ingestion_config.source_url,
                # self.data_ingestion_config.ZIP_FILE_NAME,
                # self.data_ingestion_config.DATA_INGESTION_ARTIFACTS_DIR,
            )

            logging.info(f"Downloaded data from {dataset_url} into folder {unzip_dir}")            
            


            logging.info(f"Exited the {current_function_name} method of {self.__class__.__name__} class")
        except Exception as e:
            raise e

    def initiate_data_ingestion(self):
        current_function_name = inspect.stack()[0][3]
        logging.info(f"Entered the {current_function_name} method of {self.__class__.__name__} class")
        
        try:
            self.get_data_from_cloud()
            logging.info(f"Fetched the data from Kaggle's GCloud Bucket using {current_function_name} method of {self.__class__.__name__} class")           

        except Exception as e:
            raise e        