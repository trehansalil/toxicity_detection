import os
import zipfile
import gdown
import inspect
from toxic import logging
from toxic.utils.common import get_size
from toxic.entity.config_entity import DataIngestionConfig
from toxic.configuration.configuration import KaggleSync


class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig):
        self.data_ingestion_config = data_ingestion_config
        self.kagglesync = KaggleSync()
        
    def get_data_from_cloud(self) -> None:
        current_function_name = inspect.stack()[0][3]
        try:
            logging.info(f"Entered the {current_function_name} method of {self.__class__.__name__} class")
            # os.makedirs(self.data_ingestion_config.DATA_INGESTION_ARTIFACTS_DIR, exist_ok=True)
            dataset_url = self.data_ingestion_config.source_URL
            zip_download_dir = self.data_ingestion_config.local_data_file
            os.makedirs("artifacts/data_ingestion", exist_ok=True)
            logging.info(f"Downloading data from {dataset_url} into file {zip_download_dir}")

            self.kagglesync.sync_folder_from_gcloud(
                self.data_ingestion_config.source_URL,
                # self.data_ingestion_config.ZIP_FILE_NAME,
                # self.data_ingestion_config.DATA_INGESTION_ARTIFACTS_DIR,
            )

            logging.info(f"Downloaded data from {dataset_url} into file {zip_download_dir}")            
            


            logging.info(f"Exited the {current_function_name} method of {self.__class__.__name__} class")
        except Exception as e:
            raise e

    # def unzip_and_clean(self):
    #     current_function_name = inspect.stack()[0][3]
    #     logging.info(f"Entered the {current_function_name} method of {self.__class__.__name__} class")
    #     try:
    #         with ZipFile(self.data_ingestion_config.ZIP_FILE_PATH) as zip_ref:
    #             zip_ref.extractall(self.data_ingestion_config.ZIP_FILE_DIR)
            
    #         logging.info(f"Exited the {current_function_name} method of {self.__class__.__name__} class")
            
    #         return self.data_ingestion_config.DATA_ARTIFACTS_DIR, self.data_ingestion_config.NEW_DATA_ARTIFACTS_DIR
    #     except Exception as e:
    #         raise CustomException(e, sys) from e
    
    # def initiate_data_ingestion(self) -> DataIngestionArtifacts:
    #     current_function_name = inspect.stack()[0][3]
    #     logging.info(f"Entered the {current_function_name} method of {self.__class__.__name__} class")
        
    #     try:
    #         self.get_data_from_cloud()
    #         logging.info(f"Fetched the data from GCloud Bucket using {current_function_name} method of {self.__class__.__name__} class")
            
    #         imbalance_data_file_path, raw_data_file_path = self.unzip_and_clean()
    #         logging.info(f"Unzipped file & split into train and validation set using {current_function_name} method of {self.__class__.__name__} class")
                        
    #         data_ingestion_artifacts = DataIngestionArtifacts(
    #             imbalance_data_file_path = imbalance_data_file_path, 
    #             raw_data_file_path = raw_data_file_path
    #         )
            
    #         return data_ingestion_artifacts
    #     except Exception as e:
    #         raise CustomException(e, sys) from e        