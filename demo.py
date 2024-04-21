from hate.logger import logging
from hate.exception import CustomException
from hate.configuration.gcloud_syncer import GCloudSync

import sys

# logging.info("Welcome to our Project")

# try:
#     a= 7/"0"
# except Exception as e:
#     raise CustomException(e, sys) from e

obj = GCloudSync()
obj.sync_folder_from_gcloud("hatespeech2024", 'dataset.zip', 'data')