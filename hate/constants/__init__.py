import os

from datetime import datetime


# common constants
TIMESTAMP: str = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
ARTIFACTS_DIR = os.path.join("artifacts", TIMESTAMP)
BUCKET_NAME = 'hatespeech2024'
ZIP_FILE_NAME = 'dataset.zip'
LABEL = 'label'
TWEET = 'tweet'


# Data ingestion constants
DATA_INGESTION_ARTIFACTS_DIR = "DataIngestionArtifacts"
DATA_INGESTION_IMBALANCE_DATA_DIR = "imbalanced_data.csv"
DATA_INGESTION_RAW_DATA_DIR = "raw_data.csv"


# Data validation constants
IMBALANCE_DATA_DIR = os.path.join(ARTIFACTS_DIR, DATA_INGESTION_ARTIFACTS_DIR, DATA_INGESTION_IMBALANCE_DATA_DIR)
RAW_DATA_DIR = os.path.join(ARTIFACTS_DIR, DATA_INGESTION_ARTIFACTS_DIR, DATA_INGESTION_RAW_DATA_DIR)
IMBALANCE_DATA_COLUMNS = ['id', 'label', 'tweet']
RAW_DATA_COLUMNS = ['Unnamed: 0', 'count', 'hate_speech', 'offensive_language',	'neither', 'class', 'tweet']

# Data transformation constants
DATA_TRANSFORMATION_ARTIFACTS_DIR = "DataTransformationArtifacts"
TRANSFORMED_FILE_NAME = "final.csv"
DATA_DIR = "data"
ID = 'id'
AXIS = 1
DROP_COLUMNS = ['Unnamed: 0', 'count', 'hate_speech', 'offensive_language',	'neither', 'class']
CLASS = 'class'
MAPPING_CLASS_COL_DICT = {0: 1, 1:1, 2: 0}
LABEL = 'label'
TWEET = 'tweet'
INPLACE = True

# Model Trainer constants
MODEL_TRAINER_ARTIFACTS_DIR = "ModelTrainerArtifacts"
TRAINED_MODEL_DIR = 'trained_model'
TRAINED_MODEL_NAME = 'model.h5'
X_TEST_FILE_NAME = 'xtest.csv'
Y_TEST_FILE_NAME = 'ytest.csv'

X_TRAIN_FILE_NAME = 'x_train.csv'
RANDOM_STATE = 42
EPOCH = 2
BATCH_SIZE = 128
VALIDATION_SPLIT = 0.2

# Model Arhcitecture constants
MAX_WORDS = 5000
MAX_LEN = 300
LOSS = 'binary_crossentropy'
METRICS = ['accuracy']
ACTIVATION = 'sigmoid'

# Model Evaluation constants
MODEL_EVALUATION_ARTIFACTS_DIR = "ModelEvaluationArtifacts"
BEST_MODEL_DIR = "best_Model"
MODEL_EVALUATION_FILE_NAME = 'loss.csv'


MODEL_NAME = 'model.h5'
APP_HOST = '0.0.0.0'
APP_PORT = 8080