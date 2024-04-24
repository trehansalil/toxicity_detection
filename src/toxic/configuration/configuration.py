from src.toxic.constants import *
from src.toxic.utils.common import read_yaml, create_directories
from src.toxic.entity.config_entity import (DataIngestionConfig,
                                        DataTransformationConfig,
                                        TrainingConfig,
                                        EvaluationConfig)

import numpy as np
import pandas as pd
import os
import random
import time

import re
import string
import nltk
from nltk.corpus import stopwords

import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="ticks", context="talk")
plt.style.use('dark_background')

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as func
from torch.utils.data import DataLoader, Dataset

import mlflow.pytorch
from mlflow import MlflowClient
from mlflow.models import infer_signature

import transformers
from transformers import AdamW, get_linear_schedule_with_warmup

import tokenizers
from sklearn.metrics import mean_squared_error, roc_auc_score, roc_curve, auc

import warnings
warnings.simplefilter('ignore')


class ConfigurationManager:
    def __init__(
        self,
        config_filepath = CONFIG_FILE_PATH,
        params_filepath = PARAMS_FILE_PATH):

        self.config = read_yaml(config_filepath)
        self.params = read_yaml(params_filepath)

        create_directories([self.config.artifacts_root])


    
    def get_data_ingestion_config(self) -> DataIngestionConfig:
        config = self.config.data_ingestion

        create_directories([config.root_dir])

        data_ingestion_config = DataIngestionConfig(
            root_dir=config.root_dir,
            source_url=config.source_url,
            unzip_dir=config.unzip_dir,
            train_file=config.train_file,
            test_file=config.test_file,
            labels_file=config.labels_file,
            sample_sub_file=config.sample_sub_file,
            bert_uncased=config.bert_uncased
        )

        return data_ingestion_config
    

    def get_data_transformation_config(self) -> DataTransformationConfig:
        config = self.config.data_transformation
        
        create_directories([config.root_dir])

        data_transformation_config = DataTransformationConfig(
            root_dir=Path(config.root_dir),
            base_model_path=Path(config.base_model_path),
            updated_base_model_path=Path(config.updated_base_model_path),
            params_batch_size=self.params.BATCH_SIZE,
            params_epochs=self.params.EPOCHS,
            params_num_workers=self.params.NUM_WORKERS,
            params_pin_memory=self.params.PIN_MEMORY,
            params_fold=self.params.FOLD,
            params_train_subset=self.params.TRAIN_SUBSET,
            params_test_subset=self.params.TEST_SUBSET,
            params_seed=self.params.SEED,
            params_max_len=self.params.MAX_LEN,
            params_pre_trained_model=self.params.PRE_TRAINED
        )

        return data_transformation_config
    


    def get_training_config(self) -> TrainingConfig:
        training = self.config.training
        create_directories([
            Path(training.root_dir)
        ])

        training_config = TrainingConfig(
            root_dir=Path(training.root_dir),
            trained_model_path=Path(training.trained_model_path),
            dagshub_mlflow_remote_uri=training.dagshub_mlflow_remote_uri,
            params_batch_size=self.params.BATCH_SIZE,
            params_epochs=int(self.params.EPOCHS),
            params_classes=self.params.CLASSES,
            params_num_workers=self.params.NUM_WORKERS,
            params_learning_rate=float(self.params.LEARNING_RATE),
            params_weight_decay=float(self.params.WEIGHT_DECAY),
            params_beta1=self.params.BETA1,
            params_beta2=self.params.BETA2,
            params_pin_memory=self.params.PIN_MEMORY,
            params_fold=self.params.FOLD,
            params_max_len=self.params.MAX_LEN,
            params_train_subset=self.params.TRAIN_SUBSET,
            params_test_subset=self.params.TEST_SUBSET,
            params_seed=self.params.SEED,
            params_pre_trained_model=self.params.PRE_TRAINED
        )

        return training_config
    



    def get_evaluation_config(self) -> EvaluationConfig:
        eval_config = EvaluationConfig(
            path_of_model="artifacts/training/best_model.pt",
            training_data="artifacts/data_ingestion/toxicity_detection",
            mlflow_uri="https://dagshub.com/trehansalil/toxicity_detection.mlflow",
            all_params=self.params,
            params_pre_trained_model=self.params.PRE_TRAINED,
            params_batch_size=self.params.BATCH_SIZE
        )
        return eval_config

      