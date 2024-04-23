from dataclasses import dataclass
from src.toxic.constants import *
import os

@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    source_url: str
    unzip_dir: Path
    train_file: str
    test_file: str
    labels_file: str
    sample_sub_file: str
    bert_uncased: Path


@dataclass(frozen=True)
class DataTransformationConfig:
    root_dir: Path
    base_model_path: Path
    updated_base_model_path: Path
    params_batch_size: list
    params_epochs: int
    params_num_workers: int
    params_pin_memory: bool
    params_fold: int
    params_train_subset: int
    params_test_subset: int
    params_seed: int
    params_max_len: int
    params_pre_trained_model: str
    




@dataclass(frozen=True)
class TrainingConfig:
    root_dir: Path
    trained_model_path: Path
    dagshub_mlflow_remote_uri: str
    params_batch_size: list
    params_epochs: int
    params_classes: int
    params_num_workers: int
    params_learning_rate: float
    params_weight_decay: float
    params_beta1: float
    params_beta2: float
    params_pin_memory: bool
    params_fold: int
    params_max_len: int
    params_train_subset: int
    params_test_subset: int
    params_seed: int
    params_pre_trained_model: str



@dataclass(frozen=True)
class EvaluationConfig:
    path_of_model: Path
    training_data: Path
    all_params: dict
    mlflow_uri: str
    params_pre_trained_model: str
    params_batch_size: int