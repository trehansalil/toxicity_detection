from dataclasses import dataclass

@dataclass
class DataIngestionArtifacts:
    imbalance_data_file_path: str
    raw_data_file_path: str

@dataclass    
class DataValidationArtifacts:    
    imbalance_data_valid: bool
    raw_data_valid: bool
    
@dataclass    
class DataTransformationArtifacts:    
    transformation_data_file_path: str  
    
@dataclass    
class ModelTrainerArtifacts:    
    trained_model_path: str
    x_test_path: list
    y_test_path: list 
    
@dataclass
class ModelEvaluationArtifacts:
    is_model_accepted: bool 

@dataclass
class ModelPusherArtifacts:
    bucket_name: str    