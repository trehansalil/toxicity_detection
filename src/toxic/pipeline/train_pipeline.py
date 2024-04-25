import inspect
import sys
from src.toxic import logging
from src.toxic.components.data_ingestion import DataIngestion
from src.toxic.components.data_transformation import DataTransformation

from src.toxic.components.model_trainer import ModelTrainer

from src.toxic.configuration.configuration import ConfigurationManager

class TrainingPipeline:
    def __init__(self):
        config = ConfigurationManager()        
        self.data_ingestion_config = config.get_data_ingestion_config()
        self.data_transformation_config = config.get_data_transformation_config()
        self.model_trainer_config = config.get_training_config()
        self.model_evaluation_config = config.get_evaluation_config()
        
    def start_data_ingestion(self):
        current_function_name = inspect.stack()[0][3]
        try:
            logging.info(f"Getting the data from GCloud Storage bucket using the {current_function_name} method of {self.__class__.__name__} class")
            data_ingestion = DataIngestion(data_ingestion_config = self.data_ingestion_config)
            
            data_ingestion_artifacts = data_ingestion.initiate_data_ingestion()
            logging.info(f"Got the train and validation data from GCloud Storage using the {current_function_name} method of {self.__class__.__name__} class")
            logging.info(f"Exited the {current_function_name} method of {self.__class__.__name__} class")
            
        except Exception as e:
            raise e     
        
    def start_data_transformation(self):
        current_function_name = inspect.stack()[0][3]
        try:
            logging.info(f"Starting Transformation of data using the {current_function_name} method of {self.__class__.__name__} class")
            
            data_transformation = DataTransformation(
                config = self.data_transformation_config, 
                data_config = self.data_ingestion_config               
            )
            
            train_dataloader_list, valid_dataloader_list, test_dataloader = data_transformation.initiate_data_transformation()
            logging.info(f"Data Transformation done using the {current_function_name} method of {self.__class__.__name__} class")
            return train_dataloader_list, valid_dataloader_list, test_dataloader
        except Exception as e:
            raise e    

    def start_model_trainer(
        self, 
        train_dataloader_list, 
        valid_dataloader_list
    ):
        current_function_name = inspect.stack()[0][3]
        try:
            logging.info(f"Starting Model Training using the {current_function_name} method of {self.__class__.__name__} class")
            
            model_trainer = ModelTrainer(
                train_dataloader_list = train_dataloader_list, 
                validation_dataloader_list = valid_dataloader_list, 
                model_trainer_config = self.model_trainer_config,
                data_config = self.data_ingestion_config
            )
            
            best_model_path = model_trainer.initiate_model_trainer()
            logging.info(f"Model Training done using the {current_function_name} method of {self.__class__.__name__} class")
            return best_model_path
        except Exception as e:
            raise e                    
        
    # def start_model_evaluation(self, best_model_path: str):
        
    #     current_function_name = inspect.stack()[0][3]
    #     logging.info(f"Starting Model Evaluation using the {current_function_name} method of {self.__class__.__name__} class")
            
    #     try:
    #         model_evaluation = ModelEvaluation(data_transformation_artifacts = data_transformation_artifacts,
    #                                             model_evaluation_config=self.model_evaluation_config,
    #                                             model_trainer_artifacts=model_trainer_artifacts)

    #         model_evaluation_artifacts = model_evaluation.initiate_model_evaluation()
    #         logging.info(f"Model Evaluation done using the {current_function_name} method of {self.__class__.__name__} class")
    #         return model_evaluation_artifacts

    #     except Exception as e:
    #         raise CustomException(e, sys) from e
        
    

    # def start_model_pusher(self,) -> ModelPusherArtifacts:
    #     current_function_name = inspect.stack()[0][3]
    #     logging.info(f"Starting Model Evaluation using the {current_function_name} method of {self.__class__.__name__} class")
    #     try:
    #         model_pusher = ModelPusher(
    #             model_pusher_config=self.model_pusher_config,
    #         )
    #         model_pusher_artifact = model_pusher.initiate_model_pusher()
    #         logging.info("Initiated the model pusher")
    #         logging.info(f"Model Evaluation done using the {current_function_name} method of {self.__class__.__name__} class")
    #         return model_pusher_artifact

    #     except Exception as e:
    #         raise CustomException(e, sys) from e  
                     
    def run_pipeline(self):
        current_function_name = inspect.stack()[0][3]
        logging.info(f"Started the {current_function_name} method of {self.__class__.__name__} class")
        
        try:
            logging.info(f"Starting Ingestion using the {current_function_name} method of {self.__class__.__name__} class")
            self.start_data_ingestion()

            logging.info(f"Starting Transformation using the {current_function_name} method of {self.__class__.__name__} class")            
            train_dataloader_list, valid_dataloader_list, test_dataloader = self.start_data_transformation()
            
            logging.info(f"Starting Model Training using the {current_function_name} method of {self.__class__.__name__} class")            
            best_model_path = self.start_model_trainer(train_dataloader_list, valid_dataloader_list)

            # model_evaluation_artifacts = self.start_model_evaluation(best_model_path=best_model_path, test_dataloader=test_dataloader) 

            # if not model_evaluation_artifacts.is_model_accepted:
            #     raise Exception("Trained model is not better than the best model")
            
            # model_pusher_artifacts = self.start_model_pusher()    
            logging.info(f"Exited the {current_function_name} method of {self.__class__.__name__} class")
        except Exception as e:
            raise e           
 
if __name__ == "__main__":
      training_pip = TrainingPipeline()
      training_pip.run_pipeline()