import copy
import os
import random
import inspect
from urllib.parse import urlparse
import dagshub
import mlflow
import numpy as np
import inspect
import torch
from tqdm import tqdm
from src.toxic import logging
from src.toxic.entity.config_entity import TrainingConfig, DataIngestionConfig
from src.toxic.constants import *
from tqdm import tqdm
import transformers

import torch.nn as nn
import torch
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader
from typing import List

class ModelTrainer:
    def __init__(
            self, 
            data_config: DataIngestionConfig,
            model_trainer_config: TrainingConfig, 
            train_dataloader_list: List[DataLoader], 
            validation_dataloader_list: List[DataLoader],
            train_steps: int,
            num_steps: int
        ):
        self.config = model_trainer_config
        self.data_config = data_config
        
        self.best_score = 1000
        self.remote_server_uri = self.config.dagshub_mlflow_remote_uri
        os.environ['MLFLOW_TRACKING_URI'] = self.remote_server_uri
        os.environ['MLFLOW_TRACKING_PASSWORD'] = '88855b61c077c3a7538eda58ac1a8a33eb4d1098'
        os.environ['MLFLOW_TRACKING_USERNAME'] = 'trehansalil'
        
        self.max_len = self.config.params_max_len               
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # dagshub.init(
        #     repo_owner='trehansalil', 
        #     repo_name='toxicity_detection', 
        #     mlflow=True
        # )       
        
        self.train_dataloader_list = train_dataloader_list 
        self.validation_dataloader_list = validation_dataloader_list 
        
        self.train_steps = train_steps
        self.num_steps = num_steps 
        
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.loss_fn.to(self.device)      
        self.scaler = torch.cuda.amp.GradScaler()  
        
        self.best_model_path = os.path.join(self.config.root_dir, "model.pth")
        
    def random_seed(self, SEED):
        random.seed(SEED)
        os.environ['PYTHONHASHSEED'] = str(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        torch.cuda.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.deterministic = True  
    
        
    def training(self, train_dataloader):
        current_function_name = inspect.stack()[0][3]
        logging.info(f"Entered the {current_function_name} method of {self.__class__.__name__} class")        
        try:
            self.model.train()
            torch.backends.cudnn.benchmark = True
            correct_predictions = 0

            for a in train_dataloader:
                losses = []
                self.optimizer.zero_grad()

                #allpreds = []
                #alltargets = []

                with torch.cuda.amp.autocast():

                    ids = a['ids'].to(self.device, non_blocking = True)
                    mask = a['mask'].to(self.device, non_blocking = True)

                    output = self.model(ids, mask) #This gives model as output, however we want the values at the output
                    output = output['logits'].squeeze(-1).to(torch.float32)

                    output_probs = torch.sigmoid(output)
                    preds = torch.where(output_probs > 0.5, 1, 0)

                    toxic_label = a['toxic_label'].to(self.device, non_blocking = True)
                    loss = self.loss_fn(output, toxic_label)

                    losses.append(loss.item())
                    #allpreds.append(output.detach().cpu().numpy())
                    #alltargets.append(toxic.detach().squeeze(-1).cpu().numpy())
                    correct_predictions += torch.sum(preds == toxic_label)

                self.scaler.scale(loss).backward() #Multiplies (‘scales’) a tensor or list of tensors by the scale factor.
                                            #Returns scaled outputs. If this instance of GradScaler is not enabled, outputs are returned unmodified.
                self.scaler.step(self.optimizer) #Returns the return value of optimizer.step(*args, **kwargs).
                self.scaler.update() #Updates the scale factor.If any optimizer steps were skipped the scale is multiplied by backoff_factor to reduce it.
                                #If growth_interval unskipped iterations occurred consecutively, the scale is multiplied by growth_factor to increase it
                self.scheduler.step() # Update learning rate schedule

            losses = np.mean(losses)
            corr_preds = correct_predictions.detach().cpu().numpy()
            accuracy = corr_preds/(len(train_dataloader)*self.config.params_classes)

            logging.info(f"Exited the {current_function_name} method of {self.__class__.__name__} class")
            
            return losses, accuracy    
            
        except Exception as e:
            raise e     

    def validating(self, valid_dataloader):
        current_function_name = inspect.stack()[0][3]
        logging.info(f"Entered the {current_function_name} method of {self.__class__.__name__} class")        
        try:
            self.model.eval()
            correct_predictions = 0
            all_output_probs = []

            for a in valid_dataloader:
                losses = []
                ids = a['ids'].to(self.device, non_blocking = True)
                mask = a['mask'].to(self.device, non_blocking = True)
                output = self.model(ids, mask)
                output = output['logits'].squeeze(-1).to(torch.float32)
                output_probs = torch.sigmoid(output)
                preds = torch.where(output_probs > 0.5, 1, 0)

                toxic_label = a['toxic_label'].to(self.device, non_blocking = True)
                loss = self.loss_fn(output, toxic_label)
                losses.append(loss.item())
                all_output_probs.extend(output_probs.detach().cpu().numpy())

                correct_predictions += torch.sum(preds == toxic_label)
                corr_preds = correct_predictions.detach().cpu().numpy()

            losses = np.mean(losses)
            corr_preds = correct_predictions.detach().cpu().numpy()
            accuracy = corr_preds/(len(valid_dataloader)*self.config.params_classes)
            
            logging.info(f"Exited the {current_function_name} method of {self.__class__.__name__} class")
            
            return losses, accuracy, all_output_probs    
            
        except Exception as e:
            raise e        

    def get_model(self):
        current_function_name = inspect.stack()[0][3]
        logging.info(f"Entered the {current_function_name} method of {self.__class__.__name__} class")        
        try:
            self.tokenizer = transformers.BertTokenizer.from_pretrained(
                self.config.params_pre_trained_model
            ) 
                    
            self.model = transformers.BertForSequenceClassification.from_pretrained(
                self.data_config.bert_uncased, 
                num_labels = self.config.params_classes
            )
            
            self.model.to(self.device)
            # self.model.train()     
            self.optimizer = AdamW(
                self.model.parameters(), 
                self.config.params_learning_rate,
                betas = (self.config.params_beta1, self.config.params_beta2), 
                weight_decay = self.config.params_learning_rate
            )   
            
            self.scheduler = get_linear_schedule_with_warmup(
                self.optimizer, 
                self.num_steps, 
                self.train_steps
            )
            logging.info(f"Exited the {current_function_name} method of {self.__class__.__name__} class")
        except Exception as e:
            raise e        
    
    def update_model(self):
        
        current_function_name = inspect.stack()[0][3]
        logging.info(f"Entered the {current_function_name} method of {self.__class__.__name__} class")    
        try:     
            best_results = []        
            best_scores = []
            
            pre_training_best_score = copy.copy(self.best_score)
            
            logging.info("Started mlflow Experiment Tracking")
            
            mlflow.autolog()
            
            with mlflow.start_run():
                
                
                mlflow.log_param('fold', self.config.params_fold)
                mlflow.log_param('num_workers', self.config.params_num_workers)
                mlflow.log_param('pin_memory', self.config.params_pin_memory)
                mlflow.log_param('beta1', self.config.params_beta1)
                mlflow.log_param('beta2', self.config.params_beta2)
                mlflow.log_param('weight_decay', self.config.params_weight_decay)
                mlflow.log_param('best_score', self.best_score)
                mlflow.log_param('Learning_rate', self.config.params_learning_rate)                   
            
                for fold in tqdm(range(0,self.config.params_fold)):

                    self.get_model()

                    best_valid_probs = []

                    print("-------------- Fold = " + str(fold) + "-------------")

                    for epoch in tqdm(range(self.config.params_epochs)):
                        print("-------------- Epoch = " + str(epoch) + "-------------")

                        train_loss, train_acc = self.training(self.train_dataloader_list[fold])
                        valid_loss, valid_acc, valid_probs = self.validating(self.validation_dataloader_list[fold])


                        print('train losses: %.4f' %(train_loss), 'train accuracy: %.3f' %(train_acc))
                        print('valid losses: %.4f' %(valid_loss), 'valid accuracy: %.3f' %(valid_acc))

                        if (valid_loss < self.best_score):

                            self.best_score = valid_loss
                            print("Found an improved model! :)")

                            state = {
                                'state_dict': self.model.state_dict(),
                                'optimizer_dict': self.optimizer.state_dict(),
                                'best_score': self.best_score
                            }
                            
                            best_results.append(
                                [
                                    train_loss, 
                                    train_acc, 
                                    valid_loss, 
                                    valid_acc
                                ]
                            )
                            logging.info("saving the model")
                            self.save_model(state, path=self.best_model_path)

                            best_valid_prob = valid_probs
                            # torch.cuda.memory_summary(device = None, abbreviated = False)
                        else:
                            pass


                    best_scores.append(self.best_score)
                    best_valid_probs.append(best_valid_prob)
                    
                mlflow.set_tracking_uri(self.remote_server_uri)
                tracking_url_type_store = urlparse(mlflow.get_tracking_uri()).scheme

                mlflow.log_param('train_steps', self.train_steps)
                mlflow.log_param('num_steps', self.num_steps)         
                mlflow.log_param('epoch', self.config.params_epochs)
                
                if self.best_score < pre_training_best_score:
                    mlflow.log_metric('train_loss', best_results[-1][0])   
                    mlflow.log_metric('train_acc', best_results[-1][1]) 

                    mlflow.log_metric('valid_loss', best_results[-1][2])   
                    mlflow.log_metric('valid_acc', best_results[-1][3])
                
                # Model registry does not work with file store
                if tracking_url_type_store != "file":
                    # Register the model
                    # There are other ways to use the Model Registry, which depends on the use case,
                    # please refer to the doc for more information:
                    # https://mlflow.org/docs/latest/model-registry.html#api-workflow
                    mlflow.pytorch.log_model(
                        state, "model", registered_model_name="ElasticnetWineModel"
                    )
                else:
                    mlflow.pytorch.log_model(state, "model")                        

            logging.info("Ended mlflow Experiment Tracking")
            
            logging.info(f"Exited the {current_function_name} method of {self.__class__.__name__} class")
        except Exception as e:
            raise e  
               
    def initiate_model_trainer(self,):
        current_function_name = inspect.stack()[0][3]
        logging.info(f"Entered the {current_function_name} method of {self.__class__.__name__} class")
        
        """
        Method Name :   initiate_model_trainer
        Description :   This function initiates a model trainer steps
        
        On Failure  :   Write an exception log and then raise an exception
        """
        self.random_seed(self.config.params_seed)
        try:
            
            logging.info("Entered into model training")
            self.update_model()
            logging.info("Model training finished")

            
            logging.info(f"Exited the {current_function_name} method of {self.__class__.__name__} class")

            return self.best_model_path

        except Exception as e:
            raise e
        
    @staticmethod
    def save_model(model_dict, path: Path):
        torch.save(model_dict, path)          