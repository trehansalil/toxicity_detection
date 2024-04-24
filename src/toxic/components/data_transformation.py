import os
import random
import re
import string
import inspect
import numpy as np

import torch
import transformers
from src.toxic.entity.config_entity import DataTransformationConfig, DataIngestionConfig
from src.toxic.configuration.bert_data import BertDataSet
from src.toxic import logging
from torch.utils.data import DataLoader
import pandas as pd                                                


class DataTransformation:
    def __init__(
        self, 
        config: DataTransformationConfig, 
        data_config: DataIngestionConfig
    ):
        self.config = config
        self.data_config = data_config
        self.max_len = self.config.params_max_len  
        self.tokenizer = transformers.BertTokenizer.from_pretrained(
            self.config.params_pre_trained_model
        )         
           
    def random_seed(self, SEED):
        random.seed(SEED)
        os.environ['PYTHONHASHSEED'] = str(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        torch.cuda.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.deterministic = True  
    
    def clean_text(self, text):

        text = re.sub('\[.*?\]', '', text)
        text = re.sub('https?://\S+|www\.\S+', '', text)
        text = re.sub('<.*?>+', '', text)
        text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
        text = re.sub('\n', '', text)
        text = re.sub('\w*\d\w*', '', text)
        return text          
    
    def read_data(self):
        current_function_name = inspect.stack()[0][3]
        logging.info(f"Entered the {current_function_name} method of {self.__class__.__name__} class")        
        try:
            self.random_seed(self.config.params_seed)
            
            self.train_path = os.path.join(self.data_config.unzip_dir, 
                                    self.data_config.train_file)
            self.labels_path = os.path.join(self.data_config.unzip_dir, 
                                    self.data_config.labels_file)
            self.test_path = os.path.join(self.data_config.unzip_dir, 
                                    self.data_config.test_file)
            self.submission_path = os.path.join(self.data_config.unzip_dir, 
                                    self.data_config.sample_sub_file)         
            
            self.train = pd.read_csv(self.train_path, nrows = self.config.params_train_subset)
            self.test = pd.read_csv(self.train_path, nrows = self.config.params_test_subset)
            self.test = self.test.iloc[self.config.params_train_subset:self.config.params_train_subset+self.config.params_test_subset, :]
            self.submission = pd.read_csv(self.submission_path)   
            # test_labels = pd.read_csv(self.labels_path, nrows = self.config.params_test_subset)    
            
            self.train['clean_text'] = self.train['comment_text'].apply(str).apply(lambda x: self.clean_text(x))
            self.test['clean_text'] = self.test['comment_text'].apply(str).apply(lambda x: self.clean_text(x)) 

            self.train['kfold'] = self.train.index % self.config.params_fold 
            
            test_dataset = BertDataSet(
                self.test['clean_text'], 
                self.test[['toxic', 'severe_toxic','obscene', 'threat', 'insult','identity_hate']], 
                tokenizer=self.tokenizer, 
                max_len=self.max_len
            ) 
            
            self.test_dataloader = DataLoader(
                test_dataset, 
                batch_size = self.config.params_batch_size, 
                pin_memory = self.config.params_pin_memory, 
                num_workers = self.config.params_num_workers, 
                shuffle = False
            )
            
            logging.info(f"Exited the {current_function_name} method of {self.__class__.__name__} class") 
        except Exception as e:
            raise e         
    
    def process_data(self, fold):
        current_function_name = inspect.stack()[0][3]
        logging.info(f"Entered the {current_function_name} method of {self.__class__.__name__} class")      
        try:
            
            self.p_train = self.train[self.train["kfold"] != fold].reset_index(drop = True)
            self.p_valid = self.train[self.train["kfold"] == fold].reset_index(drop = True)
            
            self.train_steps = int(len(self.p_train)/self.config.params_batch_size * self.config.params_epochs)
            self.num_steps = int(self.train_steps * 0.1)                   
            
            train_dataset = BertDataSet(
                self.p_train['clean_text'], 
                self.p_train[['toxic', 'severe_toxic','obscene', 'threat', 'insult','identity_hate']], 
                tokenizer=self.tokenizer, 
                max_len=self.max_len
            )
            valid_dataset = BertDataSet(
                self.p_valid['clean_text'], 
                self.p_valid[['toxic', 'severe_toxic','obscene', 'threat', 'insult','identity_hate']], 
                tokenizer=self.tokenizer, 
                max_len=self.max_len
            )   
            
            train_dataloader = DataLoader(
                train_dataset, 
                batch_size = self.config.params_batch_size, 
                pin_memory = self.config.params_pin_memory, 
                num_workers = self.config.params_num_workers, 
                shuffle = True
            )

            valid_dataloader = DataLoader(
                valid_dataset, 
                batch_size = self.config.params_batch_size, 
                pin_memory = self.config.params_pin_memory, 
                num_workers = self.config.params_num_workers, 
                shuffle = False
            )  
            logging.info(f"Exited the {current_function_name} method of {self.__class__.__name__} class")
            
            return train_dataloader, valid_dataloader
        
        except Exception as e:
            raise e         
        
    def initiate_data_transformation(self):
        current_function_name = inspect.stack()[0][3]
        logging.info(f"Entered the {current_function_name} method of {self.__class__.__name__} class")  
        try:
            self.read_data()
            train_dataloader_list = []
            valid_dataloader_list = []
            
            for fold in range(self.config.params_fold):
                train_dataloader, valid_dataloader = self.process_data(fold=fold)
                train_dataloader_list.append(train_dataloader)
                valid_dataloader_list.append(valid_dataloader)
                
            logging.info(f"Exited the {current_function_name} method of {self.__class__.__name__} class")               
                
            return train_dataloader_list, valid_dataloader_list, self.test_dataloader
        except Exception as e:
            raise e