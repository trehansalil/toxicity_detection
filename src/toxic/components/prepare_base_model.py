import os
import urllib.request as request
from zipfile import ZipFile
import tensorflow as tf
from pathlib import Path
from src.toxic.entity.config_entity import PrepareBaseModelConfig, DataIngestionConfig
import pandas as pd                                                




class PrepareBaseModel:
    def __init__(self, config: PrepareBaseModelConfig, 
                 data_config: DataIngestionConfig):
        self.config = config
        self.data_config = data_config
    
    def read_data(self):
        train_path = os.path.join(self.data_config.unzip_dir, 
                                  self.data_config.train_file)
        labels_path = os.path.join(self.data_config.unzip_dir, 
                                  self.data_config.labels_file)
        test_path = os.path.join(self.data_config.unzip_dir, 
                                  self.data_config.test_file)
        submission_path = os.path.join(self.data_config.unzip_dir, 
                                  self.data_config.sample_sub_file)   
        
        train = pd.read_csv(train_path, nrows = 2000)
        test = pd.read_csv(test_path, nrows = 100)
        submission = pd.read_csv(submission_path)                             

    
    def get_base_model(self):
        self.model = tf.keras.applications.vgg16.VGG16(
            input_shape=self.config.params_image_size,
            weights=self.config.params_weights,
            include_top=self.config.params_include_top
        )

        self.save_model(path=self.config.base_model_path, model=self.model)


    
    @staticmethod
    def _prepare_full_model(model, classes, freeze_all, freeze_till, learning_rate):
        if freeze_all:
            for layer in model.layers:
                model.trainable = False
        elif (freeze_till is not None) and (freeze_till > 0):
            for layer in model.layers[:-freeze_till]:
                model.trainable = False

        flatten_in = tf.keras.layers.Flatten()(model.output)
        prediction = tf.keras.layers.Dense(
            units=classes,
            activation="softmax"
        )(flatten_in)

        full_model = tf.keras.models.Model(
            inputs=model.input,
            outputs=prediction
        )

        full_model.compile(
            optimizer=tf.keras.optimizers.SGD(learning_rate=learning_rate),
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=["accuracy"]
        )

        full_model.summary()
        return full_model
    

    def update_base_model(self):
        self.full_model = self._prepare_full_model(
            model=self.model,
            classes=self.config.params_classes,
            freeze_all=True,
            freeze_till=None,
            learning_rate=self.config.params_learning_rate
        )

        self.save_model(path=self.config.updated_base_model_path, model=self.full_model)
    


    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)