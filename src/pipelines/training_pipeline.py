import os
import pandas as pd
import numpy as np
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, "..")
sys.path.append(src_dir)

from exception.exception import customException
from loggingInfo.loggingFile import logging

from components.data_ingestion import DataIngestion
from components.data_preprocessing import DataTransformation
from components.model_training import ModelTraining
# from components.model_evaluation import ModelEvaluation

class TrainingPipeline:
    def __init__(self) -> None:
        pass
    
    def start_data_ingestion(self):
        try:
            data_ing_obj = DataIngestion()
            train_path, test_path = data_ing_obj.initiate_data_ingestion()
            logging.info("data paths successfully retrived from data_ingestion component")
            
            return(train_path, test_path)
        except Exception as e:
           raise customException(e, sys)
            
    def start_data_transformation(slef, train_path:str, test_path:str):
        try:
            data_trans_obj = DataTransformation()
            training_arr, testing_arr = data_trans_obj.process_data_transformation(train_path=train_path, test_path=test_path)
            logging.info("training array and testing array are generated")
            
            return(training_arr, testing_arr)
        except Exception as e:
            raise customException(e, sys)
            
    def start_model_training(self, train_arr, test_arr):
        try:
            model_trainer_obj = ModelTraining()
            model_trainer_obj.initialize_model_training(train_arr=train_arr, test_arr=test_arr)
            logging.info("model created")
        except Exception as e:
            raise customException(e, sys)
            
    def start_training_pipeline(self):
        try:
            train_path, test_path = self.start_data_ingestion()
            train_arr, test_arr = self.start_data_transformation(train_path, test_path)
            self.start_model_training(train_arr, test_arr)
            logging.info("training pipeline successfully ran")
        except Exception as e:
            customException(e, sys)

if __name__ == "__main__":
    obj = TrainingPipeline()
    obj.start_training_pipeline()