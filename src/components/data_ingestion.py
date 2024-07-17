import os
import pandas as pd
import numpy as np
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, "..")
sys.path.append(src_dir)

from exception.exception import customException
from loggingInfo.loggingFile import logging
from sklearn.model_selection import train_test_split

from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    rawDataPath:str = os.path.join("artifacts", "raw.csv")
    trainDataPath:str = os.path.join("artifacts", "train.csv")
    testDataPath:str = os.path.join("artifacts", "test.csv")

class DataIngestion:
    def __init__(self):
        self.ingestionConfig = DataIngestionConfig()
        
    def initiate_data_ingestion(self):
        logging.info("Data ingestion stated")
        
        try:
            data = pd.read_csv("https://raw.githubusercontent.com/aathikm/datasets/main/Crop_recommendation.csv")
            
            os.makedirs(os.path.dirname(os.path.join(self.ingestionConfig.rawDataPath)), exist_ok=True)
            data.to_csv(self.ingestionConfig.rawDataPath, index=False)
            logging.info("Raw data successfully stored in artifacts folder")
            
            train_data, test_data = train_test_split(data, test_size=0.2, random_state=101)
            logging.info("train and test data is splitted")
            
            train_data.to_csv(self.ingestionConfig.trainDataPath, index = False)
            test_data.to_csv(self.ingestionConfig.testDataPath, index = False)
            
            logging.info("Data ingestion completed")
            return(
                self.ingestionConfig.trainDataPath,
                self.ingestionConfig.testDataPath
            )
        
        except Exception as e:
            logging.warning("error")
            raise customException(e, sys)
        
if __name__ == "__main__":
    obj = DataIngestion()
    obj.initiate_data_ingestion()