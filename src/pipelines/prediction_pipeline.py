import os
import pandas as pd
import numpy as np
import sys
import pickle

current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, "..")
sys.path.append(src_dir)

from exception.exception import customException
from loggingInfo.loggingFile import logging
from utils.utils import load_object


class PredictionPipeline:
    def __init__(self) -> None:
        pass
    
    def predict(self, features):
        try:
            logging.info("training pipeline started in prediction pipeline")
            logging.info("training pipeline sucessfully ran")
            
            model_path = os.path.join("artifacts", "model.pkl")
            preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")
            encoded_df_path = os.path.join("artifacts", "encoded_df.csv")
            
            logging.info(f"data: {features}")
            model= load_object(model_path)
            preprocessor = load_object(preprocessor_path)
            
            data_transformation = preprocessor.transform(features)
            logging.info(f"preprocessed value: {data_transformation}")
            predicted_val = model.predict(data_transformation)
            
            label_encoded_df = pd.read_csv(encoded_df_path, encoding='unicode_escape')
            logging.info(f"encoded value df: {label_encoded_df}")
            logging.info(f"predicted value: {predicted_val}")
            prediction_value = label_encoded_df[label_encoded_df['encoded_val'] == predicted_val[0]] #["label"]
            return(prediction_value["label"].values[0])
            # return prediction_value["label"]
            
        except Exception as e:
            logging.info("Error occured in prediction pipeline")
            raise customException(e, sys)

class GetCustomData:
    def __init__(self,
                 N: float,
                 P: float,
                 K: float,
                 temperature: float,
                 humidity: float,
                 ph: float,
                 rainfall: float
                 ) -> None:
        
        self.N = N
        self.P = P
        self.K = K
        self.temperature = temperature
        self.humidity = humidity
        self.ph = ph
        self.rainfall = rainfall
    
    def get_data(self):
        try:
            custom_data = {
                'N': [self.N],
                'P': [self.P],
                'K': [self.K],
                'temperature': [self.temperature],
                'humidity': [self.humidity],
                'ph': [self.ph],
                'rainfall': [self.rainfall]
            }
            
            df = pd.DataFrame(custom_data)
            logging.info(f"custom data converted as dataframe. And the data is, {df}")
            return(df)
        except Exception as e:
            logging.info("Error occured in prediction pipeline data preparation")
            raise customException(e, sys)

if __name__ == "__main__":
    obj1 = GetCustomData(25,28,32,21.77046169,80.31964408,7.038096361,226.6555374)
    data = obj1.get_data()
    
    obj2 = PredictionPipeline()
    res = obj2.predict(data)
    print(res)
    