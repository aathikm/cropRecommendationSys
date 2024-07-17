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

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import mlflow
import mlflow.sklearn
from urllib.parse import urlparse

# @dataclass
# class ModelEvaluationConfig:
#     pass

class ModelEvaluation:
    def __init__(self):
        logging.info("model evaluation was started...")
    
    def evaluation_matrix(self, actual:list, pred:list):
        """Evaluation matrics

        Args:
            actual (list): model actual label value
            pred (list): model predicted label value
        """
        rmse_val = np.sqrt(mean_squared_error(actual, pred))
        r2_val = r2_score(actual, pred)
        mae_val = mean_absolute_error(actual, pred)
        logging.info("evaluation metrics cerated and captured")
        return(r2_val, rmse_val, mae_val)
    
    def initialize_model_evaluation(self, trian_arr, test_arr):
        try:
            logging.info("model evaluation started")
            x_test, y_test = (test_arr[:, :-1], test_arr[:, -1])
            
            model_path = os.path.join("artifacts", "model.pkl")
            model = load_object(model_path)
            
            logging.info("model sucessfully registered")
            
            tracking_url_type = urlparse(mlflow.get_tracking_uri()).scheme
            logging.info(f"tracking url type: {tracking_url_type}")
            
            with mlflow.start_run():
                
                pred = model.predict(x_test)
                
                (r2_val, rmse_val, mae_val) = self.evaluation_matrix(y_test, pred)
                
                mlflow.log_param("r2_val", r2_val)
                mlflow.log_param("rmse_val", rmse_val)
                mlflow.log_param("mae_val", mae_val)
                
                if (tracking_url_type != 'file'):
                    mlflow.sklearn.log_model(model, "model", registered_model_name="ml_model")
                else:
                    mlflow.sklearn.log_model(model, "model")
        
        except Exception as e:
            logging.info("exception happened at model evaluation code block")
            customException(e, sys)
            
            
if __name__ == "__main__":
    obj = ModelEvaluation()
    
    train_arr_path, test_arr_path = os.path.join("artifacts", "train_arr.pkl"), os.path.join("artifacts", "test_arr.pkl")
    with open(train_arr_path, "rb") as f:
        train_arr = pickle.load(f)
        
    with open(test_arr_path, "rb") as f:
        test_arr = pickle.load(f)
            
    
    obj.initialize_model_evaluation(trian_arr=train_arr, test_arr=test_arr)