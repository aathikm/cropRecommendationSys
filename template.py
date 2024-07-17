import os 
import sys
from pathlib import Path

list_of_files = [
    ".gitignore",
    "experiments/experiment.ipynb",
    "src/components/data_ingestion.py",
    "src/components/data_preprocessing.py",
    "src/components/model_training.py",
    "src/components/model_evaluation.py",
    "src/components/__init__.py",
    "src/pipelines/training_pipeline.py",
    "src/pipelines/prediction_pipeline.py",
    "src/pipelines/__init__.py",
    "src/utils/__init__.py",
    "src/utils/utils.py",
    "src/exception/__init__.py",
    "src/exception/exception.py",
    "src/loggingInfo/__init__.py",
    "src/loggingInfo/loggingFile.py",
    "src/__init__.py",
    "templates/index.html",
    "templates/prediction.html",
    "templates/home.html",
    "app.py",
    "requirements.txt",
    "main.py",
    "Dockerfile",
    "setup.py"
]

for file in list_of_files:
    fileFullPath = Path(file)
    fileExt, fileName = os.path.split(fileFullPath)
    
    if fileExt != "":
        os.makedirs(fileExt, exist_ok=True)
    
    if not (os.path.exists(fileFullPath)):
        
        with open(fileFullPath, "wb") as f:
            pass
    