import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s:')

project_name = "spamDetection"
list_of_files = [
    f"src/{project_name}/__init__.py",
    f"src/{project_name}/components/__init__.py",
    f"src/{project_name}/components/data_ingestion.py",
    f"src/{project_name}/components/data_transformation.py",
    f"src/{project_name}/components/model_trainer.py",
    f"src/{project_name}/pipelines/__init__.py",
    f"src/{project_name}/pipelines/train_pipeline.py",
    f"src/{project_name}/pipelines/predict_pipeline.py",
    f"src/{project_name}/logger.py",
    f"src/{project_name}/exception.py",
    f"src/{project_name}/utils.py",
    f"notebooks/eda.ipynb",
    f"notebooks/model_trails.ipynb",
    "setup.py",
    "requirements.txt"
    ]


for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)

    if filedir != "":
        os.makedirs(filedir, exist_ok=True)

        logging.info("Creating directory: {} for the file {}".format(filedir, filename))

    if (not os.path.exists(filepath)) or (os.path.getsize(filepath)==0):
        with open(filepath, "wb") as f:
            pass

        logging.info("Creating empty file: {}".format(filename))

    else:
        logging.info("{} file already exists".format(filename))
