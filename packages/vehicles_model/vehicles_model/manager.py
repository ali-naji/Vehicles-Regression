import pandas as pd
import boto3
import os
import joblib
import logging
import vehicles_model
from vehicles_model import __version__ as _version
from vehicles_model.config import config

_logger = logging.getLogger('vehicles_model')

s3 = boto3.client('s3', aws_access_key_id=os.environ.get(
    'AWS_ACCESS_KEY_ID'), aws_secret_access_key=os.environ.get('AWS_SECRET_ACCESS_KEY'))


def load_dataset(dataset_file='vehicles.csv'):
    save_path = (config.DATASET_DIR / dataset_file).absolute().as_posix()
    if not os.path.exists(save_path):
        _logger.info(f'Dataset not found in {save_path}, will downloaded from source')
        s3.download_file('myprivatedatasets', 'vehicles_model/'+ dataset_file,  save_path)

    df = pd.read_csv(save_path)
    _logger.info(f'dataset loaded successfully from path : {save_path}')

    return df

def load_pipeline(pipeline_file=f"{config.PIPELINE_FILENAME}{_version}.pkl"):
    
    save_path = (config.TRAINED_MODEL_DIR / pipeline_file).absolute().as_posix()
    if not os.path.exists(save_path):
        _logger.info(f'Trained model not found in {save_path}, will downloaded from source')
        s3.download_file('mytrainedmodels', 'vehicles_model/' +pipeline_file, save_path)
    _pipeline = joblib.load(config.TRAINED_MODEL_DIR / pipeline_file)
    _logger.info(f'pipeline loaded successfully from path : {save_path}')

    return _pipeline


def save_pipeline(pipeline_object) -> None:
    pipeline_name = f"{config.PIPELINE_FILENAME}{_version}.pkl"
    save_path = config.TRAINED_MODEL_DIR / pipeline_name
    joblib.dump(pipeline_object, save_path)
    _logger.info(
        f'{config.PIPELINE_FILENAME} saved successfully in path : {save_path}')
