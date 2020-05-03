from vehicles_model.config import config
from vehicles_model import __version__ as _version
from vehicles_model.manager import load_pipeline
import pandas as pd
import numpy as np
import joblib
import logging

_logger = logging.getLogger('vehicles_model')


def predict(data):
    data = pd.DataFrame(data)
    pipeline = load_pipeline()
    predictions = np.exp(pipeline.predict(data))
    results = {'predictions': predictions, "version": _version}
    _logger.info(
        f"making predictions with model version: {_version} "
        f"Inputs: {data} "
        f"Predictions: {predictions}"
    )
    return results
