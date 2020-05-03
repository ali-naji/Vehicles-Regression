from vehicles_model.config import config
from vehicles_model import __version__ as model_version
from api import __version__ as api_version
from vehicles_model.manager import load_dataset
import pandas as pd
import json


def test_health_endpoint_returns_200(flask_test_client):
    # When
    response = flask_test_client.get('/health')

    # Then
    assert response.status_code == 200


def test_versions(flask_test_client):
    response = json.loads(flask_test_client.get('/version').data)

    assert response['model_version'] == model_version
    assert response['api_version'] == api_version


def test_prediction_endpoint_returns_prediction(flask_test_client):
    sample_data = load_dataset().loc[:1000, :]

    sample_input = sample_data[0:1].to_json(orient='records')

    response = flask_test_client.post('/v1/predict',
                                      json=json.loads(sample_input))
    assert response.status_code == 200
    response_json = json.loads(response.data)
    prediction = response_json['predictions']
    response_version = response_json['version']
    assert prediction is not None
    assert response_version == model_version
