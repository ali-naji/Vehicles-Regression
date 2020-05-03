from vehicles_model.predict import predict
from vehicles_model.config import config
from vehicles_model.manager import load_dataset
import pandas as pd

sample_data = load_dataset().iloc[:1000]


def test_single_prediction():

    subject = predict(sample_data.loc[8:9, :])

    # tests
    assert subject is not None
    assert isinstance(subject.get('predictions')[0], float)
    assert subject.get('predictions')[0] > 0


def test_multiple_predictions():

    subject = predict(sample_data)

    # tests
    assert subject is not None
    assert len(subject.get('predictions')) <= len(sample_data)
