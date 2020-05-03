from vehicles_model.config import config
from vehicles_model.processing.preprocessors import Pipeline
import logging
from vehicles_model.manager import *

_logger = logging.getLogger('vehicles_model')

pipeline = Pipeline(target=config.TARGET,
                    categorical_to_impute=config.CATEGORICAL_TO_IMPUTE,
                    numerical_to_impute=config.NUMERICAL_TO_IMPUTE,
                    categoricals=config.CATEGORICALS,
                    ohe_catgs=config.OHE_FEATS,
                    ord_catgs=config.ORD_FEATS,
                    unneeded_vars=config.VARS_TO_DROP,
                    numerical_ranges=config.NUM_RANGES)


if __name__ == '__main__':
    data = load_dataset()
    _logger.info('pipeline training started')
    pipeline.fit(data)
    save_pipeline(pipeline)
