from flask import Blueprint, request, jsonify, render_template
from api.config import get_logger
from vehicles_model.predict import predict
from vehicles_model import __version__ as model_version
from api import __version__ as api_version
from vehicles_model.config import config

_logger = get_logger(logger_name=__name__)

prediction_app = Blueprint('prediction_app', __name__)


@prediction_app.route('/main')
@prediction_app.route('/')
def index():
    return render_template('main.html')


@prediction_app.route('/form', methods=['GET', 'POST'])
def form():
    if request.method == 'GET':
        return render_template('form.html')
    else:
        input_data = {key: [value] for (
            key, value) in request.form.items()}

        for key in config.VARS_TO_DROP:
            input_data[key] = ['']
        input_data[config.TARGET] = [1000.0]
        for key in config.NUMERICAL_TO_IMPUTE:
            input_data[key] = [float(input_data[key][0])]

        _logger.info(f'Inputs: {input_data}')
        
        prediction = predict(input_data)['predictions'][0]
        _logger.info(f'Outputs: {prediction}')

        return render_template('result.html', prediction=prediction)

@prediction_app.route('/test', methods=['POST'])
def test_prediction():
    json_data = request.get_json()
    _logger.debug(f'Inputs: {json_data}')

    result = predict(json_data)
# Step 4: Convert numpy ndarray to list
    predictions = result.get('predictions').tolist()
    version = result.get('version')

    # Step 5: Return the response as JSON
    return jsonify({'predictions': predictions,
                    'version': version})
    



@prediction_app.route('/contact', methods=['GET', 'POST'])
def contact():
    if request.method == 'GET':
        return render_template('contact.html')
    else:
        contact_dict = {key: value for (key, value) in request.form.items()}
        return str(contact_dict)


@prediction_app.route('/version', methods=['GET'])
def version():
    if request.method == 'GET':
        return jsonify({'model_version': model_version,
                        'api_version': api_version})
