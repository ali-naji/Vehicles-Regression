from flask import Blueprint, request, jsonify, render_template, flash, redirect
from api.config import get_logger
from vehicles_model.predict import predict
from vehicles_model import __version__ as model_version
from api import __version__ as api_version
from vehicles_model.config import config
import numpy as np

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

        input_dict = create_input_dict(input_data)

        _logger.info(f'Inputs: {input_dict}')

        prediction = np.round(predict(input_dict)['predictions'][0], 2)
        _logger.info(f'Outputs: {prediction}')

        return render_template('result.html', prediction=prediction)


def create_input_dict(input_dict):
    ''' returns dictionary matching column order of training-data '''
    return {'id': [np.nan], 'url': [np.nan], 'region': input_dict['region'], 'region_url': [np.nan], 'price': [1000],
            'year': [float(input_dict['year'][0])], 'manufacturer': input_dict['manufacturer'], 'model': input_dict['model'], 'condition': input_dict['condition'], 'cylinders': input_dict['cylinders'], 'fuel': input_dict['fuel'], 'odometer': [float(input_dict['odometer'][0])], 'title_status': input_dict['title_status'], 'transmission': input_dict['transmission'], 'vin': [np.nan], 'drive': input_dict['drive'], 'size': input_dict['size'], 'type': input_dict['type'], 'paint_color': input_dict['paint_color'], 'image_url': [np.nan], 'description': input_dict['description'], 'county': [np.nan], 'state': input_dict['state'], 'lat': [float(input_dict['lat'][0])], 'long': [float(input_dict['long'][0])]}


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
        flash("Message sent successfully. Thanks for letting us know")
        return redirect('/contact')


@prediction_app.route('/version', methods=['GET'])
def version():
    if request.method == 'GET':
        return jsonify({'model_version': model_version,
                        'api_version': api_version})
