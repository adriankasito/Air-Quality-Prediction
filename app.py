from flask import Flask, render_template, Response
from flask_restful import reqparse, Api
import flask

import numpy as np
import pandas as pd
import ast

import os
import json

from model import predict_yield

curr_path = os.path.dirname(os.path.realpath(__file__))

from flask import Flask, request, jsonify
from model import AirQualityModel

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    # Retrieve the request data
    data = request.get_json()
    date = data['Date']  # Date for which prediction is requested

    # Create an instance of the AirQualityModel
    model = ARIMA_model()

    # Generate the predictions
    predictions = model.predict(date)

    # Format the predictions
    result = {
        'date': predictions['Date'],
        'PT08.S1(CO)': predictions['PT08.S1(CO)']
    }

    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
