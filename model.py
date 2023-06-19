import joblib 
import pandas as pd
import numpy as np
import os

# load the model file
curr_path = os.path.dirname(os.path.realpath(__file__))
ARIMA_model = joblib.load(curr_path + "/Air Quality Prediction/ARIMA_model.joblib")

# function to predict the air_quality
def predict(air_quality: int):
    """ Returns PT08.S1(CO) Level"""


    pred = ARIMA_model.predict(air_quality)
    

    return pred



'''import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

class AirQualityModel:
    def __init__(self):
        # Load the trained dataset with a 15-hour lag
        self.dataset = pd.read_excel('air-quality.xlsx')  # Replace with the path to your dataset

        # Preprocess the dataset
        # Assuming the dataset has columns 'Date' and 'PT08.S1(CO)'
        self.dataset['Date'] = pd.to_datetime(self.dataset['Date'])
        self.dataset = self.dataset.set_index('Date')

        # Create the ARIMA model
        self.model = ARIMA(self.dataset['PT08.S1(CO)'], order=(1, 0, 0))  # Modify the order as per your model

        # Fit the model
        self.model_fit = self.

    def predict(self, date):
        # Convert date to pandas datetime format
        target_date = pd.to_datetime(date)

        # Generate the predictions
        predictions = self.model_fit.predict(start=len(self.dataset), end=len(self.dataset)+1, typ='levels')  # Predict 2 values (month and next month)

        # Format the predictions
        result = {
            'date': [str(target_date), str(target_date + pd.DateOffset(months=1))],
            'PT08.S1(CO)': [predictions[0], predictions[1]]
        }

        return result'''
