import streamlit as st
import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)
import time
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import seaborn as sns
from sklearn.metrics import mean_absolute_error
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller


def wrangle(file_path):
    df = pd.read_excel(file_path)

    def format_time(time_str):
        # Split time into parts
        hours, minutes, seconds = time_str.split(".")
        # Add leading zero if necessary
        hours = hours.zfill(2)  # Ensures two-digit hours
        return f"{hours}.{minutes}.{seconds}"

    # Apply the function to format time strings
    df['Time'] = df['Time'].apply(format_time)

    df["Timestamp"] = df["Date"].astype(str) + " " + df["Time"].astype(str)
    # Convert to datetime using the standard format string
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='%Y-%m-%d %H.%M.%S')

    df.drop(columns=['Time', 'Date'], inplace=True)
    df = df.set_index("Timestamp")
    df.index = df.index.tz_localize("UTC").tz_convert("Africa/Kampala")
    df = df[df["PT08.S1(CO)"] > 650]
    y = df['PT08.S1(CO)'].resample("15H").mean().fillna(method="ffill")
    return y


# Function for model training and prediction
def train_predict_model(y_train, y_test):
    model = ARIMA(y_train, order=(16,0,0)).fit()
    y_pred = model.forecast(steps=len(y_test))
    return y_pred, model

# Function for visualizing predictions and diagnostics
def visualize_results(y_test, y_pred, model):
    df_pred_test = pd.DataFrame({"y_test":y_test, "y_pred": y_pred})

    # Predictions vs. Actuals with Plotly Express
    fig = px.line(df_pred_test, labels={"value": "PT08.S1(CO)"})
    fig.update_layout(
        title="Walk Forward Validation Predictions",
        xaxis_title="Date",
        yaxis_title="PT08.S1(CO) Level",
    )
    st.plotly_chart(fig)

    # Model Diagnostics with statsmodels (unchanged)
    fig, ax = plt.subplots(figsize=(15, 12))
    model.plot_diagnostics(fig=fig)
    st.pyplot(fig)

# Streamlit app
def main():
    st.title("Air Quality Prediction App")
    st.write("""
    Welcome to the Air Quality Prediction App! This app uses ARIMA models to forecast air quality levels based on historical data.
    """)

    # Upload data file
    uploaded_file = st.file_uploader("airq", type=["xlsx"])

    if uploaded_file is not None:
        st.write("Data file uploaded successfully!")

        # Wrangle data
        y = wrangle(uploaded_file)

        # Perform ADF test for stationarity
        result = adfuller(y)
        adf_statistic = result[0]
        p_value = result[1]

        st.write(f'ADF Statistic: {adf_statistic}')
        st.write(f'p-value: {p_value}')

        # Visualize histogram of data with Plotly Express
        fig = px.histogram(y, title='Histogram of PT08.S1(CO)', color_discrete_sequence=['skyblue'])
        st.plotly_chart(fig)

        # Perform train-test split
        cutoff_test = int(len(y) * 0.80)
        y_train = y.iloc[:cutoff_test]
        y_test = y.iloc[cutoff_test:]

        # Train and predict using ARIMA model
        y_pred, model = train_predict_model(y_train, y_test)

        # Visualize predictions and diagnostics
        st.write("Predictions vs. Actuals:")
        visualize_results(y_test, y_pred, model)

if __name__ == "__main__":
    main()
