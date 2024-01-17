# Air Quality Prediction Project

This GitHub repository contains code for predicting air quality levels using time series analysis, specifically implementing the Autoregressive Integrated Moving Average (ARIMA) model. The primary focus is on the PT08.S1(CO) parameter, which represents carbon monoxide levels.

## Prerequisites

Ensure you have the following libraries installed before running the code:

- NumPy
- pandas
- openpyxl
- Matplotlib
- Seaborn
- Plotly
- Statsmodels
- Scikit-learn
- Pandas Profiling

You can install the required packages using the following:

```bash
pip install numpy pandas matplotlib seaborn plotly statsmodels scikit-learn pandas-profiling
```

## Data

The project utilizes air quality data stored in the 'air-quality.xlsx' file. The dataset includes timestamped records of the PT08.S1(CO) parameter. The data is preprocessed and cleaned before applying the ARIMA model.

## Code Structure

### Data Exploration and Preprocessing

- **Import Libraries:**
  - Importing necessary libraries for data analysis, visualization, and time series modeling.

- **Directory Setup:**
  - Setting up the project directory on a macOS system.

- **Pandas Profiling:**
  - Generating a comprehensive report using Pandas Profiling to explore the dataset.

### Stationarity Check

- **Augmented Dickey-Fuller Test:**
  - Performing the Augmented Dickey-Fuller (ADF) test to check the stationarity of the time series data.

- **Histogram Visualization:**
  - Visualizing the distribution of PT08.S1(CO) values using a histogram.

### Data Wrangling

- **Data Wrangling Function:**
  - Defining a function to load, preprocess, and wrangle the dataset.

- **Resampling:**
  - Resampling PT08.S1(CO) data to provide the mean for each 15 hours.

### Data Visualization

- **Time Series Plot:**
  - Plotting a time series graph of PT08.S1(CO) levels.

- **Autocorrelation and Partial Autocorrelation Plots:**
  - Visualizing the autocorrelation function (ACF) and partial autocorrelation function (PACF) of the time series.

### Model Training and Evaluation

- **Train-Test Split:**
  - Splitting the data into training and test sets.

- **Baseline Mean Absolute Error:**
  - Calculating the baseline Mean Absolute Error (MAE).

- **Grid Search for Hyperparameters:**
  - Performing a grid search to find optimal hyperparameters (p and q) for the ARIMA model.

- **Grid Search Visualization:**
  - Visualizing the grid search results using a heatmap.

- **ARIMA Model Diagnostics:**
  - Displaying diagnostic plots of the ARIMA model.

### Walk Forward Validation

- **Model Prediction:**
  - Implementing walk-forward validation to predict air quality levels.

- **Walk Forward Validation Visualization:**
  - Visualizing the walk-forward validation predictions.

### Model Interpretability

- **SHAP Values:**
  - Utilizing SHAP (SHapley Additive exPlanations) values for model interpretability.

- **Model Persistence:**
  - Saving the trained ARIMA model for future use.

## Usage

1. Clone the repository:

   ```bash
   git clone https://github.com/adriankasito/Air-Quality-Prediction.git
   ```

2. Navigate to the project directory:

   ```bash
   cd Air-Quality-Prediction
   ```

3. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Run the Jupyter Notebook or Python script to perform air quality prediction and visualize the results.

   ```bash
   jupyter notebook Air Quality Prediction.ipynb
   ```

   or

   ```bash
   python Air Quality Prediction.py
   ```
