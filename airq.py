#!/usr/bin/env python
# coding: utf-8

# "Forecasting Air Quality in Cities Using ARIMA Models: Unlocking the Future" 
# 
# 
# "Imagine having the power to predict the future air quality in cities, enabling us to make informed decisions and take proactive measures for a cleaner and healthier environment. In this captivating article, we delve into the world of Autoregressive Integrated Moving Average (ARIMA) models and their extraordinary capabilities in forecasting air quality over long time periods. Join us on this fascinating journey as we explore the intricacies of ARIMA modeling, unravel its potential for environmental forecasting, and provide you with step-by-step code examples to unleash the predictive power of ARIMA models. Get ready to dive into a world where data meets sustainability, and embark on an adventure that empowers us to shape a better tomorrow."
# 

# The data was accessed from https://www.kaggle.com/datasets/harinarayanan22/airquality?select=AirQualityUCI.csv and tranformed into an excel file for easy reading. Data was collected from March 2004 to February 2005 (a period of one year), making it the longest free recording of responses from on-field installed chemical air quality sensor devices. A co-located reference certified analyzer supplied Ground Truth hourly averaged readings for CO, Non Methane Hydrocarbons, Benzene, Total Nitrogen Oxides (NOx), and Nitrogen Dioxide (NO2). According to De Vito et al., Sens. And Act. B, Vol. 129,2,2008, there are indications of cross-sensitivities as well as concept and sensor drifts

# In[1]:


#import required libraries
import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)
import inspect
import time
import warnings
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import seaborn as sns
from sklearn.metrics import mean_absolute_error
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
warnings.filterwarnings("ignore")


# In[27]:


# setting up os env on my mac 
import os
for dirname, _, filenames in os.walk('/Air Quality Prediction'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[2]:


#A glimpse of the data
data = pd.read_excel('air-quality (1).xlsx')
data.head(4)


# In[31]:


data.Date.unique()


# In[3]:


from pandas_profiling import ProfileReport
data_report = ProfileReport(data)
data_report


# In[4]:


#Test Stationarity
dat = data['PT08.S1(CO)']
# Perform ADF test
result = adfuller(dat)

# Extract ADF test statistics and p-value
adf_statistic = result[0]
p_value = result[1]

# Print the results
print(f'ADF Statistic: {adf_statistic}')
print(f'p-value: {p_value}')

# Interpret the results
if p_value < 0.05:
    print('The time series is likely stationary (reject the null hypothesis of non-stationarity).')
else:
    print('The time series is likely non-stationary (fail to reject the null hypothesis of non-stationarity).')


# In[5]:


data['PT08.S1(CO)'].hist(bins=30, color='skyblue', edgecolor='black')
# Customize the plot
plt.title('Histogram of PT08.S1(CO)')
plt.xlabel('Values')
plt.ylabel('Frequency')

# Add gridlines
plt.grid(True, linestyle='--', alpha=0.5)

# Show the plot
plt.show()


# Outliers are data points that deviate significantly from the typical pattern of the air quality dataset. A visual depiction of the distribution of data is represented by the above histogram. By examining the shape of the histogram, you can gain insights into the central tendency, spread, and skewness of the data. There’s an unusual or unexpected pattern in the histogram that  indicates the presence of values far below 500 that I would classify as outliers.
# 

# In[6]:


def wrangle(file_path):
      
    # Read to dataframe
    df = pd.read_excel(file_path)
    
    # Convert date column to datetime data type
    df['Date'] = pd.to_datetime(df['Date'])

     # Convert time column to datetime data type
    df['Time'] = pd.to_datetime(df['Time'], format='%H.%M.%S').dt.time

     # Concatenate date and time columns
    df['Timestamp'] = df.apply(lambda row: pd.datetime.combine(row['Date'].date(), row['Time']), axis=1)
    df.drop(columns=['Time', 'Date'], inplace=True)
    
    #Set Timestamp as index
    df = df.set_index("Timestamp")
    
    #Localize timezone
    df.index = df.index.tz_localize("UTC").tz_convert("Africa/Kampala")
    
    #Remove outliers above 650
    df = df[df["PT08.S1(CO)"] > 650]
    
    #Resample PT08.S1(CO) data to provide mean for each 15 hours
    y = df['PT08.S1(CO)'].resample("15H").mean().fillna(method="ffill")
    
    return y


# In this dataset we shall predict CO with a 15 hour window. Based on the chemoresistance concept, the PT08.S1(CO) gas sensor measures the amount and concentration of carbon monoxide in the air and adjusts its electrical resistance accordingly. The sensor comprises a sensing device that, when in contact with CO, reacts chemically, often through the use of a metal oxide substance. The sensor can detect and measure the concentration of carbon monoxide because this reaction changes the sensing material's electrical conductivity.

# In[7]:


y = wrangle('air-quality.xlsx')
y.head()


# In[8]:


fig, ax = plt.subplots(figsize=(15, 6))
plt.plot(y, color='darkred', linestyle='-', linewidth=2)
# Label axes
plt.ylabel('PT08.S1(CO)')
plt.xlabel("Date")
plt.title('PT08.S1(CO) Levels')
# Add gridlines
plt.grid(True, linestyle='--', alpha=0.5)

# Show or save the plot
plt.show()


# With the localized time zone, eliminated outliers, resampled data, and filled null values, this line plot of the PT08.S1(CO) air quality levels can offer important insights for ARIMA modeling. Understanding historical CO patterns and changes facilitates choosing and fine-tuning the ARIMA model parameters for precise forecasting and analysis.
# 
# 

# In[9]:


#Plotting the ACF
fig, ax = plt.subplots(figsize=(13, 4))
plot_acf(y, ax=ax, color='green')
plt.xlabel("Lags [hours]")
plt.ylabel("Correlation Coefficient")
plt.title("PT08.S1(CO)Readings, ACF")
# Add gridlines
plt.grid(True, linestyle='--', alpha=0.5)

# Show the plot
plt.show()


# In[10]:


#Plotting the PACF
fig, ax = plt.subplots(figsize=(13, 4))
plot_pacf(y, ax=ax)
plt.xlabel("Lags [hours]")
plt.ylabel("Correlation Coefficient")
plt.title("PT08.S1(CO) Readings, PACF")
# Add gridlines
plt.grid(True, linestyle='--', alpha=0.5)

# Show the plot
plt.show()


# 
# We can determine whether there is a relationship between present and historical values and whether the data shows any patterns or trends thanks to autocorrelation (ACF). When the autocorrelation at a certain lag is large, it is likely that the previous values at that lag have an impact on the present observation.
# On the other hand, partial autocorrelation (PACF) helps pinpoint a specific lag's direct impact on the present observation while minimizing the impact of other lags. In other words, partial autocorrelation, which is independent of other lags, demonstrates the distinctive contribution of each lag to the correlation with the current observation.
# In order to assess the association between data in a time series, Box et al., 2015 states that autocorrelation and partial autocorrelation are statistical metrics employed in ARIMA models and they are critical in defining the right orders (p, d, q) for ARIMA modeling, making it easier to capture temporal relationships and making it possible to make precise forecasts.
# 
# 

# In[11]:


#Splitting data into train and test data 
cutoff_test = int(len(y) * 0.80)
y_train = y.iloc[:cutoff_test]
y_test = y.iloc[cutoff_test:]
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)


# The final step in our data exploration is to split the small dataset into training and test sets. We use a 80/20 split, where 80% of the data was our training set, and 20% of it is our test set  though it doesn't bring it into line with "statsmodels" default confidence interval which is a 95/5 split.

# In[12]:


y_train_mean = y_train.mean()
y_pred_baseline = [y_train_mean] * len(y_train)
mae_baseline = mean_absolute_error(y_train, y_pred_baseline)

print("Mean PT08.S1(CO) Reading:", y_train_mean)
print("Baseline Mean Absolute Error:", mae_baseline)


# In[13]:


p_params = range(0,25, 8)
q_params = range(0,3,1)


# In[14]:


# Create dictionary to store MAEs
mae_grid = dict()
# Outer loop: Iterate through possible values for `p`
for p in p_params:
    # Create key-value pair in dict. Key is `p`, value is empty list.
    mae_grid[p] = list()
    # Inner loop: Iterate through possible values for `q`
    for q in q_params:
        # Combination of hyperparameters for model
        order = (p, 0, q)
        # Note start time
        start_time = time.time()
        # Train model
        model = ARIMA(y_train, order=order).fit()
        # Calculate model training time
        elapsed_time = round(time.time() - start_time, 2)
        print(f"Trained ARIMA {order} in {elapsed_time} seconds.")
        # Generate in-sample (training) predictions
        y_pred = model.predict()
        # Calculate training MAE
        mae = mean_absolute_error(y_train, y_pred)
        # Append MAE to list in dictionary
        mae_grid[p].append(mae)

print()
print(mae_grid)


# In[15]:


mae_df = pd.DataFrame(mae_grid)
mae_df.round(4)


# Every possible combination of the hyperparameters in p_params and q_params is used to train a model. Looking at q_params and p_params,  the mean absolute error is determined and then saved to a dictionary for each training cycle of the model. p_params comprise longer ranges because that is how ARMA models work, when you look at AR, you look into the past and far into the past to make predictions about present or future, but q_params is short term so you have few numbers in the range.
# 
# In the ARMA model, when we think about hyperparameters, we think in terms of p values and q values. p values represent the number of lagged observations included in the model (AR), and the q is the size of the moving average window (MA) or the normally called the error lag. These values count as hyperparameters because we get to decide what they are.
# Hyperparameters are set before the model is trained, which means that they significantly impact how the model is trained, and how it subsequently performs.
# 

# In[16]:


sns.heatmap(mae_df, cmap="Blues")
plt.xlabel("p values")
plt.ylabel("q values")
plt.title("ARMA Grid Search (Criterion: MAE)");


# From the histogram, with a p value of 0, the model doesn't perform well as compared to values of 8 and above. The best model can be determined from the data frame by looking at the MAE. Nevertheless, even if model, ARIMA(16,0,2) is the best performer, it takes longer to train than model, ARIMA(16,0,0) and as a data scientist, you want to reduce the MAE as much as you can whilst taking into consideration other resources as well
# 
# The more terms you include in your model for resources, the more calculations your computer must perform to train the model. This is a small model, but consider larger ones. Resources are essential; time spent staring at a computer screen has greater operational costs, such as power, so cutting down on the time can be cost-effective. Hence, when considering the model, you should consider performance and resource usage, which is advantageous for any firm.
# 

# In[17]:


fig, ax = plt.subplots(figsize=(15, 12))
model.plot_diagnostics(fig=fig);


# A random set of data points distributed evenly on both sides of the line makes up the ideal residual plot. Although there are some notable outliers, overall, the bars depict an even band of values, which is what we're looking for, in the plot we just created.
# Although the centre bar in our histogram is quite big, a normal distribution-like shape is described by all the bars.
# 
# 

# In[18]:


y_pred_wfv = pd.Series()
history = y_train.copy()
for i in range(len(y_test)):
    model = ARIMA(history, order=(16,0,0)).fit()
    next_pred = model.forecast()
    y_pred_wfv = y_pred_wfv.append(next_pred)
    history = history.append(y_test[next_pred.index])


# In[19]:


df_pred_test = pd.DataFrame({"y_test":y_test, "y_pred_wfv": y_pred_wfv})
fig = px.line(df_pred_test, labels={"value": "PM2.5"})
fig.update_layout(
    title="Walk Forward Validation Predictions",
    xaxis_title="Date",
    yaxis_title="PT08.S1(CO) Level",
)


# In[22]:


import statsmodels.api as sm
import shap
import joblib


# In[23]:


model


# In[24]:


# After training, save the model using joblib library
joblib.dump(model, 'ARIMA_model.joblib')


# In[25]:


import os

current_path = os.getcwd()
print(current_path)


# In[ ]:


os.path.realpath(


# That looks much better! Now the predictions are actually tracking the test data

# # THANK YOU, BYE
# ADRIAN KASITO

# In[ ]:




