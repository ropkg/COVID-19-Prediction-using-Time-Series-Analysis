import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Download the COVID-19 dataset from JHU GitHub
url = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv"
data = pd.read_csv(url)

# Transpose the data to have dates as index and countries/regions as columns
data = data.drop(['Province/State', 'Lat', 'Long'], axis=1)
data = data.groupby('Country/Region').sum().transpose()

# Reset index to have 'Date' as a column
data = data.reset_index()

# Rename columns
data.columns = ['Date'] + list(data.columns[1:])

# Convert the 'Date' column to datetime format
data['Date'] = pd.to_datetime(data['Date'])

# Split the data into training and testing sets from the beginning
train_size = int(len(data) * 0.8)
train, test = data[:train_size], data[train_size:]

# Fit the Exponential Smoothing model
model = ExponentialSmoothing(train['US'], seasonal='add', seasonal_periods=7)  # You can choose a different country or region
fit_model = model.fit()

# Make predictions on the entire dataset
predictions = fit_model.predict(start=0, end=len(data) - 1)

# Calculate Mean Squared Error on the entire dataset
mse = mean_squared_error(data['US'], predictions)
print(f'Mean Squared Error: {mse}')

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(data['Date'], data['US'], label='Actual')
plt.plot(data['Date'], predictions, label='Predictions', color='red')
plt.title('COVID-19 Cases Prediction using Time Series Analysis')
plt.xlabel('Date')
plt.ylabel('Number of Cases')
plt.legend()
plt.show()
