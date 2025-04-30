import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import root_mean_squared_error, mean_squared_error, mean_absolute_error
from statsmodels.tsa.arima.model import ARIMA
from math import sqrt
from sklearn.model_selection import train_test_split

df = pd.read_csv('supermarket_sales - Sheet1.csv')

print(df.head())
print(df.info())

df1 = df[df['City'] == 'Yangon']
df1 = df1[['Date', 'Total']]

df1['Date'] = pd.to_datetime(df1['Date'])

df1 = df1.sort_values('Date')
df1.set_index('Date', inplace=True)

df1.plot(kind='line')
plt.show()

k = adfuller(df1)
print(k[0], k[1])

df1 = df1['Total'].resample('D').mean()

multi_mode = seasonal_decompose(df1)
multi_mode.plot()
plt.show()

# Plot ACF and PACF
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plot_acf(df1, lags=30, ax=plt.gca())
plt.title("Autocorrelation Function (ACF)")

plt.subplot(1, 2, 2)
plot_pacf(df1, lags=30, ax=plt.gca())
plt.title("Partial Autocorrelation Function (PACF)")

plt.tight_layout()
plt.show()

# model = ARIMA(df1, order=(2, 1, 2))  # You can adjust (p,d,q) as needed
# model_fit = model.fit()
# print(model_fit.summary())
#
# # Forecast next 30 days
# forecast = model_fit.forecast(steps=30)
#
# # Plot forecast
# plt.figure(figsize=(10, 5))
# plt.plot(df1, label='Historical')
# plt.plot(forecast, label='Forecast (30 days)', linestyle='--')
# plt.legend()
# plt.title("ARIMA Forecast")
# plt.show()
target = df.dropna()['Total']
split_point = int(len(target) * 2 / 3)
y_train, y_test = target[:split_point], target[split_point:]
model = ARIMA(y_train, order=(0, 3, 1))  # p,q,d
model_fit = model.fit()
predictions = model_fit.forecast(len(y_test))
mse = mean_squared_error(y_test, predictions)
mae = mean_absolute_error(y_test, predictions)
rmse = sqrt(mse)
print(mse)
print(mae)
print(rmse)

plt.figure(figsize=(10, 5))
plt.plot(y_train, label='Historical')
plt.plot(predictions, label='Forecast (30 days)', linestyle='--')
plt.legend()
plt.title("ARIMA Forecast")
plt.show()
