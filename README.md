# CodeAlpha 
# Step 1: Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Step 2: Load Dataset
df = pd.read_csv('AMZN_2006-01-01_to_2018-01-01.csv')
df = df[['Date', 'Close']]
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)
df.sort_index(inplace=True)

# Step 3: Plot Closing Prices
plt.figure(figsize=(14, 6))
plt.plot(df['Close'], label='Amazon Closing Price')
plt.title('Amazon Stock Price History')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.show()

# Step 4: Normalize Data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(df['Close'].values.reshape(-1, 1))

# Step 5: Prepare Training Data
prediction_days = 60
X_train, y_train = [], []

for i in range(prediction_days, len(scaled_data)):
    X_train.append(scaled_data[i - prediction_days:i, 0])
    y_train.append(scaled_data[i, 0])

X_train, y_train = np.array(X_train), np.array(y_train)
X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))

# Step 6: Build LSTM Model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(units=1))  # Predicted price

model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, epochs=25, batch_size=32)

# Step 7: Make Predictions
test_data = scaled_data[-(prediction_days + 30):]
X_test = []

for i in range(prediction_days, len(test_data)):
    X_test.append(test_data[i - prediction_days:i, 0])

X_test = np.array(X_test)
X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
predicted_prices = model.predict(X_test)
predicted_prices = scaler.inverse_transform(predicted_prices)

real_prices = df['Close'].values[-30:]

# Step 8: Plot Results
plt.figure(figsize=(14, 6))
plt.plot(real_prices, color='black', label='Actual AMZN Price')
plt.plot(predicted_prices, color='blue', label='Predicted AMZN Price')
plt.title('Amazon Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Stock Price (USD)')
plt.legend()
plt.show()

# Step 9: Predict Next Day Price
last_60_days = df['Close'].values[-60:]
last_60_days_scaled = scaler.transform(last_60_days.reshape(-1, 1))
X_future = np.array([last_60_days_scaled])
X_future = X_future.reshape((X_future.shape[0], X_future.shape[1], 1))
future_price = model.predict(X_future)
future_price = scaler.inverse_transform(future_price)

print(f"Predicted next day Amazon price: ${future_price[0][0]:.2f}")
