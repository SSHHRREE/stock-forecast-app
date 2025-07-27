
import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import feedparser

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# 📌 App Title
st.title("📈 LSTM Stock Price Forecast + News + Indicators")

# 📌 User Input
stock = st.text_input("Enter stock ticker (e.g., AAPL, TCS.NS, INFY.NS)", value="AAPL").upper()

# 📌 Fetch Data
start_date = "2023-01-01"
end_date = datetime.datetime.today().strftime('%Y-%m-%d')
data = yf.download(stock, start=start_date, end=end_date)

if data.empty:
    st.error("Invalid stock symbol or no data found.")
    st.stop()

# 📌 Preprocessing
data = data[['Close']]
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

prediction_days = 7
x_train, y_train = [], []
for i in range(prediction_days, len(scaled_data)):
    x_train.append(scaled_data[i - prediction_days:i, 0])
    y_train.append(scaled_data[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# 📌 Train Model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(units=1))
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(x_train, y_train, epochs=10, batch_size=32, verbose=0)

# 📌 Forecast
forecast_days = 7
last_60_days = scaled_data[-prediction_days:]
forecast_input = last_60_days.reshape(1, prediction_days, 1)
future_predictions = []

for _ in range(forecast_days):
    pred = model.predict(forecast_input, verbose=0)
    future_predictions.append(pred[0, 0])
    forecast_input = np.append(forecast_input[:, 1:, :], [[[pred[0, 0]]]], axis=1)

future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

# 📅 Forecast Dates
last_date = data.index[-1]
future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_days)
forecast_df = pd.DataFrame({'Date': future_dates, 'Predicted Price': future_predictions.flatten()})

# 📊 Plot Forecast
st.subheader("📊 7-Day Price Forecast")
fig1, ax1 = plt.subplots(figsize=(10, 4))
ax1.plot(data.index[-100:], data['Close'].values[-100:], label="Past Price (100 days)")
ax1.plot(future_dates, future_predictions, label="Predicted Price (7 days)", color='red')
ax1.set_title(f"{stock} - 7 Day Price Forecast")
ax1.set_xlabel("Date")
ax1.set_ylabel("Price")
ax1.legend()
ax1.grid()
st.pyplot(fig1)

# 📈 Show Forecast Table
st.dataframe(forecast_df)

# 📊 SMA Plot
data['SMA20'] = data['Close'].rolling(window=20).mean()
data['SMA50'] = data['Close'].rolling(window=50).mean()
st.subheader("📊 Simple Moving Averages")
fig2, ax2 = plt.subplots(figsize=(10, 4))
ax2.plot(data['Close'], label='Close Price')
ax2.plot(data['SMA20'], label='SMA 20')
ax2.plot(data['SMA50'], label='SMA 50')
ax2.set_title(f"{stock} - SMA (20 & 50)")
ax2.set_xlabel("Date")
ax2.set_ylabel("Price")
ax2.legend()
ax2.grid()
st.pyplot(fig2)

# 📉 RSI Plot
st.subheader("📉 Relative Strength Index (RSI)")
delta = data['Close'].diff()
gain = delta.where(delta > 0, 0)
loss = -delta.where(delta < 0, 0)
avg_gain = gain.rolling(window=14).mean()
avg_loss = loss.rolling(window=14).mean()
rs = avg_gain / avg_loss
rsi = 100 - (100 / (1 + rs))

fig3, ax3 = plt.subplots(figsize=(10, 3))
ax3.plot(rsi, label='RSI')
ax3.axhline(70, linestyle='--', color='red', alpha=0.5)
ax3.axhline(30, linestyle='--', color='green', alpha=0.5)
ax3.set_title(f"{stock} - RSI")
ax3.set_xlabel("Date")
ax3.set_ylabel("RSI")
ax3.legend()
ax3.grid()
st.pyplot(fig3)

# 📰 News Feed
st.subheader("📰 Latest Stock News")
query = stock + " stock"
rss_url = f"https://news.google.com/rss/search?q={query.replace(' ', '+')}"
news_feed = feedparser.parse(rss_url)

if not news_feed.entries:
    st.warning("❌ No news found.")
else:
    for entry in news_feed.entries[:5]:
        st.markdown(f"**🔗 [{entry.title}]({entry.link})**")
