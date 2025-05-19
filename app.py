import numpy as np
import pandas as pd
import yfinance as yf
from keras.models import load_model
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Page config
st.set_page_config(page_title="📈 Stock Market Predictor", layout="wide")

# Load model
model = load_model("Stock Predictions Model.keras")

# Sidebar
st.sidebar.title("🔧 Settings")
stock = st.sidebar.text_input('Enter Stock Symbol', 'GOOG')
start = st.sidebar.date_input("Start Date", pd.to_datetime("2012-01-01"))
end = st.sidebar.date_input("End Date", pd.to_datetime("2022-12-31"))

# Title & Header
st.markdown("<h1 style='text-align: center; color: #3366cc;'>📊 Stock Market Predictor</h1>", unsafe_allow_html=True)
st.markdown("---")

# Load Data
data = yf.download(stock, start, end)
st.subheader(f"💹 Raw Stock Data for: `{stock}`")
st.dataframe(data.tail(), use_container_width=True)

# Split data
data_train = pd.DataFrame(data.Close[0: int(len(data)*0.80)])
data_test = pd.DataFrame(data.Close[int(len(data)*0.80):])

# Scaling
scaler = MinMaxScaler(feature_range=(0,1))
pas_100_days = data_train.tail(100)
data_test = pd.concat([pas_100_days, data_test], ignore_index=True)
data_test_scale = scaler.fit_transform(data_test)

# Moving Averages
ma_50_days = data.Close.rolling(50).mean()
ma_100_days = data.Close.rolling(100).mean()
ma_200_days = data.Close.rolling(200).mean()

# Charts
col1, col2 = st.columns(2)

with col1:
    st.subheader('📈 Price vs MA50')
    fig1 = plt.figure(figsize=(8,5))
    plt.plot(data.Close, label='Close Price', color='green')
    plt.plot(ma_50_days, label='MA50', color='red')
    plt.legend()
    st.pyplot(fig1)

with col2:
    st.subheader('📈 Price vs MA50 vs MA100')
    fig2 = plt.figure(figsize=(8,5))
    plt.plot(data.Close, label='Close Price', color='green')
    plt.plot(ma_50_days, label='MA50', color='red')
    plt.plot(ma_100_days, label='MA100', color='blue')
    plt.legend()
    st.pyplot(fig2)

st.subheader('📈 Price vs MA100 vs MA200')
fig3 = plt.figure(figsize=(10,5))
plt.plot(data.Close, label='Close Price', color='green')
plt.plot(ma_100_days, label='MA100', color='red')
plt.plot(ma_200_days, label='MA200', color='blue')
plt.legend()
st.pyplot(fig3)

# Prepare data for prediction
x, y = [], []
for i in range(100, data_test_scale.shape[0]):
    x.append(data_test_scale[i-100:i])
    y.append(data_test_scale[i,0])

x, y = np.array(x), np.array(y)

# Prediction
predict = model.predict(x)
scale = 1 / scaler.scale_
predict = predict * scale
y = y * scale

# Plot predictions
st.subheader('🎯 Original Price vs Predicted Price')
fig4 = plt.figure(figsize=(10,5))
plt.plot(y, label='Original Price', color='green')
plt.plot(predict, label='Predicted Price', color='red')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig4)

# Footer
st.markdown("---")
st.markdown("<h6 style='text-align: center;'>Made with ❤️ by Gunjan Singh</h6>", unsafe_allow_html=True)
