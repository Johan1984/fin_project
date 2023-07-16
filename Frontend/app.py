import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px
from PIL import Image
import time
import requests

# Title
st.title('MoonStrat')

# Description
st.write("""
# MoonStrat: A Visionary Trading Strategy
MoonStrat is a next-gen trading model that aims to predict the direction of the S&P 500 index.
It leverages the power of Recurrent Neural Networks (RNN) and Long Short-Term Memory (LSTM) models,
capable of learning temporal dependencies and making predictions based on historical data.

The model forecasts the S&P 500's direction and makes strategic decisions accordingly.
If it predicts an upward trend, the model takes a long position. Conversely, during a predicted downturn,
the model stays in cash, mitigating potential losses.

As a forward-looking strategy, MoonStrat strives to optimize investment returns while minimizing risks.
""")

# Call to prediction API
response = requests.get("https://my-api-a2qb64qvma-uc.a.run.app/predict")
prediction = response.json()['prediction'] # Access the 'prediction' key of the dictionary

if prediction == "BUY":
    st.markdown("<p style='color:green; font-size:20px;'>The model's current prediction is to BUY the SP500.</p>", unsafe_allow_html=True)
elif prediction == "SELL":
    st.markdown("<p style='color:red; font-size:20px;'>The model's current prediction is to SELL the SP500.</p>", unsafe_allow_html=True)
else:
    st.markdown("<p style='color:orange; font-size:20px;'>There is an error in the prediction.</p>", unsafe_allow_html=True)



# Load Images
img1 = Image.open('strategy_vs_spx.png')
st.image(img1, caption='Strategy vs SPX: This image displays the performance of our strategy compared to the S&P 500 index.', use_column_width=True)

img2 = Image.open('perfomance_strategy.png')
st.image(img2, caption='Performance Strategy: This image shows the performance of our strategy over time.', use_column_width=True)

img3 = Image.open('all_features.png')
st.image(img3, caption='All Features: This image shows all the features considered in the prediction model.', use_column_width=True)

# Real-Time SP500 chart
st.write("""
# S&P 500 Real-Time Chart
The following chart updates every hour with the latest S&P 500 price data.
""")

def plot_real_time():
    data = yf.download('^GSPC', period = '1d', interval = '1m')
    fig = px.line(data, x=data.index, y='Close', labels={'Close':'Price'}, title='S&P 500 Real-Time Chart')
    st.plotly_chart(fig)

plot_real_time()
