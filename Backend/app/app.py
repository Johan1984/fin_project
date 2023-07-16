from fastapi import FastAPI
from typing import Dict
from pydantic import BaseModel

# Data minipulation
import pandas as pd
import numpy as np
import uvicorn

# Feature creation
import yfinance as yf
from ta import momentum
from ta.volatility import BollingerBands
from ta.trend import MACD
from ta.momentum import RSIIndicator
from ta.volume import OnBalanceVolumeIndicator
from ta.momentum import AwesomeOscillatorIndicator

#Tensorflow
import tensorflow as tf

model = tf.keras.models.load_model('model_test2')

def yf_download(tickers, years='25y'):
    ticker_dict = {}
    for ticker in tickers:
        df_ticker = yf.download(ticker,
                                period=years,
                                interval='1d',
                                ignore_tz=True,
                                prepost=False
                                )
        ticker_dict[ticker] = df_ticker
    return ticker_dict

bench_target = '^SPX'
ticker_list = [bench_target, '^VIX', 'DX-Y.NYB', '^TNX', 'GC=F', 'CL=F', '^FVX', '^IRX']

ticker_dict = yf_download(ticker_list)

def extract_features(ticker_dict):
    """
    Extract various features for each stock in the given dictionary.

    Parameters:
    ticker_dict (dict): A dictionary with stock symbols as keys and pandas DataFrames as values.
                        Each DataFrame represents historical data for a stock and should contain columns for 'Adj Close', 'Open', 'Close', 'High', 'Low', and 'Volume'.

    Returns:
    pandas.DataFrame: A DataFrame containing various calculated features for each stock.
    """

    all_df = pd.DataFrame(index=ticker_dict[bench_target].index)
    print(all)
    ticker_start_date_list = []
    ticker_end_date_list = []

    # Iterate through the dictionary of stocks
    for index, df_index in ticker_dict.items():
        # Create a prefix for column names with the stock symbol
        column_prefix = f'{index}_'

        # Add the stock data to the combined DataFrame with modified column names
        # all_df[column_prefix + 'Adj_Close'] = df_index['Adj Close']

        all_df[column_prefix + '50ma'] = df_index['Adj Close'].rolling(window=50).mean()
        all_df[column_prefix + '125ma'] = df_index['Adj Close'].rolling(window=125).mean()
        all_df[column_prefix + '125ma_50ma_delta'] = all_df[column_prefix + '125ma'] - all_df[column_prefix + '50ma']
        all_df[column_prefix + '50ma_delta'] = df_index['Adj Close'] - all_df[column_prefix + '50ma']
        all_df[column_prefix + '125ma_delta'] = df_index['Adj Close'] - all_df[column_prefix + '125ma']

        bb = BollingerBands(df_index['Adj Close'])
        all_df[column_prefix + 'bb_upper'] = bb.bollinger_hband()
        all_df[column_prefix + 'bb_lower'] = bb.bollinger_lband()
        all_df[column_prefix + 'bb_upper_delta'] = all_df[column_prefix + 'bb_upper'] - df_index['Adj Close']
        all_df[column_prefix + 'bb_lower_delta'] = df_index['Adj Close'] - all_df[column_prefix + 'bb_lower']

        macd = MACD(df_index['Adj Close'])
        all_df[column_prefix + 'macd'] = macd.macd()

        rsi = RSIIndicator(df_index['Adj Close'])
        all_df[column_prefix + 'rsi'] = rsi.rsi()

        # Calculate 1, 3, 6, and 12 month momentum using adjusted close
        all_df[column_prefix + 'momentum_1m'] = momentum.roc(df_index['Adj Close'], window=20)
        all_df[column_prefix + 'momentum_3m'] = momentum.roc(df_index['Adj Close'], window=60)
        all_df[column_prefix + 'momentum_6m'] = momentum.roc(df_index['Adj Close'], window=120)
        # all_df[column_prefix + 'momentum_12m'] = momentum.roc(df_index['Adj Close'], window=240)

        volume_zero = (df_index['Volume'] == 0).sum()
        volume_percentage = volume_zero / len(df_index['Volume'])
        if volume_percentage <= 0.95:
            obv = OnBalanceVolumeIndicator(df_index['Adj Close'], df_index['Volume'])
            all_df[column_prefix + 'obv'] = obv.on_balance_volume()
            all_df[column_prefix + 'obv_log_return'] = np.log(all_df[column_prefix + 'obv']/all_df[column_prefix + 'obv'].shift(1))

        awesome_oscillator = AwesomeOscillatorIndicator(df_index['Adj Close'], df_index['Low'])
        all_df[column_prefix + 'awesome_oscillator'] = awesome_oscillator.awesome_oscillator()

        # all_df[column_prefix + 'return'] = df_index['Adj Close'].pct_change()
        all_df[column_prefix + 'log_return'] = np.log(df_index['Adj Close']/df_index['Adj Close'].shift(1))

        all_df[column_prefix + '50ma_log_return'] = np.log(all_df[column_prefix + '50ma']/all_df[column_prefix + '50ma'].shift(1))
        all_df[column_prefix + '125ma_log_return'] = np.log(all_df[column_prefix + '125ma']/all_df[column_prefix + '125ma'].shift(1))
        all_df[column_prefix + '125ma_50ma_delta'] = np.log(all_df[column_prefix + '125ma_50ma_delta']/all_df[column_prefix + '125ma_50ma_delta'].shift(1))

        all_df[column_prefix + 'oc_delta'] = df_index['Open'] - df_index['Close']
        all_df[column_prefix + 'hl_delta'] = df_index['High'] - df_index['Low']


        ticker_start_date_list.append(df_index.index.min())
        ticker_end_date_list.append(df_index.index.max())


    # Display the resulting DataFrame
    all_df.columns = all_df.columns.str.lower()

    start_date = max(ticker_start_date_list)
    end_date = min(ticker_end_date_list)
    print(start_date, end_date)
    all_df = all_df[all_df.index >= start_date]
    return all_df

def handle_missing_values(df):
    # Calculate the threshold based on the percentage
    threshold = len(df) * 0.975
    print(threshold)

    # Drop columns with more NaN values than the threshold
    df = df.dropna(axis=1, thresh=threshold, inplace=False)

    # Once confirming why there are NAs, proceed to drop
    df1 = df.dropna(axis=0, inplace=False)

    return df1

def create_X_sequences(X, time_steps=20):
    Xs = []
    for i in range(len(X) - time_steps):
        Xs.append(X.iloc[i:(i + time_steps)].values)
    return np.array(Xs)

def preprocess_new_data(tickers, years='25y'):
    # Download new data
    ticker_dict = yf_download(tickers, years)

    # Extract features
    df = extract_features(ticker_dict)

    # Handle missing values
    df = handle_missing_values(df)

    # Create sequences
    input_steps = 60
    Xs = create_X_sequences(df, input_steps)  # Assuming the first column of df as a dummy target variable

    return Xs

# Usage:
new_data = preprocess_new_data(ticker_list, '25y')

# Making predictions
predictions = model.predict(new_data)


class Predictions(BaseModel):
    prediction: str

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.get("/predict", response_model=Predictions)
def predict():
    try:
        new_data = preprocess_new_data(ticker_list, '25y')
        predictions = model.predict(new_data)
        latest_prediction = np.around(predictions[-1])
        action = 'BUY' if latest_prediction == 1 else 'SELL'
        return {'prediction': action}
    except Exception as e:
        return {'error': str(e)}

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)
