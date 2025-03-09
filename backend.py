import yfinance as yf
import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt
import requests

def get_stock_data(symbol):
    company = yf.Ticker(symbol)
    history_prices = company.history(period="max")
    return history_prices

def get_income_statement(symbol):
    company = yf.Ticker(symbol)
    quarterly_financials = company.quarterly_financials
    return quarterly_financials

def merge(df_stockdata, income_statement):
    df_stockdata['Quarter'] = pd.PeriodIndex(df_stockdata.index, freq='Q')
    income_statement = income_statement.T
    income_statement['Quarter'] = pd.PeriodIndex(income_statement.index, freq='Q')
    income_statement['Quarter'] = income_statement['Quarter'] + 1
    date_index = df_stockdata.index
    df_stockdata.merge(income_statement, on='Quarter', how='left')
    df_stockdata.index = date_index
    return df_stockdata


def predict_stock_prices(df):
    df = df.sort_index() 
    df = df.tail(30)  
    X = np.array(range(len(df))).reshape(-1, 1)  
    y = df['Close'].values 
    model = LinearRegression()
    model.fit(X, y)
    future_X = np.array(range(len(df), len(df) + 5)).reshape(-1, 1)
    predicted_prices = model.predict(future_X)
    return predicted_prices





def plot_predictions(df, predictions):
    df = df.sort_index()
    df = df.tail(30)
    print(df)
    plt.figure(figsize=(10, 5))
    plt.plot(df.index, df['Close'], label="Real price", marker='o')
    future_dates = pd.date_range(df.index[-1], periods=6, freq='D')[1:] 
    plt.plot(future_dates, predictions, label="Prediction", marker='o', linestyle="dashed")
    plt.xlabel("Date")
    plt.ylabel("Stock price")
    plt.title("Stock price predition")
    plt.legend()
    plt.show()


symbol = "AAPL"
df_stockdata = get_stock_data(symbol)
income_statement=get_income_statement(symbol)
df_stock_full_info_with_nan = merge(df_stockdata, income_statement)
df_stock_full_info = df_stock_full_info_with_nan
predictions = predict_stock_prices(df_stock_full_info)
plot_predictions(df_stock_full_info, predictions)
pd.set_option('display.max_columns', None)  # Wyświetla wszystkie kolumny
print(df_stock_full_info)
print(df_stock_full_info.dtypes)
