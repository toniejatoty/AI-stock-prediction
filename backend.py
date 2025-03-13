import yfinance as yf
import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt
import requests
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

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
    df_stockdata=df_stockdata.merge(income_statement, on='Quarter', how='left')
    df_stockdata.index = date_index
    return df_stockdata

def delete_Nan_and_useless_col_in_dataframe(df_stockdata):
    useless_col = ["Tax Rate For Calcs","Quarter", "Tax Effect Of Unusual Items", "Normalized EBITDA", "Normalized Income","Net Interest Income", "Interest Expense",
                    "Interest Income", "Interest Expense Non Operating", "Interest Income Non Operating", "Other Income Expense", "Other Non Operating Income Expenses",
                    "Net Non Operating Interest Income Expense", "Diluted Average Shares", "Basic Average Shares","Net Income From Continuing Operation Net Minority Interest",
                    "Net Income From Continuing And Discontinued Operation", "Net Income Including Noncontrolling Interests", "Net Income Continuous Operations", 
                    "Diluted NI Availto Com Stockholders", "Net Income Common Stockholders", "Tax Provision", "Pretax Income"]
    df_stockdata = df_stockdata.drop(columns=useless_col)
    df_stockdata = df_stockdata.dropna(axis=1, how='all')
    df_stockdata=df_stockdata.replace(np.nan, 0)
    return df_stockdata

def predict_stock_prices(df_org,days_in_future_to_predict):
    days_in_past_to_train = { 1000:0, 500:0, 200:0, 100:0,  60:0, 50:0, 40:0, 30:0, 20:0, 10:0}
    df_org = df_org.sort_index()
    for days in days_in_past_to_train:
        proc = 0.8
        days_to_train = int(proc * days)

        df=df_org.tail(days) 
        X_train = np.array(range(0, int(days_to_train))).reshape(-1,1)
        X_test = np.array(range(days_to_train, days)).reshape(-1,1)
        y_train = df['Close'].iloc[0:days_to_train]
        y_test = df['Close'].iloc[days_to_train:]
        model = LinearRegression()
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        mse = mean_squared_error(predictions, y_test)
        days_in_past_to_train[days] = mse
        future_X = np.array(range(len(df), len(df) + days_in_future_to_predict)).reshape(-1, 1)
        predicted_prices = model.predict(future_X)
    best_days = min(days_in_past_to_train, key=lambda k: days_in_past_to_train[k])
    print(f"najlepszy wynik dla past days:{best_days}")
    df=df_org.tail(best_days)
    X_train = np.array(range(0, best_days)).reshape(-1,1)
    y_train = df['Close']
    model.fit(X_train, y_train) 
    future_days = np.array(range(0, days_in_future_to_predict)).reshape(-1,1)
    return model.predict(future_days)





def plot_predictions(df, predictions, days_in_future_to_predict):
    df = df.sort_index()
    df=df.tail(30)
    plt.figure(figsize=(10, 5))
    plt.plot(df.index, df['Close'], label="Real price", marker='o')
    future_dates = pd.date_range(df.index[-1], periods=days_in_future_to_predict+1, freq='D')[1:] 
    plt.plot(future_dates, predictions, label="Prediction", marker='o', linestyle="dashed")
    plt.xlabel("Date")
    plt.ylabel("Stock price")
    plt.title("Stock price predition")
    plt.legend()
    plt.show()

days_in_future_to_predict=10# user is giving whis arg
symbol = "AAPL"#user is giving this arg
df_stockdata = get_stock_data(symbol)
income_statement=get_income_statement(symbol)
df_stock_full_info_with_nan = merge(df_stockdata, income_statement)
df_stock_full_info = delete_Nan_and_useless_col_in_dataframe (df_stock_full_info_with_nan)
predictions = predict_stock_prices(df_stock_full_info,days_in_future_to_predict)
plot_predictions(df_stock_full_info, predictions, days_in_future_to_predict)
pd.set_option('display.max_columns', None)  # Wyświetla wszystkie kolumny

