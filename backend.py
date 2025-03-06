from alpha_vantage.timeseries import TimeSeries
from alpha_vantage.fundamentaldata import FundamentalData
import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt
API_KEY = "OYOR7X0J85IO3RDD"

def get_stock_data(symbol):
    ts = TimeSeries(key=API_KEY, output_format='pandas')
    data,meta = ts.get_daily(symbol=symbol, outputsize='compact')
    return data
def get_stock_overview(symbol):
    fd = FundamentalData(key=API_KEY, output_format='pandas')
    data, meta = fd.get_company_overview(symbol=symbol)
    return data


def predict_stock_prices(df):
    df = df.sort_index() 
    df = df.tail(30)  
    X = np.array(range(len(df))).reshape(-1, 1)  
    y = df['4. close'].values 
    model = LinearRegression()
    model.fit(X, y)
    future_X = np.array(range(len(df), len(df) + 5)).reshape(-1, 1)
    predicted_prices = model.predict(future_X)
    return predicted_prices





def plot_predictions(df, predictions):
    df = df.sort_index()
    df = df.tail(30)
    plt.figure(figsize=(10, 5))
    plt.plot(df.index, df['4. close'], label="Rzeczywiste ceny", marker='o')
    future_dates = pd.date_range(df.index[-1], periods=6, freq='D')[1:] 
    plt.plot(future_dates, predictions, label="Prognoza", marker='o', linestyle="dashed")
    plt.xlabel("Data")
    plt.ylabel("Cena akcji")
    plt.title("Predykcja cen akcji")
    plt.legend()
    plt.show()


symbol = "AAPL"
df_stockdata = get_stock_data(symbol)
df_fundamentdata = get_stock_overview(symbol)
print(df_fundamentdata)
#print(df.head())
predictions = predict_stock_prices(df_stockdata)
#print("Prognozowane ceny na kolejne dni:", predictions)
plot_predictions(df_stockdata, predictions)    