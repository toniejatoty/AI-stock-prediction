import numpy as np
import pandas as pd

import matplotlib.pyplot as plt


def show_plot(
    df, result_of_testing, days_to_train, predictions, days_in_future_to_predict
):
    plt.figure(figsize=(10, 5))
    plt.plot(df['Close'])
    plt.show()
    df = df.tail(days_to_train)
    plt.figure(figsize=(10, 5))

    plt.plot(df.index, df["Close"], label="Real price")
    dates_to_visualize_test = df.index[-days_in_future_to_predict:]

    plt.plot(
        dates_to_visualize_test,
        result_of_testing,
        label="result_of_testing",
        marker="o",
    )
    last_date = last_date = df.index[-1]
    future_dates = pd.date_range(
        last_date + pd.Timedelta(days=1), periods=days_in_future_to_predict, freq="B"
    )
    plt.plot(future_dates, predictions, label="Prediction", marker="o")
    plt.xlabel("Date")
    plt.ylabel("Stock price")
    plt.title("Stock price predition")
    plt.legend(
        title=f"Number of days to train: {days_to_train}\nNumber of days to predict: {days_in_future_to_predict}"
    )
    plt.show()
