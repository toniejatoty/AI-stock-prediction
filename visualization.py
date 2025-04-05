import pandas as pd
import matplotlib.dates as mdates
import matplotlib.pyplot as plt


def show_plot(
    df, result_of_testing, days_to_train, predictions, days_in_future_to_predict
):
    df = df.tail(days_to_train)

    fig, ax = plt.subplots(figsize=(12, 6))
    # Poprawione: używamy ax.plot() zamiast ax.plt.plot()
    ax.plot(df.index, df["Close"], label="Real price")
    dates_to_visualize_test = df.index[-days_in_future_to_predict:]

    ax.plot(
        dates_to_visualize_test,
        result_of_testing,
        label="result_of_testing",
        marker="o",
    )

    last_date = df.index[-1]
    future_dates = pd.date_range(
        last_date + pd.Timedelta(days=1), periods=days_in_future_to_predict, freq="B"
    )
    ax.plot(future_dates, predictions, label="Prediction", marker="o")
    ax.set_xlabel("Date")
    ax.set_ylabel("Stock price")
    ax.set_title("Stock price prediction")
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.legend(
        title=f"Number of days to train: {days_to_train}\nNumber of days to predict: {days_in_future_to_predict}"
    )
    plt.tight_layout()
    return fig


def show_all_historical_data(df):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df.index, df["Close"], label="Cena zamknięcia", linewidth=2)
    ax.set_title(
        f"Historyczne ceny akcji ({df.index[0].date()} - {df.index[-1].date()})", pad=20
    )
    ax.set_xlabel("Data")
    ax.set_ylabel("Cena ($)")
    ax.grid(True, linestyle="--", alpha=0.7)
    ax.legend()
    plt.tight_layout()
    return fig
