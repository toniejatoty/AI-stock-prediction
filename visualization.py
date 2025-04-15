import pandas as pd
import matplotlib.dates as mdates
import matplotlib.pyplot as plt


def get_plot(
    df,
    result_of_testing,
    days_to_train,
    predictions,
    days_in_future_to_predict,
    score_of_training,
    result_of_testing_on_XTest=None
):
    df = df.tail(days_in_future_to_predict * 2)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df.index, df["Close"], label="Real price")
    dates_to_visualize_test = df.index[-days_in_future_to_predict:]

    ax.plot(
        dates_to_visualize_test,
        result_of_testing,
        label="result_of_testing",
    )
    if result_of_testing_on_XTest is not None:
        dates_to_visualize_test = df.index[-days_in_future_to_predict:]

        ax.plot(
        dates_to_visualize_test,
        result_of_testing_on_XTest,
        label="result_of_testing_on_xtest",
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
        title=f"Number of days to train: {days_to_train}\nNumber of days to predict: {days_in_future_to_predict}\nScore:{score_of_training:.4f}"
    )
    plt.tight_layout()
    return fig


def show_all_historical_data(df):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df.index, df["Close"], label="Real Price")
    ax.set_title(f"Historic stock price ({df.index[0].date()} - {df.index[-1].date()})")
    ax.set_xlabel("Date")
    ax.set_ylabel("Stock price")

    ax.legend()
    plt.tight_layout()
    return fig
