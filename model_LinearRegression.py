import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import sys


def predict_stock_prices(df_org, days_in_future_to_predict, loss_function):
    Status="OK"
    int_max_value = int(sys.maxsize)
    max_size_df = df_org.shape[0]
    days_in_past_to_train = {}

    days_in_past_to_train.update({
            i: int_max_value
            for i in range(max_size_df - days_in_future_to_predict, 1000, -1000)
        })

    days_in_past_to_train.update({
            i: int_max_value
            for i in range(min(1000, max_size_df - days_in_future_to_predict), 200, -50)
        })

    days_in_past_to_train.update({
            i: int_max_value
            for i in range(min(200, max_size_df - days_in_future_to_predict), 50, -10)
        })
    days_in_past_to_train.update({i: int_max_value for i in range(50, 10, -5)})
    model = LinearRegression()
    for days_to_train in days_in_past_to_train:
        #if days_to_train > days_in_future_to_predict:
        predictions, y_test = get_prediction(
            df_org,
            days_to_train,
            days_in_future_to_predict,
            model,
            days_in_future_to_predict,
        )

        if loss_function == "mse":
            score = mean_squared_error(y_test, predictions)
        elif loss_function == "mae":
            score = mean_absolute_error(y_test, predictions)
        days_in_past_to_train[days_to_train] = score

    days_to_train = min(days_in_past_to_train, key=lambda k: days_in_past_to_train[k])
    score_of_training = days_in_past_to_train[days_to_train]
    # to visualize result of training
    result_of_testing_to_visualize, _ = get_prediction(
        df_org,
        days_to_train,
        days_in_future_to_predict,
        model,
        days_in_future_to_predict,
    )
    # to predict in future

    result_of_predictions, _ = get_prediction(
        df_org, days_to_train, 0, model, days_in_future_to_predict
    )
    return (
        result_of_testing_to_visualize,
        result_of_predictions,
        days_to_train,
        score_of_training,
        Status
    )


def get_prediction(
    df_org, days_to_train, days_in_future_to_test_model, model, days_in_future_to_predict
):
    df = df_org.tail(days_to_train + days_in_future_to_test_model).copy()
    X_train = np.array(range(0, days_to_train)).reshape(-1, 1)
    X_test = np.array(
        range(days_to_train, days_to_train + days_in_future_to_predict)
    ).reshape(-1, 1)
    y_train = df["Close"].iloc[0:days_to_train]
    y_test = df["Close"].iloc[days_to_train : days_to_train + days_in_future_to_test_model]
    model.fit(X_train, y_train)
    return model.predict(X_test), y_test
