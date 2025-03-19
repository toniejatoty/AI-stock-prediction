import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import sys


def predict_stock_prices(df_org, days_in_future_to_predict):
    int_max_value = int(sys.maxsize)
    max_size_df = df_org.shape[0]
    days_in_past_to_train={}
    if max_size_df >1000:
        days_in_past_to_train.update( {i: int_max_value for i in range(max_size_df-days_in_future_to_predict, 1000, -1000)})
    if max_size_df>200:
        days_in_past_to_train.update({i: int_max_value for i in range(min(1000,max_size_df-days_in_future_to_predict), 200, -50)})
    if max_size_df>50:
        days_in_past_to_train.update({i: int_max_value for i in range(min(200,max_size_df-days_in_future_to_predict), 50, -10)})
    days_in_past_to_train.update({i: int_max_value for i in range(50, 10, -5)})
    model = RandomForestRegressor()
    for days_to_train in days_in_past_to_train:
        if days_to_train > days_in_future_to_predict:

            predictions,y_test = get_prediction(df_org,days_to_train, days_in_future_to_predict, model,days_in_future_to_predict)
            
            mse = mean_squared_error(predictions, y_test)
            days_in_past_to_train[days_to_train] = mse

    days_to_train = min(days_in_past_to_train, key=lambda k: days_in_past_to_train[k])

    # to visualize result of training
    result_of_testing_to_visualize, _ =get_prediction(df_org,days_to_train, days_in_future_to_predict, model,days_in_future_to_predict)
    # to predict in future

    result_of_predictions, _ = get_prediction(df_org,days_to_train, 0, model,days_in_future_to_predict)
    return result_of_testing_to_visualize, result_of_predictions, days_to_train




def get_prediction(df_org, days_to_train, days_to_test_model, model, days_in_future_to_predict):
    df = df_org.tail(days_to_train + days_to_test_model)
    X_train = np.array(range(0, days_to_train)).reshape(-1, 1)
    X_test = np.array(
        range(days_to_train, days_to_train + days_in_future_to_predict)
    ).reshape(-1, 1)
    y_train = df["Close"].iloc[0:days_to_train]
    y_test = df["Close"].iloc[
                days_to_train : days_to_train + days_in_future_to_predict
            ]
    model.fit(X_train, y_train)
    return model.predict(X_test), y_test