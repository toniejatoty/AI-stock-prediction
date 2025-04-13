import pandas as pd
import get_data
import model_LinearRegression
import model_RandomForest
import model_LSTM
from visualization import get_plot, show_all_historical_data


def get_predictions(
    days_in_future_to_predict,
    symbol,
    start_date,
    days_to_train,
    epochs,
    loss_function,
    optimizer_name,
    learning_rate,
    batch_size,
    stop_check,
    progress_callback,
):
    try:
        df_stockdata = get_data.get_stock_data(symbol, start_date)
    except Exception as e:
        raise ValueError(f"Probably you gave bad ticker: {symbol}")
    
    days_to_train,days_in_future_to_predict = validate_params(days_to_train,df_stockdata,days_in_future_to_predict)
    
    (
        LINEAR_result_of_testing_to_visualize,
        LINEAR_result_of_predictions,
        LINEAR_best_days_to_train,
        LINEAR_score_of_training
    ) = model_LinearRegression.predict_stock_prices(
        df_stockdata, days_in_future_to_predict,loss_function
    )
    LSTM_result_of_testing_to_visualize, LSTM_result_of_predictions, LSTM_score_of_training= (
        model_LSTM.predict_stock_prices(
            df_stockdata,
            days_in_future_to_predict,
            days_to_train,
            epochs,
            loss_function,
            optimizer_name,
            learning_rate,
            batch_size,
            stop_check,
            progress_callback,
        )
    )

    fig_all = show_all_historical_data(df_stockdata)
    fig_linear = get_plot(
        df_stockdata,
        LINEAR_result_of_testing_to_visualize,
        LINEAR_best_days_to_train,
        LINEAR_result_of_predictions,
        days_in_future_to_predict,
        LINEAR_score_of_training
    )
    fig_lstm = get_plot(
        df_stockdata,
        LSTM_result_of_testing_to_visualize,
        days_to_train,
        LSTM_result_of_predictions,
        days_in_future_to_predict,
        LSTM_score_of_training
    )
    return fig_all, fig_linear, fig_lstm

def validate_params(days_to_train,df_stockdata,days_in_future_to_predict):
    if days_to_train + days_in_future_to_predict > df_stockdata.shape[0]:
        proc = int(0.8 * df_stockdata.shape[0])
        days_to_train = proc-1
        days_in_future_to_predict = df_stockdata.shape[0] -proc
        
    return days_to_train,days_in_future_to_predict

# def update_progress(epoch, total_epochs):
#     return
# fig_all, fig_linear, fig_lstm = get_predictions(
#             30,
#             "AAPL",
#             pd.to_datetime(20180310),
#             10,
#             1,
#             "mae",
#             "adam",
#             0.1,
#             32,
#             lambda: True,
#             update_progress
#         )