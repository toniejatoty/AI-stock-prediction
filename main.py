import get_data
import model_LSTM2
import model_LinearRegression
import model_XGBRegressor
import model_LSTM1
from visualization import get_plot, show_all_historical_data


def get_predictions(
    charts,
    days_in_future_to_predict,
    symbol,
    start_date,
    days_to_train,
    XGB_params,
    epochs,
    loss_function,
    optimizer_name,
    learning_rate,
    batch_size,
    early_stopping,
    stop_check,
    progress_callback,
    lstm_layers
):
    try:
        df_stockdata = get_data.get_stock_data(symbol, start_date)
    except Exception as e:
        raise ValueError(f"{e}")
    
    days_to_train,days_in_future_to_predict, Status_main = validate_params(days_to_train,df_stockdata,days_in_future_to_predict)
    
    LINEAR_status=""
    XGBRegressor_status=""
    LSTM_status=""
    LSTM2_status=""
    fig_linear=None
    fig_xgb=None
    fig_lstm=None
    fig_lstm2=None
    if charts[0] == True:
        LINEAR_result_of_testing_to_visualize,LINEAR_result_of_predictions,LINEAR_best_days_to_train,LINEAR_score_of_training, LINEAR_status=(
            model_LinearRegression.predict_stock_prices(
            df_stockdata, days_in_future_to_predict,loss_function
        ))
        fig_linear = get_plot(
            df_stockdata,
            LINEAR_result_of_testing_to_visualize,
            LINEAR_best_days_to_train,
            LINEAR_result_of_predictions,
            days_in_future_to_predict,
            LINEAR_score_of_training,
            f"{symbol} Linear Regression"
        )


    if charts[1] == True:
        XGBRegressor_result_of_testing_to_visualize, XGBRegressor_result_of_predictions, XGBRegressor_score_of_training,XGBRegressor_status= (
            model_XGBRegressor.predict_stock_prices(
                df_stockdata,
                days_in_future_to_predict,
                days_to_train,
                XGB_params,
                loss_function,
                stop_check,
                progress_callback
                )
        )
        fig_xgb = get_plot(
            df_stockdata,
            XGBRegressor_result_of_testing_to_visualize,
            days_to_train,
            XGBRegressor_result_of_predictions,
            days_in_future_to_predict,
            XGBRegressor_score_of_training,
            f"{symbol} XGBRegression"
        )

    if charts[2] == True:
        LSTM_result_of_testing_to_visualize, LSTM_result_of_predictions, LSTM_score_of_training,LSTM_status= (
            model_LSTM1.predict_stock_prices(
                df_stockdata,
                days_in_future_to_predict,
                days_to_train,
                epochs,
                loss_function,
                optimizer_name,
                learning_rate,
                batch_size,
                early_stopping,
                stop_check,
                progress_callback,
                lstm_layers
            )
        )
        fig_lstm = get_plot(
            df_stockdata,
            LSTM_result_of_testing_to_visualize,
            days_to_train,
            LSTM_result_of_predictions,
            days_in_future_to_predict,
            LSTM_score_of_training,
            f"{symbol} LSTM"
        )
    if charts[3] == True:
        LSTM2_result_of_testing_to_visualize, LSTM2_result_of_predictions, LSTM2_score_of_training,LSTM2_status= (
            model_LSTM2.predict_stock_prices(
                df_stockdata,
                days_in_future_to_predict,
                days_to_train,
                epochs,
                loss_function,
                optimizer_name,
                learning_rate,
                batch_size,
                early_stopping,
                stop_check,
                progress_callback,
                lstm_layers
            )
        )
        fig_lstm2 = get_plot(
            df_stockdata,
            LSTM2_result_of_testing_to_visualize,
            days_to_train,
            LSTM2_result_of_predictions,
            days_in_future_to_predict,
            LSTM2_score_of_training,
            f"{symbol} LSTM2"
        )

    fig_all = show_all_historical_data(df_stockdata)


    return_status=""
    if Status_main !=None:
        return_status="main.py:"+Status_main+"\n"
    
    return_status+"Linear:"+LINEAR_status+"\n"+"XGBRegressor:"+XGBRegressor_status+"\n"+"LSTM1:"+LSTM_status +"\n"+"LSTM2:"+ LSTM2_status
    
    return fig_all, fig_linear, fig_xgb, fig_lstm, fig_lstm2, return_status

def validate_params(days_to_train,df_stockdata,days_in_future_to_predict):
    Status=None
    if df_stockdata.shape[0] <=2:
        raise ValueError(f"Please give wider Start date")
    if 2*days_to_train + days_in_future_to_predict > df_stockdata.shape[0]:
        proc = int(0.4 * df_stockdata.shape[0])
        days_to_train = max(1,proc)
        days_in_future_to_predict = max(1,int(0.1 * df_stockdata.shape[0]))
        Status = f"You provided days to predict + 2*days to train > Start date, thus i set: days to train:{days_to_train}, days to predict:{days_in_future_to_predict}  "
        
    return days_to_train,days_in_future_to_predict, Status
