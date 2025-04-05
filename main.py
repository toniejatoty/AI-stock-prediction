import get_data
import visualization
import model_LinearRegression
import model_RandomForest
import model_LSTM
from visualization import show_plot, show_all_historical_data

def get_predictions(days_in_future_to_predict, symbol="AAPL"):
    #days_in_future_to_predict = 30  # user is giving this arg
    #symbol = "SPY"  # user is giving this arg
    df_stockdata = get_data.get_stock_data(symbol)
    LINEAR_result_of_testing_to_visualize, LINEAR_result_of_predictions, LINEAR_best_days_to_train = (
        model_LinearRegression.predict_stock_prices(
            df_stockdata, days_in_future_to_predict
        )
    )
    #visualization.show_all_historical_data(df_stockdata)
    LSTM_result_of_testing_to_visualize, LSTM_result_of_predictions, LSTM_best_days_to_train = (
        model_LSTM.predict_stock_prices(
            df_stockdata, days_in_future_to_predict
        )
    )

    fig_all = show_all_historical_data(df_stockdata)  
    fig_linear = show_plot(
        df_stockdata,
        LINEAR_result_of_testing_to_visualize,
        LINEAR_best_days_to_train,
        LINEAR_result_of_predictions,
        days_in_future_to_predict
    )
    fig_lstm = show_plot(
        df_stockdata,
        LSTM_result_of_testing_to_visualize,
        LSTM_best_days_to_train,
        LSTM_result_of_predictions,
        days_in_future_to_predict
    )
    return fig_all,fig_linear,fig_lstm
