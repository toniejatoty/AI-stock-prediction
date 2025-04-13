import get_data
import model_LinearRegression
import model_RandomForest
import model_LSTM
from visualization import get_plot, show_all_historical_data


def get_predictions(
    days_in_future_to_predict, symbol, days_to_train, epochs, start_date,stop_check
):
    try:
        df_stockdata = get_data.get_stock_data(symbol, start_date)
    except Exception as e:
        raise ValueError(f"Probably you gave bad ticker: {symbol}")
    

    (
        LINEAR_result_of_testing_to_visualize,
        LINEAR_result_of_predictions,
        LINEAR_best_days_to_train,
    ) = model_LinearRegression.predict_stock_prices(
        df_stockdata, days_in_future_to_predict
    )
    LSTM_result_of_testing_to_visualize, LSTM_result_of_predictions = (
        model_LSTM.predict_stock_prices(
            df_stockdata,
            days_in_future_to_predict,
            days_to_train,
            epochs,
            stop_check
        )
    )

    fig_all = show_all_historical_data(df_stockdata)
    fig_linear = get_plot(
        df_stockdata,
        LINEAR_result_of_testing_to_visualize,
        LINEAR_best_days_to_train,
        LINEAR_result_of_predictions,
        days_in_future_to_predict,
    )
    fig_lstm = get_plot(
        df_stockdata,
        LSTM_result_of_testing_to_visualize,
        days_to_train,
        LSTM_result_of_predictions,
        days_in_future_to_predict,
    )
    return fig_all, fig_linear, fig_lstm
