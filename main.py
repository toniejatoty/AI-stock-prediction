import get_data
import visualization
import model_LinearRegression
import model_RandomForest
import model_LSTM

def main():
    days_in_future_to_predict = 30  # user is giving this arg
    symbol = "INTC"  # user is giving this arg
    df_stockdata = get_data.get_stock_data(symbol)
    result_of_testing_to_visualize, result_of_predictions, best_days_to_train = (
        model_LinearRegression.predict_stock_prices(
            df_stockdata, days_in_future_to_predict
        )
    )

    visualization.show_plot(
        df_stockdata,
        result_of_testing_to_visualize,
        best_days_to_train,
        result_of_predictions,
        days_in_future_to_predict,
    )

    # result_of_testing_to_visualize, result_of_predictions, best_days_to_train = (
    #     model_RandomForest.predict_stock_prices(
    #         df_stockdata, days_in_future_to_predict
    #     )
    # )

    # visualization.show_plot(
    #     df_stockdata,
    #     result_of_testing_to_visualize,
    #     best_days_to_train,
    #     result_of_predictions,
    #     days_in_future_to_predict,
    # )

    result_of_testing_to_visualize, result_of_predictions, best_days_to_train = (
        model_LSTM.predict_stock_prices(
            df_stockdata, days_in_future_to_predict
        )
    )

    visualization.show_plot(
        df_stockdata,
        result_of_testing_to_visualize,
        best_days_to_train,
        result_of_predictions,
        days_in_future_to_predict,
    )

if __name__ == "__main__":
    main()
