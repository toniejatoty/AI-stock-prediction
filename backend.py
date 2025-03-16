import pandas as pd
import get_data
import visualization 
import predictions



days_in_future_to_predict=30# user is giving this arg
symbol = "AAPL"#user is giving this arg
df_stockdata = get_data.get_stock_data(symbol)

result_of_testing_to_visualize,   result_of_predictions,   best_days_to_train = predictions.predict_stock_prices(df_stockdata,days_in_future_to_predict)
visualization.plot_predictions(df_stockdata, result_of_testing_to_visualize , best_days_to_train,  result_of_predictions, days_in_future_to_predict)

