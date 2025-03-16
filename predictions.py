import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import sys

def predict_stock_prices(df_org,days_in_future_to_predict):
    int_max_value = int(sys.maxsize)
    days_in_past_to_train = { 1000:int_max_value, 750:int_max_value, 500:int_max_value,300:int_max_value, 200:int_max_value,
                              150:int_max_value, 100:int_max_value, 80:int_max_value, 60:int_max_value,
                              50:int_max_value, 40:int_max_value, 30:int_max_value, 20:int_max_value, 10:int_max_value}
    model = LinearRegression()
    for days_to_train in days_in_past_to_train:
        if days_to_train>days_in_future_to_predict:
            
            

            df=df_org.tail(days_to_train+days_in_future_to_predict) 
            X_train = np.array(range(0, days_to_train)).reshape(-1,1)
            X_test = np.array(range(days_to_train, days_to_train+days_in_future_to_predict)).reshape(-1,1)
            y_train = df['Close'].iloc[0:days_to_train]
            y_test = df['Close'].iloc[days_to_train:days_to_train+days_in_future_to_predict]
            
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            mse = mean_squared_error(predictions, y_test)
            days_in_past_to_train[days_to_train] = mse

    days_to_train = min(days_in_past_to_train, key=lambda k: days_in_past_to_train[k])
    
    #to visualize result of training
    df=df_org.tail(days_to_train+days_in_future_to_predict)
    X_train = np.array(range(0, days_to_train)).reshape(-1,1)
    X_test = np.array(range(days_to_train, days_to_train+days_in_future_to_predict)).reshape(-1,1)
    y_train = df['Close'].iloc[0:days_to_train]
    model.fit(X_train, y_train)
    result_of_testing_to_visualize = model.predict(X_test)
    #to predict in future

    df=df_org.tail(days_to_train)
    X_train = np.array(range(0, days_to_train)).reshape(-1,1)
    y_train = df['Close']
    model.fit(X_train, y_train) 
    future_days = np.array(range(days_to_train, days_to_train + days_in_future_to_predict)).reshape(-1,1)
    result_of_predictions = model.predict(future_days)
    return result_of_testing_to_visualize, result_of_predictions , days_to_train

