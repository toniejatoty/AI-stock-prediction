import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

def predict_stock_prices(df_org, days_in_future_to_predict, loss_function):
    Status="OK"
    
    #to test model 
    days_to_train_testing = get_best_num_of_days_to_train(df_org[0:-days_in_future_to_predict], days_in_future_to_predict,loss_function)
    
    result_of_testing=get_pred(df_org.iloc[0:-days_in_future_to_predict],days_to_train_testing,days_in_future_to_predict)
    score = get_score(result_of_testing, df_org['Close'].iloc[-days_in_future_to_predict:], loss_function)
    
    days_to_train_pred = get_best_num_of_days_to_train(df_org, days_in_future_to_predict,loss_function)
    #to predict future
    future_pred=get_pred(df_org,days_to_train_pred,days_in_future_to_predict)
    print(days_to_train_testing)
    return ( result_of_testing,future_pred,days_to_train_pred,score, Status )



def get_test(df_org, days_to_train, days_in_future_to_predict):
    model = LinearRegression()
    df=df_org.copy()
    X_train = np.array(range(0, days_to_train)).reshape(-1, 1)
    y_train = df["Close"].iloc[-(days_to_train+days_in_future_to_predict): -days_in_future_to_predict]
    X_test = np.array(range(days_to_train, days_to_train+days_in_future_to_predict)).reshape(-1, 1)
    y_test = df["Close"].iloc[-days_in_future_to_predict:]
    model.fit(X_train, y_train)

    return model.predict(X_test), y_test

def get_pred(df_org, days_to_train, days_in_future_to_predict):
    model = LinearRegression()
    df=df_org.copy()
    X_train = np.array(range(0, days_to_train)).reshape(-1, 1)
    y_train = df["Close"].iloc[-(days_to_train):]
    model.fit(X_train, y_train)

    X_future=np.array(range(days_to_train, days_to_train+days_in_future_to_predict)).reshape(-1,1)
    return model.predict(X_future)

def get_score(y_test, predictions, loss_function):
        if loss_function == "mse":
            score = mean_squared_error(y_test, predictions)
        elif loss_function == "mae":
            score = mean_absolute_error(y_test, predictions)
        return score

def get_best_num_of_days_to_train(df_org, days_in_future_to_predict,loss_function):
    infinity = float('inf')
    max_size_df = df_org.shape[0]
    days_in_past_to_train = {}
    days_in_past_to_train.update({i: infinity for i in range(max_size_df - days_in_future_to_predict, 1000, -1000)})
    days_in_past_to_train.update({i: infinity for i in range(min(1000, max_size_df - days_in_future_to_predict), 200, -50)})
    days_in_past_to_train.update({i: infinity for i in range(min(200, max_size_df - days_in_future_to_predict), 50, -10)})
    days_in_past_to_train.update({i: infinity for i in range(min(50, max_size_df - days_in_future_to_predict), 10, -5)})
    days_in_past_to_train.update({i: infinity for i in range(min(10, max_size_df - days_in_future_to_predict), 0, -1)})
    for days_to_train in days_in_past_to_train:
        x_train_pred, y_train = get_test(df_org,days_to_train,days_in_future_to_predict)

        score = get_score(x_train_pred, y_train, loss_function)
        days_in_past_to_train[days_to_train] = score
    days_to_train = min(days_in_past_to_train, key=lambda k: days_in_past_to_train[k])

    return days_to_train